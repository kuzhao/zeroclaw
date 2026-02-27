//! Azure OpenAI provider.
//!
//! Speaks the same Chat Completions JSON protocol as the core OpenAI provider
//! but adapts two Azure-specific concerns:
//!
//! 1. **URL shape** – `{endpoint}/openai/deployments/{model}/chat/completions?api-version={version}`
//! 2. **Authentication** – three credential modes, resolved in priority order:
//!    - **API key** (`api-key: <key>` header) when `AZURE_OPENAI_API_KEY` is set
//!      (or the `api_key` field in config.toml / `api_key` CLI flag).
//!    - **Service principal** (`Authorization: Bearer <token>`) when
//!      `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET` are all set.
//!    - **Managed Identity** (`Authorization: Bearer <token>`) via the IMDS endpoint.
//!      An optional user-assigned identity can be selected with `AZURE_CLIENT_ID`.
//!
//! The Azure AD access token is cached in memory and refreshed automatically
//! before it expires.
//!
//! Configuration example (`config.toml`):
//! ```toml
//! default_provider = "azure-openai"
//! default_model    = "gpt-4o"          # deployment name
//! api_url          = "https://myresource.openai.azure.com"
//! api_key          = "my-azure-openai-api-key"   # omit for Azure AD auth
//! ```
//!
//! Or via environment variables:
//! ```sh
//! AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com
//! AZURE_OPENAI_API_KEY=<key>           # key-based auth
//! # — or — for service principal:
//! AZURE_TENANT_ID=<tenant>
//! AZURE_CLIENT_ID=<app-client-id>
//! AZURE_CLIENT_SECRET=<secret>
//! # — or — for managed identity (no additional vars needed) —
//! ```

use crate::providers::traits::{
    ChatMessage, ChatRequest as ProviderChatRequest, ChatResponse as ProviderChatResponse,
    Provider, ProviderCapabilities, TokenUsage, ToolCall as ProviderToolCall,
};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

// ── Constants ────────────────────────────────────────────────────────────────

/// Default Azure OpenAI REST API version.
const DEFAULT_API_VERSION: &str = "2024-10-21";

/// OAuth scope for Azure Cognitive Services.
const AZURE_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

/// Azure IMDS token endpoint (Managed Identity).
const IMDS_TOKEN_URL: &str =
    "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01";

/// Refresh an AD token this many seconds before its reported expiry.
const TOKEN_REFRESH_BEFORE_EXPIRY_SECS: u64 = 300;

// ── Request / response types (OpenAI Chat Completions JSON schema) ────────────
//
// These are structurally identical to the types in openai.rs; they are
// redeclared here so the azure_openai module remains self-contained.

#[derive(Debug, Serialize)]
struct SimpleMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct SimpleChatRequest {
    model: String,
    messages: Vec<SimpleMessage>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct SimpleChatResponse {
    choices: Vec<SimpleChoice>,
}

#[derive(Debug, Deserialize)]
struct SimpleChoice {
    message: SimpleResponseMessage,
}

#[derive(Debug, Deserialize)]
struct SimpleResponseMessage {
    #[serde(default)]
    content: Option<String>,
}

// ── Native tool-calling types ────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct NativeChatRequest {
    model: String,
    messages: Vec<NativeMessage>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<NativeToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
}

#[derive(Debug, Serialize)]
struct NativeMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<NativeToolCall>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeToolSpec {
    #[serde(rename = "type")]
    kind: String,
    function: NativeToolFunctionSpec,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeToolFunctionSpec {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    function: NativeFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct NativeFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct NativeChatResponse {
    choices: Vec<NativeChoice>,
    #[serde(default)]
    usage: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct NativeChoice {
    message: NativeResponseMessage,
}

#[derive(Debug, Deserialize)]
struct NativeResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<NativeToolCall>>,
}

// ── Azure AD token response ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct AzureTokenResponse {
    access_token: Option<String>,
    /// Seconds until expiry (client-credentials flow).
    #[serde(default)]
    expires_in: Option<u64>,
    /// Absolute Unix timestamp of expiry (managed identity flow).
    #[serde(default)]
    expires_on: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    error_description: Option<String>,
}

// ── Credential source ────────────────────────────────────────────────────────

/// How the provider authenticates with Azure OpenAI.
enum AzureCredentialSource {
    /// Static API key — sent as `api-key: <key>` header.
    ApiKey(String),
    /// Azure AD token via service-principal client-credentials flow.
    ClientSecret {
        tenant_id: String,
        client_id: String,
        client_secret: String,
    },
    /// Azure AD token via Managed Identity (IMDS).
    ManagedIdentity {
        /// User-assigned identity client ID, or `None` for system-assigned.
        client_id: Option<String>,
    },
}

/// Cached Azure AD bearer token.
struct CachedToken {
    value: String,
    /// When the cached token should be considered stale and refreshed.
    refresh_after: Instant,
}

// ── Provider struct ───────────────────────────────────────────────────────────

/// Azure OpenAI provider.
pub struct AzureOpenAiProvider {
    /// Base endpoint, e.g. `https://myresource.openai.azure.com`.
    endpoint: String,
    /// Azure OpenAI REST API version string.
    api_version: String,
    /// How to obtain credentials.
    source: AzureCredentialSource,
    /// In-memory cache for Azure AD access tokens.
    token_cache: Mutex<Option<CachedToken>>,
}

impl AzureOpenAiProvider {
    /// Create a provider, resolving credentials from `key_override` and/or
    /// environment variables.
    ///
    /// Resolution order for the endpoint:
    ///   1. `endpoint_override` parameter
    ///   2. `AZURE_OPENAI_ENDPOINT` environment variable
    ///
    /// Resolution order for credentials:
    ///   1. `key_override` (non-empty) → API key
    ///   2. `AZURE_OPENAI_API_KEY` env var → API key
    ///   3. `AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET` → service principal
    ///   4. Managed Identity (with optional `AZURE_CLIENT_ID`)
    pub fn new(endpoint_override: Option<&str>, key_override: Option<&str>) -> Self {
        let endpoint = endpoint_override
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .or_else(|| read_env("AZURE_OPENAI_ENDPOINT"))
            .unwrap_or_default();

        let api_version =
            read_env("AZURE_OPENAI_API_VERSION").unwrap_or_else(|| DEFAULT_API_VERSION.to_string());

        // Resolve credential source.
        let source = Self::resolve_source(key_override);

        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_version,
            source,
            token_cache: Mutex::new(None),
        }
    }

    fn resolve_source(key_override: Option<&str>) -> AzureCredentialSource {
        // 1. Explicit key override.
        if let Some(key) = key_override.map(str::trim).filter(|s| !s.is_empty()) {
            return AzureCredentialSource::ApiKey(key.to_string());
        }

        // 2. AZURE_OPENAI_API_KEY env var.
        if let Some(key) = read_env("AZURE_OPENAI_API_KEY") {
            return AzureCredentialSource::ApiKey(key);
        }

        // 3. Service principal credentials.
        if let (Some(tenant_id), Some(client_id), Some(client_secret)) = (
            read_env("AZURE_TENANT_ID"),
            read_env("AZURE_CLIENT_ID"),
            read_env("AZURE_CLIENT_SECRET"),
        ) {
            return AzureCredentialSource::ClientSecret {
                tenant_id,
                client_id,
                client_secret,
            };
        }

        // 4. Managed Identity (optionally user-assigned).
        AzureCredentialSource::ManagedIdentity {
            client_id: read_env("AZURE_CLIENT_ID"),
        }
    }

    /// Build the Chat Completions URL for the given model/deployment.
    fn chat_completions_url(&self, model: &str) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint, model, self.api_version
        )
    }

    /// Obtain the current credential value and the auth header name.
    ///
    /// For API key auth returns `("api-key", key)`.
    /// For AD token auth returns `("Authorization", "Bearer {token}")`.
    async fn auth_header(&self) -> anyhow::Result<(&'static str, String)> {
        match &self.source {
            AzureCredentialSource::ApiKey(key) => {
                anyhow::ensure!(
                    !self.endpoint.is_empty(),
                    "Azure OpenAI endpoint not set. Set AZURE_OPENAI_ENDPOINT or api_url in config.toml."
                );
                Ok(("api-key", key.clone()))
            }
            AzureCredentialSource::ClientSecret {
                tenant_id,
                client_id,
                client_secret,
            } => {
                anyhow::ensure!(
                    !self.endpoint.is_empty(),
                    "Azure OpenAI endpoint not set. Set AZURE_OPENAI_ENDPOINT or api_url in config.toml."
                );
                let token = self
                    .cached_or_fetch_client_secret_token(tenant_id, client_id, client_secret)
                    .await?;
                Ok(("Authorization", format!("Bearer {token}")))
            }
            AzureCredentialSource::ManagedIdentity { client_id } => {
                anyhow::ensure!(
                    !self.endpoint.is_empty(),
                    "Azure OpenAI endpoint not set. Set AZURE_OPENAI_ENDPOINT or api_url in config.toml."
                );
                let token = self
                    .cached_or_fetch_managed_identity_token(client_id.as_deref())
                    .await?;
                Ok(("Authorization", format!("Bearer {token}")))
            }
        }
    }

    /// Return a cached AD token if still fresh; otherwise acquire a new one
    /// using the service-principal client-credentials flow.
    async fn cached_or_fetch_client_secret_token(
        &self,
        tenant_id: &str,
        client_id: &str,
        client_secret: &str,
    ) -> anyhow::Result<String> {
        {
            let guard = self.token_cache.lock().await;
            if let Some(cached) = guard.as_ref() {
                if Instant::now() < cached.refresh_after {
                    return Ok(cached.value.clone());
                }
            }
        }

        let token_url = format!("https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token");

        let client = oauth_http_client();
        let response = client
            .post(&token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", client_id),
                ("client_secret", client_secret),
                ("scope", AZURE_SCOPE),
            ])
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Azure AD token request failed: {e}"))?;

        let parsed = parse_azure_token_response(response, "client-credentials").await?;

        let mut guard = self.token_cache.lock().await;
        *guard = Some(parsed.cached);
        Ok(parsed.token)
    }

    /// Return a cached AD token if still fresh; otherwise acquire a new one
    /// from the Azure IMDS managed-identity endpoint.
    async fn cached_or_fetch_managed_identity_token(
        &self,
        client_id: Option<&str>,
    ) -> anyhow::Result<String> {
        {
            let guard = self.token_cache.lock().await;
            if let Some(cached) = guard.as_ref() {
                if Instant::now() < cached.refresh_after {
                    return Ok(cached.value.clone());
                }
            }
        }

        let mut url = format!(
            "{}&resource={}",
            IMDS_TOKEN_URL,
            urlencoding::encode("https://cognitiveservices.azure.com/")
        );
        if let Some(id) = client_id {
            url.push_str("&client_id=");
            url.push_str(&urlencoding::encode(id));
        }

        let client = oauth_http_client();
        let response = client
            .get(&url)
            .header("Metadata", "true")
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Azure IMDS token request failed: {e}"))?;

        let parsed = parse_azure_token_response(response, "managed-identity").await?;

        let mut guard = self.token_cache.lock().await;
        *guard = Some(parsed.cached);
        Ok(parsed.token)
    }

    /// Convert `ToolSpec` slice to Azure/OpenAI native tool format.
    fn convert_tools(tools: Option<&[ToolSpec]>) -> Option<Vec<NativeToolSpec>> {
        tools.map(|items| {
            items
                .iter()
                .map(|tool| NativeToolSpec {
                    kind: "function".to_string(),
                    function: NativeToolFunctionSpec {
                        name: tool.name.clone(),
                        description: tool.description.clone(),
                        parameters: tool.parameters.clone(),
                    },
                })
                .collect()
        })
    }

    /// Convert `ChatMessage` slice to `NativeMessage` vec, handling tool calls
    /// and tool results the same way the OpenAI provider does.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<NativeMessage> {
        messages
            .iter()
            .map(|m| {
                if m.role == "assistant" {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&m.content) {
                        if let Some(tool_calls_value) = value.get("tool_calls") {
                            if let Ok(parsed_calls) =
                                serde_json::from_value::<Vec<ProviderToolCall>>(
                                    tool_calls_value.clone(),
                                )
                            {
                                let tool_calls = parsed_calls
                                    .into_iter()
                                    .map(|tc| NativeToolCall {
                                        id: Some(tc.id),
                                        kind: Some("function".to_string()),
                                        function: NativeFunctionCall {
                                            name: tc.name,
                                            arguments: tc.arguments,
                                        },
                                    })
                                    .collect::<Vec<_>>();
                                let content = value
                                    .get("content")
                                    .and_then(serde_json::Value::as_str)
                                    .map(ToString::to_string);
                                return NativeMessage {
                                    role: "assistant".to_string(),
                                    content,
                                    tool_call_id: None,
                                    tool_calls: Some(tool_calls),
                                };
                            }
                        }
                    }
                }

                if m.role == "tool" {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&m.content) {
                        let tool_call_id = value
                            .get("tool_call_id")
                            .and_then(serde_json::Value::as_str)
                            .map(ToString::to_string);
                        let content = value
                            .get("content")
                            .and_then(serde_json::Value::as_str)
                            .map(ToString::to_string);
                        return NativeMessage {
                            role: "tool".to_string(),
                            content,
                            tool_call_id,
                            tool_calls: None,
                        };
                    }
                }

                NativeMessage {
                    role: m.role.clone(),
                    content: Some(m.content.clone()),
                    tool_call_id: None,
                    tool_calls: None,
                }
            })
            .collect()
    }

    /// Parse a `NativeResponseMessage` into a `ProviderChatResponse`.
    fn parse_native_response(message: NativeResponseMessage) -> ProviderChatResponse {
        let text = message.content.filter(|c| !c.is_empty());
        let tool_calls = message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ProviderToolCall {
                id: tc.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                name: tc.function.name,
                arguments: tc.function.arguments,
            })
            .collect::<Vec<_>>();

        ProviderChatResponse {
            text,
            tool_calls,
            usage: None,
            reasoning_content: None,
        }
    }

    fn api_http_client(&self) -> Client {
        crate::config::build_runtime_proxy_client_with_timeouts("provider.azure_openai", 120, 10)
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

fn read_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Build a lightweight HTTP client for Azure AD token requests.
fn oauth_http_client() -> Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .connect_timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

struct ParsedToken {
    token: String,
    cached: CachedToken,
}

async fn parse_azure_token_response(
    response: reqwest::Response,
    flow: &str,
) -> anyhow::Result<ParsedToken> {
    let status = response.status();
    let body = response
        .text()
        .await
        .unwrap_or_else(|_| "<failed to read Azure AD response body>".to_string());

    if !status.is_success() {
        // Try to extract a human-readable error message.
        let detail = serde_json::from_str::<AzureTokenResponse>(&body)
            .ok()
            .and_then(|r| r.error_description.or(r.error))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| body.chars().take(200).collect());
        anyhow::bail!("Azure AD {flow} token request failed (HTTP {status}): {detail}");
    }

    let parsed = serde_json::from_str::<AzureTokenResponse>(&body)
        .map_err(|e| anyhow::anyhow!("Azure AD {flow} token response is not valid JSON: {e}"))?;

    if let Some(err) = parsed.error.as_deref().filter(|s| !s.trim().is_empty()) {
        let detail = parsed.error_description.as_deref().unwrap_or(err);
        anyhow::bail!("Azure AD {flow} token request failed: {detail}");
    }

    let token = parsed
        .access_token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Azure AD {flow} token response missing access_token"))?
        .to_string();

    // Determine when to refresh.
    let expires_in_secs: u64 = if let Some(expires_in) = parsed.expires_in {
        expires_in
    } else if let Some(expires_on) = &parsed.expires_on {
        // Managed identity returns an absolute Unix timestamp.
        let unix_ts = match expires_on {
            serde_json::Value::Number(n) => n.as_u64().unwrap_or(0),
            serde_json::Value::String(s) => s.parse::<u64>().unwrap_or(0),
            _ => 0,
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs())
            .unwrap_or(0);
        unix_ts.saturating_sub(now)
    } else {
        3600 // safe fallback
    };

    let refresh_secs = expires_in_secs.saturating_sub(TOKEN_REFRESH_BEFORE_EXPIRY_SECS);
    let refresh_after = Instant::now() + Duration::from_secs(refresh_secs);

    Ok(ParsedToken {
        token: token.clone(),
        cached: CachedToken {
            value: token,
            refresh_after,
        },
    })
}

// ── Provider trait implementation ────────────────────────────────────────────

#[async_trait]
impl Provider for AzureOpenAiProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: false,
        }
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let (header_name, header_value) = self.auth_header().await?;

        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(SimpleMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }
        messages.push(SimpleMessage {
            role: "user".to_string(),
            content: message.to_string(),
        });

        let request = SimpleChatRequest {
            model: model.to_string(),
            messages,
            temperature,
            max_tokens: None,
        };

        let response = self
            .api_http_client()
            .post(self.chat_completions_url(model))
            .header(header_name, &header_value)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("Azure OpenAI", response).await);
        }

        let chat_response: SimpleChatResponse = response.json().await?;
        chat_response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .filter(|c| !c.is_empty())
            .ok_or_else(|| anyhow::anyhow!("No response from Azure OpenAI"))
    }

    async fn chat(
        &self,
        request: ProviderChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ProviderChatResponse> {
        let (header_name, header_value) = self.auth_header().await?;

        let tools = Self::convert_tools(request.tools);
        let native_request = NativeChatRequest {
            model: model.to_string(),
            messages: Self::convert_messages(request.messages),
            temperature,
            max_tokens: None,
            tool_choice: tools.as_ref().map(|_| "auto".to_string()),
            tools,
        };

        let response = self
            .api_http_client()
            .post(self.chat_completions_url(model))
            .header(header_name, &header_value)
            .json(&native_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("Azure OpenAI", response).await);
        }

        let native_response: NativeChatResponse = response.json().await?;
        let usage = native_response.usage.map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        });
        let message = native_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| anyhow::anyhow!("No response from Azure OpenAI"))?;
        let mut result = Self::parse_native_response(message);
        result.usage = usage;
        Ok(result)
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ProviderChatResponse> {
        let (header_name, header_value) = self.auth_header().await?;

        let native_tools: Option<Vec<NativeToolSpec>> = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .cloned()
                    .map(|v| {
                        serde_json::from_value::<NativeToolSpec>(v).map_err(|e| {
                            anyhow::anyhow!("Invalid Azure OpenAI tool specification: {e}")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };

        let native_request = NativeChatRequest {
            model: model.to_string(),
            messages: Self::convert_messages(messages),
            temperature,
            max_tokens: None,
            tool_choice: native_tools.as_ref().map(|_| "auto".to_string()),
            tools: native_tools,
        };

        let response = self
            .api_http_client()
            .post(self.chat_completions_url(model))
            .header(header_name, &header_value)
            .json(&native_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("Azure OpenAI", response).await);
        }

        let native_response: NativeChatResponse = response.json().await?;
        let usage = native_response.usage.map(|u| TokenUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        });
        let message = native_response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message)
            .ok_or_else(|| anyhow::anyhow!("No response from Azure OpenAI"))?;
        let mut result = Self::parse_native_response(message);
        result.usage = usage;
        Ok(result)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_with_api_key_override() {
        let p = AzureOpenAiProvider::new(
            Some("https://myresource.openai.azure.com"),
            Some("test-api-key"),
        );
        assert_eq!(p.endpoint, "https://myresource.openai.azure.com");
        assert!(matches!(p.source, AzureCredentialSource::ApiKey(ref k) if k == "test-api-key"));
    }

    #[test]
    fn creates_with_empty_key_falls_back_to_managed_identity() {
        // When key override is empty and no env vars are set, defaults to ManagedIdentity.
        let p = AzureOpenAiProvider::new(Some("https://myresource.openai.azure.com"), Some(""));
        // Empty key → should not be ApiKey
        assert!(!matches!(p.source, AzureCredentialSource::ApiKey(_)));
    }

    #[test]
    fn endpoint_trailing_slash_is_trimmed() {
        let p = AzureOpenAiProvider::new(Some("https://myresource.openai.azure.com/"), Some("key"));
        assert_eq!(p.endpoint, "https://myresource.openai.azure.com");
    }

    #[test]
    fn chat_completions_url_format() {
        let p = AzureOpenAiProvider::new(Some("https://myresource.openai.azure.com"), Some("key"));
        let url = p.chat_completions_url("gpt-4o");
        assert!(url.contains("/openai/deployments/gpt-4o/chat/completions"));
        assert!(url.contains("api-version="));
    }

    #[test]
    fn default_api_version_is_used_when_env_unset() {
        let p = AzureOpenAiProvider::new(Some("https://example.openai.azure.com"), Some("k"));
        // The default is used as long as AZURE_OPENAI_API_VERSION is not in the environment.
        // We can only verify it is non-empty.
        assert!(!p.api_version.is_empty());
    }

    #[tokio::test]
    async fn auth_header_returns_api_key_header() {
        let p =
            AzureOpenAiProvider::new(Some("https://myresource.openai.azure.com"), Some("my-key"));
        let (name, value) = p.auth_header().await.unwrap();
        assert_eq!(name, "api-key");
        assert_eq!(value, "my-key");
    }

    #[tokio::test]
    async fn auth_header_fails_without_endpoint() {
        // No endpoint override and AZURE_OPENAI_ENDPOINT not set → error.
        // Use a fresh provider with empty endpoint.
        let p = AzureOpenAiProvider {
            endpoint: String::new(),
            api_version: DEFAULT_API_VERSION.to_string(),
            source: AzureCredentialSource::ApiKey("key".to_string()),
            token_cache: Mutex::new(None),
        };
        let result = p.auth_header().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("endpoint not set"));
    }

    #[tokio::test]
    async fn chat_with_system_fails_without_endpoint() {
        let p = AzureOpenAiProvider {
            endpoint: String::new(),
            api_version: DEFAULT_API_VERSION.to_string(),
            source: AzureCredentialSource::ApiKey("key".to_string()),
            token_cache: Mutex::new(None),
        };
        let result = p.chat_with_system(None, "hello", "gpt-4o", 0.7).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("endpoint not set"));
    }

    #[test]
    fn convert_messages_plain_user_message() {
        let messages = vec![ChatMessage::user("Hello".to_string())];
        let native = AzureOpenAiProvider::convert_messages(&messages);
        assert_eq!(native.len(), 1);
        assert_eq!(native[0].role, "user");
        assert_eq!(native[0].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn convert_messages_tool_result() {
        let payload = serde_json::json!({
            "tool_call_id": "call_1",
            "content": "done"
        });
        let messages = vec![ChatMessage::tool(payload.to_string())];
        let native = AzureOpenAiProvider::convert_messages(&messages);
        assert_eq!(native[0].role, "tool");
        assert_eq!(native[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(native[0].content.as_deref(), Some("done"));
    }

    #[test]
    fn convert_messages_assistant_tool_calls() {
        let payload = serde_json::json!({
            "content": "I will check",
            "tool_calls": [{"id": "tc_1", "name": "shell", "arguments": "{}"}]
        });
        let messages = vec![ChatMessage::assistant(payload.to_string())];
        let native = AzureOpenAiProvider::convert_messages(&messages);
        assert_eq!(native[0].role, "assistant");
        assert!(native[0].tool_calls.is_some());
    }

    #[test]
    fn native_chat_request_serializes_with_tools() {
        let req = NativeChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![NativeMessage {
                role: "user".to_string(),
                content: Some("hi".to_string()),
                tool_call_id: None,
                tool_calls: None,
            }],
            temperature: 0.7,
            max_tokens: None,
            tools: Some(vec![NativeToolSpec {
                kind: "function".to_string(),
                function: NativeToolFunctionSpec {
                    name: "shell".to_string(),
                    description: "Run a command".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
            tool_choice: Some("auto".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"gpt-4o\""));
        assert!(json.contains("\"tool_choice\":\"auto\""));
        assert!(json.contains("\"type\":\"function\""));
    }

    #[test]
    fn native_response_parses_usage() {
        let json = r#"{
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
    }

    #[test]
    fn native_response_parses_tool_calls() {
        let json = r#"{
            "choices": [{"message": {
                "content": null,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]
            }}]
        }"#;
        let resp: NativeChatResponse = serde_json::from_str(json).unwrap();
        let msg = resp.choices.into_iter().next().unwrap().message;
        let parsed = AzureOpenAiProvider::parse_native_response(msg);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "shell");
    }

    #[test]
    fn azure_token_response_parses_client_credentials() {
        let json = r#"{
            "token_type": "Bearer",
            "expires_in": 3599,
            "access_token": "test-token-value"
        }"#;
        let parsed: AzureTokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.access_token.as_deref(), Some("test-token-value"));
        assert_eq!(parsed.expires_in, Some(3599));
    }

    #[test]
    fn azure_token_response_parses_managed_identity() {
        let json = r#"{
            "access_token": "mi-token",
            "expires_on": "1999999999",
            "resource": "https://cognitiveservices.azure.com",
            "token_type": "Bearer"
        }"#;
        let parsed: AzureTokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.access_token.as_deref(), Some("mi-token"));
        assert!(parsed.expires_on.is_some());
    }

    #[test]
    fn resolve_source_prefers_key_override() {
        let source = AzureOpenAiProvider::resolve_source(Some("explicit-key"));
        assert!(matches!(source, AzureCredentialSource::ApiKey(ref k) if k == "explicit-key"));
    }
}
