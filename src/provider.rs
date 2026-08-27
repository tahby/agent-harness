use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Config;
use crate::message::{AssistantTurn, Message, ToolCall, ToolCallId};
use crate::tools::ToolSpec;

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("OPENAI_API_KEY is required to call a model")]
    MissingApiKey,
    #[error("http error: {0}")]
    Http(String),
    #[error("api error ({status}): {body}")]
    Api { status: u16, body: String },
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

pub trait ChatProvider {
    fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> impl std::future::Future<Output = Result<AssistantTurn, ProviderError>> + Send;
}

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(config: &Config) -> Result<Self, ProviderError> {
        let api_key = config
            .api_key
            .clone()
            .ok_or(ProviderError::MissingApiKey)?;
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            model: config.model.clone(),
        })
    }
}

impl ChatProvider for OpenAiProvider {
    async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
    ) -> Result<AssistantTurn, ProviderError> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: messages.iter().map(wire_message).collect(),
            tools: tools.iter().map(wire_tool).collect(),
        };

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|err| ProviderError::Http(err.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|err| ProviderError::Http(err.to_string()))?;

        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body,
            });
        }

        let parsed: ChatResponse = serde_json::from_str(&body)
            .map_err(|err| ProviderError::InvalidResponse(err.to_string()))?;
        parsed.into_turn()
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct WireTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireFunction,
}

#[derive(Serialize)]
struct WireFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Serialize, Deserialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: WireCallFunction,
}

#[derive(Serialize, Deserialize)]
struct WireCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

impl ChatResponse {
    fn into_turn(self) -> Result<AssistantTurn, ProviderError> {
        let message = self
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::InvalidResponse("no choices".into()))?
            .message;

        let mut tool_calls = Vec::with_capacity(message.tool_calls.len());
        for call in message.tool_calls {
            // OpenAI sends `function.arguments` as a JSON string, not an object.
            let arguments = match serde_json::from_str(&call.function.arguments) {
                Ok(value) => value,
                Err(_) => Value::String(call.function.arguments),
            };
            tool_calls.push(ToolCall {
                id: ToolCallId::new(call.id),
                name: call.function.name,
                arguments,
            });
        }

        Ok(AssistantTurn {
            text: message.content,
            tool_calls,
        })
    }
}

fn wire_message(message: &Message) -> WireMessage {
    match message {
        Message::System { text } => WireMessage {
            role: "system",
            content: Some(text.clone()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message::User { text } => WireMessage {
            role: "user",
            content: Some(text.clone()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message::Assistant { text, tool_calls } => WireMessage {
            role: "assistant",
            content: text.clone(),
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.iter().map(wire_tool_call).collect())
            },
            tool_call_id: None,
        },
        Message::Tool {
            tool_call_id,
            content,
        } => WireMessage {
            role: "tool",
            content: Some(content.clone()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.as_str().to_string()),
        },
    }
}

fn wire_tool_call(call: &ToolCall) -> WireToolCall {
    WireToolCall {
        id: call.id.as_str().to_string(),
        kind: "function".into(),
        function: WireCallFunction {
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
        },
    }
}

fn wire_tool(spec: &ToolSpec) -> WireTool {
    WireTool {
        kind: "function",
        function: WireFunction {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.parameters.clone(),
        },
    }
}
