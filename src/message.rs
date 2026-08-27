use serde_json::Value;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallId(String);

impl ToolCallId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    System {
        text: String,
    },
    User {
        text: String,
    },
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        tool_call_id: ToolCallId,
        content: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    CallModel,
    RunTools(Vec<ToolCall>),
    Stop { text: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AssistantTurn {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}
