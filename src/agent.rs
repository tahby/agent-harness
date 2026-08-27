use thiserror::Error;

use crate::message::{AssistantTurn, Message, Step, ToolCall};
use crate::provider::{ChatProvider, ProviderError};
use crate::tools::ToolRegistry;

const MAX_STEPS: usize = 20;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("stopped after {MAX_STEPS} steps without a final answer")]
    IterationLimit,
}

pub fn next_step(turn: AssistantTurn) -> Step {
    if !turn.tool_calls.is_empty() {
        Step::RunTools(turn.tool_calls)
    } else {
        Step::Stop {
            text: turn.text.unwrap_or_default(),
        }
    }
}

pub async fn run<P: ChatProvider>(
    provider: &P,
    registry: &ToolRegistry,
    messages: &mut Vec<Message>,
) -> Result<String, AgentError> {
    let specs = registry.specs();
    let mut step = Step::CallModel;

    for _ in 0..MAX_STEPS {
        match step {
            Step::CallModel => {
                let turn = provider.complete(messages, &specs).await?;
                messages.push(Message::Assistant {
                    text: turn.text.clone(),
                    tool_calls: turn.tool_calls.clone(),
                });
                step = next_step(turn);
            }
            Step::RunTools(calls) => {
                for call in calls {
                    let content = dispatch(registry, &call);
                    messages.push(Message::Tool {
                        tool_call_id: call.id,
                        content,
                    });
                }
                step = Step::CallModel;
            }
            Step::Stop { text } => return Ok(text),
        }
    }

    Err(AgentError::IterationLimit)
}

fn dispatch(registry: &ToolRegistry, call: &ToolCall) -> String {
    match registry.get(&call.name) {
        Some(tool) => match tool.call(call.arguments.clone()) {
            Ok(output) => output,
            Err(err) => err.to_string(),
        },
        None => format!("unknown tool: {}", call.name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ToolCall, ToolCallId};
    use crate::tools::{Tool, ToolError, ToolSpec};
    use serde_json::{json, Value};
    use std::sync::Mutex;

    struct FakeProvider {
        turns: Mutex<Vec<AssistantTurn>>,
    }

    impl FakeProvider {
        fn new(turns: Vec<AssistantTurn>) -> Self {
            Self {
                turns: Mutex::new(turns),
            }
        }
    }

    impl ChatProvider for FakeProvider {
        async fn complete(
            &self,
            _messages: &[Message],
            _tools: &[ToolSpec],
        ) -> Result<AssistantTurn, ProviderError> {
            let mut turns = self.turns.lock().expect("fake provider mutex");
            if turns.is_empty() {
                return Err(ProviderError::InvalidResponse(
                    "fake provider has no more turns".into(),
                ));
            }
            Ok(turns.remove(0))
        }
    }

    struct Ping;

    impl Tool for Ping {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "ping".into(),
                description: "Echo a test payload.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": { "n": { "type": "number" } },
                }),
            }
        }

        fn call(&self, arguments: Value) -> Result<String, ToolError> {
            Ok(arguments.to_string())
        }
    }

    fn ping_call() -> ToolCall {
        ToolCall {
            id: ToolCallId::new("call_1"),
            name: "ping".into(),
            arguments: json!({ "n": 1 }),
        }
    }

    #[test]
    fn next_step_text_only_is_stop() {
        let step = next_step(AssistantTurn {
            text: Some("hello".into()),
            tool_calls: vec![],
        });
        assert_eq!(
            step,
            Step::Stop {
                text: "hello".into()
            }
        );
    }

    #[test]
    fn next_step_tool_calls_are_run_tools() {
        let calls = vec![ping_call()];
        let step = next_step(AssistantTurn {
            text: Some("working".into()),
            tool_calls: calls.clone(),
        });
        assert_eq!(step, Step::RunTools(calls));
    }

    #[test]
    fn next_step_empty_text_and_no_tools_is_stop() {
        let step = next_step(AssistantTurn {
            text: None,
            tool_calls: vec![],
        });
        assert_eq!(step, Step::Stop { text: String::new() });
    }

    #[tokio::test]
    async fn run_requests_one_tool_then_text() {
        let provider = FakeProvider::new(vec![
            AssistantTurn {
                text: Some("calling ping".into()),
                tool_calls: vec![ping_call()],
            },
            AssistantTurn {
                text: Some("done".into()),
                tool_calls: vec![],
            },
        ]);
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(Ping));
        let mut messages = vec![Message::User {
            text: "go".into(),
        }];

        let answer = run(&provider, &registry, &mut messages)
            .await
            .expect("run should finish");
        assert_eq!(answer, "done");
        assert_eq!(
            messages,
            vec![
                Message::User { text: "go".into() },
                Message::Assistant {
                    text: Some("calling ping".into()),
                    tool_calls: vec![ping_call()],
                },
                Message::Tool {
                    tool_call_id: ToolCallId::new("call_1"),
                    content: json!({ "n": 1 }).to_string(),
                },
                Message::Assistant {
                    text: Some("done".into()),
                    tool_calls: vec![],
                },
            ]
        );
    }

    #[tokio::test]
    async fn unknown_tool_becomes_tool_error_message() {
        let provider = FakeProvider::new(vec![
            AssistantTurn {
                text: None,
                tool_calls: vec![ToolCall {
                    id: ToolCallId::new("call_missing"),
                    name: "not_registered".into(),
                    arguments: json!({}),
                }],
            },
            AssistantTurn {
                text: Some("ok".into()),
                tool_calls: vec![],
            },
        ]);
        let registry = ToolRegistry::new();
        let mut messages = vec![Message::User {
            text: "use a missing tool".into(),
        }];

        run(&provider, &registry, &mut messages)
            .await
            .expect("unknown tool must not panic");

        match &messages[2] {
            Message::Tool {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, &ToolCallId::new("call_missing"));
                assert!(
                    content.contains("not_registered"),
                    "error should name the tool, got {content}"
                );
            }
            other => panic!("expected Tool message, got {other:?}"),
        }
    }
}
