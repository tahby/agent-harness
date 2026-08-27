# agent-harness

A small command-line agent: it sends your prompt to a chat model, optionally
runs local tools, and prints the final answer.

This crate is for learning the loop, not for production. The interesting part
is how model output becomes the next action.

## What an agent harness is

A model can reply with text, or it can ask to call a tool (read a file, write
a file, run a shell command). A harness is the glue around that:

1. Send the conversation to the model.
2. If the model requested tools, run them and append the results.
3. Send the updated conversation back.
4. Stop when the model replies with text only.

That loop lives in `src/agent.rs`. It matches on `Step`. It does not branch on
tool name strings; tools are looked up in a `HashMap`.

## Message and Step

`Message` is an enum with a payload per variant. A user message is text. An
assistant message can hold text and tool calls. A tool message is a reply to
one `ToolCallId`. There is no single struct with `role` plus a handful of
optional fields.

`next_step` is pure. It looks at one `AssistantTurn` and returns:

- `RunTools(...)` when the model requested at least one tool (any leftover
  text stays on the Assistant message)
- `Stop { text }` otherwise (missing text becomes an empty string)

The async runner starts at `CallModel`, appends whatever happened, and
repeats. It gives up after 20 steps so a tool loop cannot hang.

## Run with an API key

The crate talks to an OpenAI-compatible Chat Completions endpoint.

```
export OPENAI_API_KEY=sk-...
# optional
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4.1-mini

cargo run -- "list the files here and tell me what this crate is"
```

`OPENAI_API_KEY` is required for a live call. Tests never read it; they use a
`FakeProvider`.

Built-in tools (`read_file`, `write_file`, `shell`) operate in the process
working directory. Paths containing `..` are rejected.

## What to change next

Add a tool. Implement `Tool` in `src/tools.rs` (name, JSON Schema, `call`),
then `register` it on the registry in `builtins()`. Do not add a `match` on
the tool name in the agent loop.
