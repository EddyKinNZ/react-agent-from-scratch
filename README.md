# ReAct Agent from Scratch

A minimal ReAct agent in about 280 lines of plain Python. There is no agent framework and no SDK: the code makes plain HTTP calls to Claude or Gemini, runs a Thought → Action → Observation loop, and answers questions about a CSV with read-only tools.

I built it to understand what agent frameworks do underneath, and to practise the engineering problems that matter in production:
- tool errors
- prompt injection through tool output
- context limits
- retries and model fallback
- cost tracking
- traceability

## What it does

The agent replies in this format (abridged example):

```
$ python run.py "Which area has the lowest average punctuality for School services, and by how much does it trail Central?"

Thought: I need mean punctuality for School services by area
Action: aggregate
Action Input: {"column": "punctuality_pct", "agg": "mean", "group_by": "area", "where": {"service": "School"}}
Observation: <observation> ... </observation>
Thought: South is lowest; I need the gap to Central
Action: calculator
...
Final Answer: ...
```

The sample data (`data/bus_punctuality_sample.csv`) is **synthetic**. It is generated to resemble monthly bus punctuality by area and service level, so the demo runs without any private or licensed data.

## Design

| Concern | How it is handled | Where |
|---|---|---|
| Hallucinated observations | A stop sequence on `Observation:`, and any text the model writes after its Action is discarded | `agent.py` |
| Bad tool calls | Unknown tools, invalid JSON and tool exceptions go back to the model as an `error:` observation, so the loop recovers instead of crashing | `agent.py` |
| Prompt injection | Tool output is wrapped in `<observation>` tags, and the system prompt treats it as data, never as instructions | `agent.py` |
| Unsafe tools | The calculator walks the AST (no `eval`), and the CSV tool only filters and aggregates whitelisted columns and functions | `tools.py` |
| Context window | Observations are truncated to a fixed size, and the step limit bounds the conversation | `agent.py` |
| Rate limits and outages | Exponential backoff per model, then fallback to the next model (for example, free Gemini first, then Claude Haiku) | `llm.py` |
| Cost | Token usage and estimated USD per call | `llm.py` |
| Traceability | Every question, model reply, observation and final answer is appended to `trace.jsonl` | `agent.py` |

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...        # optional, free tier
export ANTHROPIC_API_KEY=...     # optional, used as fallback
python run.py "What is the average punctuality in South Auckland?"
```

## Tests

The tests use a scripted fake model to drive the real loop and the real tools, so they run offline with no API key.

```bash
python -m pytest -q
```

They cover:
- a two-step answer;
- recovery from bad JSON and from an unknown tool;
- discarding an invented observation;
- the step limit;
- retry followed by fallback;
- the calculator refusing code;
- the trace file.

## Next steps

- Native function calling, compared with the text-based ReAct format.
- Summarising older steps instead of truncating them.
- An evaluation set of questions with known answers, to compare models on accuracy, steps and cost.

## Licence

MIT
