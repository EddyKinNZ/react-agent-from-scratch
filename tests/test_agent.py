"""Offline tests: a scripted fake model drives the real loop and real tools."""
import json
import os

import pytest

from react_agent.agent import ReActAgent
from react_agent.llm import LLMError, ResilientLLM
from react_agent.tools import build_tools, calculator

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "bus_punctuality_sample.csv")


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies, fail_first=0):
        self.replies, self.fail_first, self.seen = list(replies), fail_first, []

    def complete(self, system, messages, stop):
        if self.fail_first:
            self.fail_first -= 1
            raise LLMError("503 overloaded")
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0), 10, 5


def make(replies, **kw):
    return ReActAgent(ResilientLLM([ScriptedLLM(replies)], sleep=lambda s: None), build_tools(CSV), **kw)


def test_two_step_answer_uses_tool_result():
    agent = make([
        'Thought: need school means\nAction: aggregate\nAction Input: '
        '{"column": "punctuality_pct", "agg": "mean", "group_by": "area", "where": {"service": "School"}}',
        "Thought: I can answer now\nFinal Answer: South is lowest.",
    ])
    out = agent.run("Which area is lowest for School services?")
    assert out == {"answer": "South is lowest.", "steps": 2}
    obs = agent.llm.models[0].seen[-1]
    assert "South" in obs and obs.startswith("Observation: <observation>")


def test_bad_json_and_unknown_tool_are_fed_back_not_raised():
    agent = make([
        "Thought: x\nAction: aggregate\nAction Input: {not json}",
        'Thought: x\nAction: delete_everything\nAction Input: {}',
        "Final Answer: done",
    ])
    assert agent.run("q")["steps"] == 3
    seen = agent.llm.models[0].seen
    assert "not valid JSON" in seen[1] and "unknown tool" in seen[2]


def test_invented_observation_is_dropped():
    agent = make([
        'Thought: x\nAction: calculator\nAction Input: {"expression": "2+2"}\nObservation: 5',
        "Final Answer: 4",
    ])
    agent.run("2+2?")
    assert "<observation>4</observation>" in agent.llm.models[0].seen[1]


def test_step_limit():
    agent = make(["Thought: hmm"] * 3, max_steps=3)
    assert agent.run("q")["answer"].startswith("Stopped after 3 steps")


def test_retry_then_fallback():
    flaky = ScriptedLLM([], fail_first=10)
    good = ScriptedLLM(["Final Answer: ok"])
    llm = ResilientLLM([flaky, good], retries=1, sleep=lambda s: None)
    assert llm.complete("s", [{"role": "user", "content": "q"}], []) == "Final Answer: ok"
    assert len(llm.usage.calls) == 1


def test_calculator_refuses_code():
    assert calculator({"expression": "(91.5-87.7)/87.7*100"}) == "4.332953"
    with pytest.raises(ValueError):
        calculator({"expression": "__import__('os').system('rm -rf /')"})


def test_trace_file(tmp_path):
    p = tmp_path / "t.jsonl"
    make(["Final Answer: hi"], trace_path=str(p)).run("q")
    events = [json.loads(l)["event"] for l in p.read_text().splitlines()]
    assert events == ["question", "model", "final"]
