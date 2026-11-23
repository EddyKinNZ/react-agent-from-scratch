"""Ask the agent a question about the sample data.

    export ANTHROPIC_API_KEY=...   # and/or GEMINI_API_KEY=...
    python run.py "Which area has the lowest average punctuality for School services?"
"""
import argparse
import os

from react_agent.agent import ReActAgent
from react_agent.llm import AnthropicLLM, GeminiLLM, ResilientLLM
from react_agent.tools import build_tools


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--csv", default="data/bus_punctuality_sample.csv")
    p.add_argument("--trace", default="trace.jsonl")
    a = p.parse_args()

    models = []
    if os.environ.get("GEMINI_API_KEY"):
        models.append(GeminiLLM())      # free tier first: the sample data is public
    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append(AnthropicLLM())   # fallback when Gemini is overloaded
    if not models:
        raise SystemExit("Set GEMINI_API_KEY and/or ANTHROPIC_API_KEY")

    llm = ResilientLLM(models)
    agent = ReActAgent(llm, build_tools(a.csv), trace_path=a.trace)
    result = agent.run(a.question)
    print(f"\n{result['answer']}\n")
    u = llm.usage
    print(f"steps={result['steps']} calls={len(u.calls)} tokens={u.input_tokens}+{u.output_tokens} "
          f"cost≈${u.cost_usd:.4f}  trace → {a.trace}")


if __name__ == "__main__":
    main()
