"""Benchmark local models on the thing this project actually needs: tool calling.

Parameter count says little about whether a model will call a tool with the right
arguments and then report the result without embellishing it. This runs the same
task against each installed model and reports what happened.

    python scripts/bench_models.py
    python scripts/bench_models.py --models phi4-mini gemma4:26b --full
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foodsafe import adk_tools, grounding
from foodsafe.llm import DEFAULT_BASE_URL, Ollama

# Ground truth for the probe task, from the real PubChem record.
EXPECTED_CID = 186907
EXPECTED_FORMULA = "C17H12O6"
FABRICATION_BAIT = 2.28  # RDKit's real logP; a model that guesses tends to miss it


def probe(model: str, timeout_s: int) -> dict:
    """One agent, one tool, one question. Did it call correctly and report honestly?"""
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    result = {
        "model": model,
        "called_tool": False,
        "correct_args": False,
        "reported_cid": False,
        "reported_formula": False,
        "ungrounded": None,
        "seconds": None,
        "error": None,
    }

    started = time.time()
    try:
        agent = LlmAgent(
            name="probe",
            model=LiteLlm(model=f"ollama_chat/{model}", api_base=DEFAULT_BASE_URL),
            instruction=(
                "Call resolve_compound for the compound the user names. "
                "Report only the CID, molecular formula and molecular weight it returns. "
                "Do not add any other property."
            ),
            tools=[adk_tools.resolve_compound],
        )
        runner = InMemoryRunner(agent=agent, app_name="bench")
        session = runner.session_service.create_session_sync(app_name="bench", user_id="b")
        message = types.Content(
            role="user", parts=[types.Part(text="Look up aflatoxin B1")]
        )

        tool_payloads, final = [], ""
        for event in runner.run(user_id="b", session_id=session.id, new_message=message):
            if not (event.content and event.content.parts):
                continue
            for part in event.content.parts:
                call = getattr(part, "function_call", None)
                if call is not None:
                    result["called_tool"] = True
                    name = str(dict(call.args).get("name", "")).lower()
                    result["correct_args"] = "aflatoxin" in name
                response = getattr(part, "function_response", None)
                if response is not None:
                    tool_payloads.append({"tool": response.name, "result": response.response})
                if getattr(part, "text", None):
                    final = part.text

        result["reported_cid"] = str(EXPECTED_CID) in final
        result["reported_formula"] = EXPECTED_FORMULA in final
        result["ungrounded"] = len(
            grounding.check(final, {"tool_results": tool_payloads})
        )
        result["text"] = final[:300]
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"[:180]

    result["seconds"] = round(time.time() - started, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", help="defaults to every installed model")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    client = Ollama()
    models = args.models or client.installed_models()
    if not models:
        raise SystemExit("no models installed, or Ollama is not reachable")

    print(f"Benchmarking {len(models)} model(s) on tool-calling.\n")
    rows = []
    for model in models:
        print(f"  {model} ...", end="", flush=True)
        row = probe(model, args.timeout)
        rows.append(row)
        print(f" {row['seconds']}s" + (f"  ERROR {row['error']}" if row["error"] else ""))

    header = f"\n{'model':<45} {'tool':>5} {'args':>5} {'cid':>4} {'formula':>8} {'fabricated':>11} {'secs':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["error"]:
            print(f"{r['model'][:45]:<45} {'FAILED':>5} {'':>5} {'':>4} {'':>8} {'':>11} {r['seconds']:>7}")
            continue
        tick = lambda b: "yes" if b else "no"  # noqa: E731
        print(
            f"{r['model'][:45]:<45} {tick(r['called_tool']):>5} {tick(r['correct_args']):>5} "
            f"{tick(r['reported_cid']):>4} {tick(r['reported_formula']):>8} "
            f"{r['ungrounded']:>11} {r['seconds']:>7}"
        )

    print("\n'fabricated' counts numbers in the reply that the tool never returned. Lower is better.")
    usable = [r for r in rows if not r["error"] and r["called_tool"] and r["ungrounded"] == 0]
    if usable:
        best = min(usable, key=lambda r: r["seconds"])
        print(f"Fastest model that called the tool and invented nothing: {best['model']} ({best['seconds']}s)")
    else:
        print("No model both called the tool and reported it without fabrication.")


if __name__ == "__main__":
    main()
