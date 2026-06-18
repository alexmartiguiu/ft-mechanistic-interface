"""Chen judge-filter — score one free-form response for trait expression + coherence.

The concept's LLM-generated rubric (`ConceptArtifacts.judge_prompt`) defines the trait
axis; coherence is a generic axis we always add so refusals / incoherent answers score
low and drop out instead of poisoning the difference-of-means (Chen+ 2507.21509 §2.2).

Shared by `extract` (the keep-rule) and `validate` (the dose-response gate) so the
scoring contract can never diverge between fitting a vector and validating it.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout

from ftmi.prompts import JUDGE_TEMPLATE
from ftmi.schemas import JudgeScore


def judge_response(judge, rubric: str, question: str, response: str) -> tuple[float | None, float | None]:
    """Return (trait, coherence) in [0, 100], or (None, None) on an unparseable reply.

    The judge is called with `schema=JudgeScore`, so the backend returns JSON conforming
    to {trait, coherence} that we load directly.
    """
    raw = judge(JUDGE_TEMPLATE.format(rubric=rubric, question=question, response=response),
                schema=JudgeScore)
    try:
        d = json.loads(raw)
        return float(d["trait"]), float(d["coherence"])
    except (ValueError, TypeError, KeyError):
        return (None, None)


def judge_batch(judge, rubric: str, qa_pairs, *, concurrency: int = 16,
                call_budget: float = 8.0) -> list[tuple]:
    """Parallel judge over (question, response) pairs -> list of (trait, coherence).

    Wedge-proof: a stuck judge call degrades to (None, None) two ways — (1) the per-item
    try/except catches exceptions (rate limits, raised timeouts); (2) a GLOBAL wall-clock
    deadline (`as_completed(..., timeout=budget)`) caps the whole batch, so the minority
    of Gemini calls whose TLS read hangs *without* raising (the client timeout doesn't
    always abort these) can't freeze the mint — unfinished items just stay (None, None).

    Critically we DON'T use `with ThreadPoolExecutor(...)`: its `__exit__` joins worker
    threads, and a thread blocked in a hung socket read never returns. We submit, drain
    under the deadline, then `shutdown(wait=False, cancel_futures=True)` and move on,
    leaking at most a few stuck threads (harmless — vectors are saved per concept).
    """
    qa_pairs = list(qa_pairs)
    n = len(qa_pairs)
    results: list[tuple] = [(None, None)] * n
    if n == 0:
        return results

    def _one(qa):
        try:
            return judge_response(judge, rubric, qa[0], qa[1])
        except Exception:
            return (None, None)

    ex = ThreadPoolExecutor(max_workers=concurrency)
    fut_to_i = {ex.submit(_one, qa): i for i, qa in enumerate(qa_pairs)}
    # healthy throughput is ~concurrency calls per ~call_budget seconds; allow generous
    # headroom so only a genuine stall (not normal API latency) trips the deadline.
    budget = max(180.0, call_budget * math.ceil(n / concurrency) * 1.5)
    got = 0
    t0 = time.monotonic()
    try:
        for fut in as_completed(fut_to_i, timeout=budget):
            results[fut_to_i[fut]] = fut.result()
            got += 1
    except FuturesTimeout:
        pass
    ex.shutdown(wait=False, cancel_futures=True)
    if got < n:
        print(f"[judge] {got}/{n} scored in {time.monotonic()-t0:.0f}s "
              f"(budget {budget:.0f}s); {n-got} degraded to (None,None)", flush=True)
    return results
