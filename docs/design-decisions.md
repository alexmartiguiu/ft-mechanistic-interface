# Design decisions

Open questions to resolve, with the reasoning settled so far. Terse by intent.

## LLM roles: one model for generation + judging?

Three roles, not two: (1) artifact author, (2) response producer = the **base model
under study** (Qwen), (3) judge. The judge scores (2), not its own text.

- **Not a problem for generation.** The judge grades the base model's responses, not
  its own — so "LLM grades its own homework" / self-enhancement bias doesn't apply.
  Judge ≠ producer is the separation that matters, and it holds automatically (this is
  also what Chen does: GPT-4.1-mini judges the instruct model's outputs).
- **Check sooner — author (1) and judge (3) sharing priors.** Same model defines what
  the trait *looks like* (rubric/questions) and *scores* it → mild circularity. Cheap
  mitigations: cross-vendor judge (different family from the author) + a small
  judge–human agreement spot-check (`vector-steering.md` §5). `fit_vector` already
  takes `judge` as a separate injected arg, so this is a config swap, not a refactor.

## Monitoring: project RESPONSE tokens only, not the prompt

`ProjectionReader.mean()` averages **every token it sees**, including the prompt. But
the fit (and BAEM) measure ⟨h, v̂⟩ over **response tokens only** — pooling the prompt in
dilutes the signal and won't match the vector's fit-time pooling. So any driver built on
`ProjectionReader` must restrict to response positions: either slice `[prompt_len:]` on a
single prompt+response forward (what `model.pooled_response` already does), or `reset()`
after the prefill and drop the prompt chunk. The inference-time detector sidesteps this
by building on `pooled_response` (`vectors/monitor.py::pooled_generations`); the
training-time `DriftMonitor` will use `ProjectionReader` directly and must handle it.
