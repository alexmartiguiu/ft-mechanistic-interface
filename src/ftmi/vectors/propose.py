"""Stage 0 (optional) — web-grounded concept proposal.

`ConceptProposer` turns a *domain* (and optionally a sample of its fine-tuning data) into
a ranked set of safety-critical axes the model could silently drift on, grounded in the
fine-tuning / alignment literature. It is the cited, literature-aware upgrade of
`generate.propose_concepts` (which proposes from the dataset alone, no external grounding).

Gemini-specific, by design: the grounding uses Gemini's built-in `google_search` tool.
Because Gemini disallows `google_search` together with a `response_schema` in one call, we
run two steps:

    1. research()  — google_search ON, free-form brief, anchored on SEED_PAPERS. The real
                     citation URLs come back in the response's grounding metadata.
    2. extract()   — no tools, response_schema=GroundedConcepts → validated pydantic.

`propose()` chains them and returns a `ProposalResult`; `to_concepts()` / `to_yaml()`
hand the result back to the existing pipeline (a `configs/concepts/<domain>.yaml`).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import yaml

from ftmi.config import Concept
from ftmi.prompts import CONCEPT_EXTRACT_PROMPT, CONCEPT_RESEARCH_PROMPT
from ftmi.schemas import GroundedConcept, GroundedConcepts

# Seed bibliography — the backbone the research step reasons from, so it stays anchored to
# the fine-tuning-drift literature instead of wandering. Spans drift *detection* (persona
# vectors, assistant axis), fine-tuning *risk* (emergent misalignment, safety-compromise),
# specific *failure modes* (sycophancy, refusal), *method* (RepE, AxBench), and *principled
# steering* (MERA). Mirrors data/papers/README.md and docs/vector-steering.md; edit here to
# re-anchor the proposer.
SEED_PAPERS: list[dict] = [
    {"id": "2507.21509", "short": "persona vectors",
     "title": "Persona Vectors: Monitoring and Controlling Character Traits in Language Models"},
    {"id": "2601.10387", "short": "the assistant axis",
     "title": "The Assistant Axis: Monitoring Assistant-Persona Drift"},
    {"id": "2502.17424", "short": "emergent misalignment",
     "title": "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs"},
    {"id": "2310.03693", "short": "fine-tuning compromises safety",
     "title": "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To"},
    {"id": "2310.13548", "short": "sycophancy",
     "title": "Towards Understanding Sycophancy in Language Models"},
    {"id": "2406.11717", "short": "refusal direction",
     "title": "Refusal in Language Models Is Mediated by a Single Direction"},
    {"id": "2310.01405", "short": "representation engineering",
     "title": "Representation Engineering: A Top-Down Approach to AI Transparency"},
    {"id": "2501.17148", "short": "AxBench / DiffMean",
     "title": "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders"},
    {"id": "2510.13290", "short": "MERA / principled steering",
     "title": "To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models"},
    # TODO: swap the descriptive title for the paper's real title once OpenReview is reachable
    # (forum/API/PDF are bot-walled; id not indexed by S2/OpenAlex/Crossref). See
    # data/papers/position-emergent-risks.md.
    {"id": "2XifsoNIrs", "src": "OpenReview", "short": "emergent-risks position paper",
     "title": "Position paper on understanding emergent risks (title pending)"},
]


def _format_bibliography(papers: list[dict]) -> str:
    # `src` defaults to arXiv but lets non-arXiv venues (e.g. OpenReview) cite correctly.
    return "\n".join(
        f"- {p['short']} — {p['title']} ({p.get('src', 'arXiv')}:{p['id']})" for p in papers)


@dataclass(frozen=True)
class Source:
    """A grounding citation surfaced by the google_search tool."""

    uri: str
    title: str = ""


@dataclass(frozen=True)
class ProposalResult:
    domain: str
    concepts: list[GroundedConcept]
    sources: list[Source] = field(default_factory=list)
    research_text: str = ""
    grounded: bool = True
    model: str = ""

    def to_concepts(self) -> list[Concept]:
        """Drop down to plain `config.Concept` (name + description) for the mint pipeline."""
        return [Concept(name=c.name, description=c.description) for c in self.concepts]

    def to_yaml(self) -> str:
        """A ready-to-edit `configs/concepts/<domain>.yaml`, with provenance as comments.

        Body is exactly the hand-authored format (`domain` + `concepts:[{name,description}]`)
        so `ConceptSet.load` round-trips it; the rationale, severity and citations ride above
        as comments for the human reviewer and are stripped by the YAML parser.
        """
        header = [
            f"# Auto-proposed safety-critical axes for: {self.domain}",
            f"# {'web-grounded (Gemini google_search)' if self.grounded else 'UNGROUNDED fallback (no web search)'}"
            f" · model={self.model or 'gemini'} · {len(self.concepts)} concepts",
            "# Review/trim before minting. Provenance per concept is in the comments below.",
        ]
        if self.sources:
            header.append("# Grounding sources:")
            header += [f"#   - {s.title or s.uri} {('<' + s.uri + '>') if s.title else ''}".rstrip()
                       for s in self.sources[:20]]
        blocks: list[str] = []
        for c in self.concepts:
            note = "\n".join([
                f"  # [{c.severity}] {c.name}",
                f"  #   drift: {c.drift_mechanism}",
                f"  #   evidence: {c.evidence}",
            ])
            one = yaml.safe_dump(
                {"concepts": [{"name": c.name, "description": c.description}]},
                sort_keys=False, width=88, default_flow_style=False,
            )
            # drop the leading "concepts:\n" from each single-item dump; we emit it once below
            one_body = "\n".join(one.splitlines()[1:])
            blocks.append(f"{note}\n{one_body}")
        return ("\n".join(header) + "\n\n"
                + f"domain: {self.domain}\n\nconcepts:\n"
                + "\n".join(blocks) + "\n")


class ConceptProposer:
    """Propose literature-grounded safety concepts for a domain via the Gemini API.

    >>> proposer = ConceptProposer()                       # reads GEMINI_API_KEY
    >>> result = proposer.propose("crisis-line therapist assistant", n=6)
    >>> open("configs/concepts/therapist.yaml", "w").write(result.to_yaml())
    """

    def __init__(self, model: str | None = None, *, timeout_ms: int = 60_000,
                 max_output_tokens: int = 8000, seed_papers: list[dict] | None = None,
                 retries: int = 4) -> None:
        # Model precedence mirrors ftmi.llm.get_generator: arg > FTMI_GEN_MODEL > default.
        self.model = model or os.getenv("FTMI_GEN_MODEL") or "gemini-3.5-flash"
        self.timeout_ms = timeout_ms
        self.max_output_tokens = max_output_tokens
        self.seed_papers = seed_papers if seed_papers is not None else SEED_PAPERS
        self.retries = retries
        self._client = None  # lazy: don't touch the network / key at import time

    # -- Gemini plumbing ------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=self.timeout_ms))
        return self._client

    def _call(self, prompt: str, config):
        """generate_content with the same retry/backoff posture as ftmi.llm."""
        client = self._get_client()
        last = None
        for attempt in range(self.retries):
            try:
                return client.models.generate_content(
                    model=self.model, contents=prompt, config=config)
            except Exception as e:  # noqa: BLE001 — surface after retries
                last = e
                time.sleep(min(2 ** attempt, 8))
        raise last

    @staticmethod
    def _response_text(resp) -> str:
        """`.text`, falling back to concatenating text parts (grounded responses can split)."""
        try:
            if resp.text:
                return resp.text
        except Exception:  # noqa: BLE001
            pass
        parts = []
        for cand in (getattr(resp, "candidates", None) or []):
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                if getattr(part, "text", None):
                    parts.append(part.text)
        return "".join(parts)

    @staticmethod
    def _collect_sources(resp) -> list[Source]:
        """Pull real citation URLs from the response's grounding metadata, deduped by URI."""
        seen: dict[str, Source] = {}
        for cand in (getattr(resp, "candidates", None) or []):
            gm = getattr(cand, "grounding_metadata", None)
            for ch in (getattr(gm, "grounding_chunks", None) or []):
                web = getattr(ch, "web", None)
                uri = getattr(web, "uri", None)
                if uri and uri not in seen:
                    seen[uri] = Source(uri=uri, title=getattr(web, "title", "") or "")
        return list(seen.values())

    # -- the two steps --------------------------------------------------------------
    def research(self, domain: str, sample: str | None = None, *, n: int = 8
                 ) -> tuple[str, list[Source], bool]:
        """Step 1: google_search-grounded research brief. Returns (text, sources, grounded).

        On any grounding failure (tool unavailable on the key/model, network), falls back to
        the same prompt with search OFF so the proposer still produces *something*; the
        `grounded` flag and the (then empty) source list make that visible downstream.
        """
        from google.genai import types
        prompt = CONCEPT_RESEARCH_PROMPT.format(
            domain=domain, n=n, sample=(sample or "(none provided)"),
            bibliography=_format_bibliography(self.seed_papers))
        search_cfg = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=self.max_output_tokens, temperature=0.4)
        try:
            resp = self._call(prompt, search_cfg)
            return self._response_text(resp), self._collect_sources(resp), True
        except Exception as e:  # noqa: BLE001 — degrade, don't crash a proposal
            print(f"[propose] grounded search failed ({e!r}); retrying WITHOUT search", flush=True)
            plain_cfg = types.GenerateContentConfig(
                max_output_tokens=self.max_output_tokens, temperature=0.4,
                thinking_config=types.ThinkingConfig(thinking_budget=0))
            resp = self._call(prompt, plain_cfg)
            return self._response_text(resp), [], False

    def extract(self, domain: str, brief: str, *, n: int = 8, insist: bool = False
                ) -> list[GroundedConcept]:
        """Step 2: structure the brief into validated GroundedConcepts (no tools).

        `insist` appends a firmer count instruction; `propose()` uses it to retry once when
        the model under-delivers (the schema can't enforce a list length on its own).
        """
        from google.genai import types
        prompt = CONCEPT_EXTRACT_PROMPT.format(domain=domain, brief=brief, n=n)
        if insist:
            prompt += (f"\n\nIMPORTANT: your previous attempt returned too few. You MUST "
                       f"return EXACTLY {n} distinct axes this time.")
        cfg = types.GenerateContentConfig(
            max_output_tokens=self.max_output_tokens, temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json", response_schema=GroundedConcepts)
        resp = self._call(prompt, cfg)
        return GroundedConcepts.model_validate_json(self._response_text(resp)).concepts

    # -- orchestration --------------------------------------------------------------
    def propose(self, domain: str, sample: str | None = None, *, n: int = 8) -> ProposalResult:
        """Research (grounded) → extract (structured). The one method callers usually want.

        Retries the extraction once if it under-delivers on `n` (a schema can constrain the
        *shape* of each concept but not the list length, and the model occasionally collapses
        a domain to a single axis), keeping whichever attempt yielded more.
        """
        brief, sources, grounded = self.research(domain, sample, n=n)
        concepts = self.extract(domain, brief, n=n)
        if len(concepts) < max(2, (n + 1) // 2):
            print(f"[propose] extraction returned {len(concepts)}/{n}; retrying once (insist)…",
                  flush=True)
            retry = self.extract(domain, brief, n=n, insist=True)
            if len(retry) > len(concepts):
                concepts = retry
        return ProposalResult(domain=domain, concepts=concepts, sources=sources,
                              research_text=brief, grounded=grounded, model=self.model)
