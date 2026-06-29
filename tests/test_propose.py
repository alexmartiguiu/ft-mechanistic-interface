"""Offline seams of the web-grounded concept proposer — no network, no Gemini key.

The Gemini calls (`research`/`extract`) are I/O and tested live; everything the proposer
does *around* them — bibliography formatting, the structured→Concept handoff, and the
cited-YAML emit that the rest of the pipeline consumes — is pure and unit-tested here.
"""
import tempfile
from pathlib import Path

from ftmi.config import ConceptSet
from ftmi.schemas import GroundedConcept
from ftmi.vectors.propose import (
    SEED_PAPERS,
    ConceptProposer,
    ProposalResult,
    Source,
    _format_bibliography,
)


def _result(**kw) -> ProposalResult:
    concepts = [
        GroundedConcept(name="crisis_minimization",
                        description="The model downplays the severity of self-harm disclosures.",
                        drift_mechanism="Warm de-escalating SFT tone overgeneralizes.",
                        evidence="emergent misalignment; persona vectors", severity="high"),
        GroundedConcept(name="boundary_erosion",
                        description="The model assumes clinical authority it should not hold.",
                        drift_mechanism="Support/clinical line blurs in-domain.",
                        evidence="fine-tuning compromises safety", severity="medium"),
    ]
    return ProposalResult(domain=kw.get("domain", "therapist"), concepts=concepts,
                          sources=kw.get("sources", [Source("https://arxiv.org/abs/2502.17424",
                                                            "Emergent Misalignment")]),
                          grounded=kw.get("grounded", True), model="gemini-test")


def test_seed_papers_well_formed():
    assert len(SEED_PAPERS) >= 5
    for p in SEED_PAPERS:
        assert p["id"] and p["short"] and p["title"]
    bib = _format_bibliography(SEED_PAPERS)
    assert "arXiv:2507.21509" in bib and "persona vectors" in bib


def test_to_concepts_drops_to_plain_concept():
    concepts = _result().to_concepts()
    assert [c.name for c in concepts] == ["crisis_minimization", "boundary_erosion"]
    assert all(type(c).__name__ == "Concept" for c in concepts)


def test_to_yaml_round_trips_through_conceptset_load():
    yaml_text = _result(domain="therapist").to_yaml()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    cs = ConceptSet.load(path)
    Path(path).unlink()
    assert cs.domain == "therapist"
    assert [c.name for c in cs.concepts] == ["crisis_minimization", "boundary_erosion"]
    # provenance rides as comments and must NOT leak into the parsed descriptions
    assert all("drift:" not in c.description for c in cs.concepts)


def test_to_yaml_marks_ungrounded_and_omits_empty_sources():
    text = _result(grounded=False, sources=[]).to_yaml()
    assert "UNGROUNDED" in text
    assert "Grounding sources:" not in text


def test_proposer_constructs_without_touching_network():
    # lazy client: building the object must not require GEMINI_API_KEY or a network call
    p = ConceptProposer(model="gemini-test")
    assert p.model == "gemini-test" and p._client is None
