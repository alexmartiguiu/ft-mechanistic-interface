"""ft-mechanistic-interface: fine-tuning with concept-vector drift instrumentation.

Pipeline: generate (Chen meta-prompt) -> extract (diff-of-means) -> project.
Monitoring, dataset audit, and mitigation are all the same projection primitive
against the fitted vectors. See docs/vector-steering.md.
"""

__version__ = "0.0.1"
