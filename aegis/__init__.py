"""
AEGIS — Adaptive, Escalating, Graded Infrastructure Survivability

Blast-radius-aware fault-tolerance runtime for LLM training & serving.
Composes R²CCL (transport), MeCeFO (compute), and TierCheck (storage)
under a unified telemetry + escalation policy plane.

Phase 0 MVP: Telemetry plane, Failure Classifier, Fault Epoch Service,
EPE stub, layer stubs, chaos-inject harness.
"""

__version__ = "0.1.0"
