"""
Unified Recovery Consensus (URC).

Generalizes TierCheck's per-tier consensus to cross-layer: determines
"what's the latest globally-valid state after this fault" so that transport
(B0), compute (B1), and storage (B2–B4) recoveries don't disagree.

Key mechanism (§3.2): every classified fault increments a fault epoch.
Checkpoints, collective ops, and compute fallbacks are tagged with the epoch
they belong to.  Consensus only considers state from a consistent epoch
boundary — the cross-layer generalization of TierCheck's
"global minimum recoverable version" reduction.

This is the highest-research-risk net-new component (§6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RecoveryDecision:
    """Output of URC: the globally-valid state after a fault."""

    epoch: int                   # the epoch at which this decision was made
    min_valid_epoch: int         # minimum epoch all surviving ranks agree on
    surviving_ranks: list[int]   # ranks that participated in consensus
    agreed: bool                 # True if at least one rank responded
    reason: str = ""


class UnifiedRecoveryConsensus:
    """
    Epoch-gated surviving-rank consensus.

    Each surviving rank reports its latest committed epoch via report_epoch().
    When a fault occurs, agree() reduces across all *active* ranks to find
    the minimum recoverable epoch — the earliest point guaranteed to be
    consistent across the entire job.

    This ensures that an in-flight transport migration (B0) or a compute
    fallback (B1) that spans an epoch boundary doesn't produce a checkpoint
    capturing torn gradient state.
    """

    def __init__(self) -> None:
        # rank → latest epoch that rank has committed and reported
        self._rank_epochs: dict[int, int] = {}

    def report_epoch(self, rank: int, epoch: int) -> None:
        """
        Called by each rank to report its latest committed epoch.

        In production, ranks call this after every gradient accumulation
        step so the URC always has fresh data for consensus.
        """
        self._rank_epochs[rank] = epoch

    def agree(self, active_ranks: list[int], current_epoch: int) -> RecoveryDecision:
        """
        Reduce across surviving *active_ranks* to find the min recoverable epoch.

        Ranks not in active_ranks (dead/missing) are excluded from the
        reduction — they are the ones we're recovering from.
        """
        surviving = [r for r in active_ranks if r in self._rank_epochs]

        if not surviving:
            logger.warning(
                "[URC] No surviving ranks with known epoch — cannot agree (current=%d)",
                current_epoch,
            )
            return RecoveryDecision(
                epoch=current_epoch,
                min_valid_epoch=0,
                surviving_ranks=[],
                agreed=False,
                reason="No surviving ranks with known epoch",
            )

        min_epoch = min(self._rank_epochs[r] for r in surviving)
        logger.info(
            "[URC] Consensus: %d surviving ranks → min_epoch=%d (current=%d)",
            len(surviving),
            min_epoch,
            current_epoch,
        )

        return RecoveryDecision(
            epoch=current_epoch,
            min_valid_epoch=min_epoch,
            surviving_ranks=surviving,
            agreed=True,
            reason=f"Min across {len(surviving)} surviving ranks",
        )

    def clear_rank(self, rank: int) -> None:
        """Remove a dead rank from the consensus pool."""
        self._rank_epochs.pop(rank, None)

    def all_ranks(self) -> list[int]:
        """Return all ranks that have reported at least once."""
        return list(self._rank_epochs.keys())
