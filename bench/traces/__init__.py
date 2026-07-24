"""Fault traces for the AEGIS benchmark suite."""

from .per_tier import TraceFaultEvent, FaultTrace, FT_B0, FT_B1, FT_B2, FT_B3, FT_B4
from .mixed import make_ft_poisson, make_ft_burst, make_ft_production

__all__ = [
    "TraceFaultEvent",
    "FaultTrace",
    "FT_B0",
    "FT_B1",
    "FT_B2",
    "FT_B3",
    "FT_B4",
    "make_ft_poisson",
    "make_ft_burst",
    "make_ft_production",
]
