import logging

import jax

from bmws.betamix import BetaMixture
from bmws.common import Observation
from bmws.estimate import estimate, estimate_em
from bmws.sim import sim_and_fit, sim_wf

jax.config.update("jax_enable_x64", True)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)


__all__ = [
    "estimate",
    "estimate_em",
    "sim_and_fit",
    "sim_wf",
    "Observation",
    "BetaMixture",
]
