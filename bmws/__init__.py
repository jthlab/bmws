import jax
jax.config.update('jax_enable_x64', True)
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

from bmws.common import Observation
from bmws.sim import sim_and_fit, sim_wf
from bmws.estimate import estimate, estimate_em
from bmws.betamix import sample_paths, BetaMixture

__all__ = ['estimate', 'estimate_em', 'sim_and_fit', 'sim_wf', 'Observation', 'BetaMixture']
