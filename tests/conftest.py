import numpy as np
import pytest

from payoff import GameConfig

TINY = GameConfig(n_towers=3, n_soldiers=6, unit=1)  # 28 pure strategies
FOUR = GameConfig(n_towers=4, n_soldiers=8, unit=1)  # 165
FIVE = GameConfig(n_towers=5, n_soldiers=10, unit=1)  # 1001
FULL = GameConfig()  # 10 towers, 100 soldiers, units of 5: ~10M


def random_allocation(cfg: GameConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(cfg.n_units, np.full(cfg.n_towers, 1 / cfg.n_towers))


def random_mixture(cfg: GameConfig, k: int, rng: np.random.Generator):
    allocs = np.array([random_allocation(cfg, rng) for _ in range(k)])
    probs = rng.dirichlet(np.ones(k))
    return allocs, probs


@pytest.fixture
def rng():
    return np.random.default_rng(0)
