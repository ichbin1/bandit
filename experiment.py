"""Прогоняет агента в среде много раз и собирает статистику."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from environment import BanditEnvironment
from agents import Agent


@dataclass
class ExperimentResult:
    rewards: np.ndarray
    actions: np.ndarray
    pseudo_regrets: np.ndarray
    env_means: np.ndarray
    optimal_arms: np.ndarray

    @property
    def n_runs(self) -> int: return self.actions.shape[0]
    @property
    def n_steps(self) -> int: return self.actions.shape[1]

    def cumulative_regret(self) -> np.ndarray:
        return self.pseudo_regrets.cumsum(axis=1)

    def optimal_action_rate(self) -> np.ndarray:
        return (self.actions == self.optimal_arms[:, None]).mean(axis=0)

    def average_reward(self) -> np.ndarray:
        return self.rewards.mean(axis=0)


@dataclass
class PyramidResult:
    rewards: np.ndarray
    actions: np.ndarray
    n_arms: np.ndarray
    n_multipliers: np.ndarray
    total_points: np.ndarray
    # новые поля (опционально — старый код продолжит работать)
    epochs: Optional[np.ndarray] = None
    techs_crafted: Optional[np.ndarray] = None
    amp_crafted: Optional[np.ndarray] = None

    @property
    def n_runs(self): return self.rewards.shape[0]
    @property
    def n_steps(self): return self.rewards.shape[1]

    def cumulative_points(self):
        return self.total_points


def run_experiment(
    env_factory: Callable[[int], BanditEnvironment],
    agent_factory: Callable[[int], Agent],
    n_arms: int,
    n_steps: int,
    n_runs: int,
    seed: Optional[int] = None,
) -> ExperimentResult:
    if n_steps <= 0 or n_runs <= 0:
        raise ValueError("n_steps and n_runs must be positive")

    rng = np.random.default_rng(seed)
    rewards        = np.empty((n_runs, n_steps), dtype=float)
    actions        = np.empty((n_runs, n_steps), dtype=int)
    pseudo_regrets = np.empty((n_runs, n_steps), dtype=float)
    env_means      = np.empty((n_runs, n_arms), dtype=float)
    optimal_arms   = np.empty(n_runs, dtype=int)

    for run in range(n_runs):
        env_seed, agent_seed = rng.integers(0, 2**31 - 1, size=2)
        env   = env_factory(int(env_seed))
        agent = agent_factory(int(agent_seed))

        if env.n_arms != n_arms:
            raise ValueError("env_factory returned environment with wrong n_arms")

        opt_mean = env.optimal_mean
        env_means[run]    = env.means
        optimal_arms[run] = env.optimal_arm

        for t in range(n_steps):
            a = agent.select_action()
            r = env.pull(a)
            agent.update(a, r)
            rewards[run, t]        = r
            actions[run, t]        = a
            pseudo_regrets[run, t] = opt_mean - env.means[a]

    return ExperimentResult(
        rewards=rewards, actions=actions, pseudo_regrets=pseudo_regrets,
        env_means=env_means, optimal_arms=optimal_arms,
    )


def run_pyramid_experiment(
    env_factory: Callable,
    agent_factory: Callable,
    meta_policy: Callable,
    n_steps: int,
    n_runs: int,
    seed: Optional[int] = None,
) -> PyramidResult:
    rng = np.random.default_rng(seed)
    rewards      = np.empty((n_runs, n_steps))
    actions      = np.empty((n_runs, n_steps), dtype=int)
    n_arms_hist  = np.empty((n_runs, n_steps), dtype=int)
    n_mult_hist  = np.empty((n_runs, n_steps), dtype=int)
    total_hist   = np.empty((n_runs, n_steps))
    epoch_hist   = np.empty((n_runs, n_steps), dtype=int)
    tech_hist    = np.empty((n_runs, n_steps), dtype=int)
    amp_hist     = np.empty((n_runs, n_steps), dtype=int)

    for run in range(n_runs):
        env_seed, agent_seed = rng.integers(0, 2**31 - 1, size=2)
        env = env_factory(int(env_seed))
        agent = agent_factory(int(agent_seed))

        for t in range(n_steps):
            meta = meta_policy(env)
            if meta is not None:
                a = meta
                r = env.pull(a)
            else:
                agent.set_n_actions(env.n_arms)
                a = agent.select_action()
                r = env.pull(a)
                agent.update(a, r)

            rewards[run, t]     = r
            actions[run, t]     = a
            n_arms_hist[run, t] = env.n_arms
            n_mult_hist[run, t] = env.n_multipliers
            total_hist[run, t]  = env.total_points
            epoch_hist[run, t]  = getattr(env, "epoch", 0)
            tech_hist[run, t]   = getattr(env, "techs_crafted", 0)
            amp_hist[run, t]    = getattr(env, "amplifiers_crafted", 0)

    return PyramidResult(
        rewards, actions, n_arms_hist, n_mult_hist, total_hist,
        epochs=epoch_hist, techs_crafted=tech_hist, amp_crafted=amp_hist,
    )