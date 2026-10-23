import matplotlib.pyplot as plt

from environment import LayeredPyramidBandit
from agents import UCB1, EpsilonGreedy, ThompsonSamplingBernoulli
from experiment import run_pyramid_experiment

N_STEPS = 3000
N_RUNS  = 100
SEED    = 42

# ----------------------------------------------------------------------
# Фабрика усложнённой среды
# ----------------------------------------------------------------------
env_factory = lambda s: LayeredPyramidBandit(
    n_arms_init=5,
    n_levels=3,
    info_per_multiplier=5,
    mult_per_arm_base=3,
    arm_cost_increase=1,
    mult_bonus=0.5,
    arm_bonus=0.5,
    arm_bonus_increase=0.1,
    max_arms=50,
    # --- экономика данных ---
    tech_multiplier_bonus=1.0,   # +1.0 множителя за каждый advancement
    tech_arm_mean_boost=0.15,    # +0.15 mean на каждый tier новой руки
    amplifier_mult_bonus=0.3,    # +0.3 множителя за amplifier
    amplifier_cost_growth=1.5,
    seed=s,
)

# ----------------------------------------------------------------------
# Мета-политики разной степени "продвинутости"
# ----------------------------------------------------------------------
def no_meta(env):
    """Никогда не использует мета-действия. Только пуллы рук агентом."""
    return None

def mult_only(env):
    """Только конверсия info → multiplier (самое простое)."""
    if env.info_units >= env.info_per_multiplier:
        return env.n_arms
    return None

def greedy_meta(env):
    """info→mult + покупка рук. Не доходит до технологий."""
    if env.info_units >= env.info_per_multiplier:
        return env.n_arms
    if env.n_multipliers >= env.current_arm_cost and env.n_arms < env.max_arms:
        return env.n_arms + 1
    return None

def amp_meta(env):
    """greedy_meta + amplifier, но БЕЗ advancement (не решает bottleneck)."""
    if env.can_craft_amplifier:
        return env.n_arms + 3
    if env.info_units >= env.info_per_multiplier:
        return env.n_arms
    if env.n_multipliers >= env.current_arm_cost and env.n_arms < env.max_arms:
        return env.n_arms + 1
    return None

def full_cycle(env):
    """Полный метацикл: gather → tech → advance → amplifier → expand.

    Приоритеты:
    1. advancement (наибольшая отложенная выгода)
    2. amplifier (линейный бонус к множителю)
    3. info → multiplier (базовая конверсия)
    4. покупка рук
    """
    if env.can_craft_gate:
        return env.n_arms + 2
    if env.can_craft_amplifier:
        return env.n_arms + 3
    if env.info_units >= env.info_per_multiplier:
        return env.n_arms
    if env.n_multipliers >= env.current_arm_cost and env.n_arms < env.max_arms:
        return env.n_arms + 1
    return None

# ----------------------------------------------------------------------
# Конфиги
# ----------------------------------------------------------------------
configs = {
    "UCB1 + no_meta":       (lambda s: UCB1(5, c=2.0, seed=s), no_meta),
    "UCB1 + mult_only":     (lambda s: UCB1(5, c=2.0, seed=s), mult_only),
    "UCB1 + greedy_meta":   (lambda s: UCB1(5, c=2.0, seed=s), greedy_meta),
    "UCB1 + amp_meta":      (lambda s: UCB1(5, c=2.0, seed=s), amp_meta),
    "UCB1 + full_cycle":    (lambda s: UCB1(5, c=2.0, seed=s), full_cycle),
    "TS + full_cycle":      (lambda s: ThompsonSamplingBernoulli(5, seed=s), full_cycle),
    "ε-greedy + full_cycle":(lambda s: EpsilonGreedy(5, 0.1, seed=s), full_cycle),
}

results = {}
for name, (agent_f, meta) in configs.items():
    print(f"[run] {name}")
    results[name] = run_pyramid_experiment(
        env_factory=env_factory,
        agent_factory=agent_f,
        meta_policy=meta,
        n_steps=N_STEPS,
        n_runs=N_RUNS,
        seed=SEED,
    )

print("\nИтоговые очки (mean ± std):")
for name, r in results.items():
    pts = r.cumulative_points()[:, -1]
    print(f"  {name:28s}  {pts.mean():9.1f} ± {pts.std():7.1f}")

# ----------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for name, r in results.items():
    axes[0, 0].plot(r.cumulative_points().mean(0), label=name)
    axes[0, 1].plot(r.n_arms.mean(0), label=name)
    axes[1, 0].plot(r.n_multipliers.mean(0), label=name)
    if r.epochs is not None:
        axes[1, 1].plot(r.epochs.mean(0), label=name)

axes[0, 0].set_title("Накопленные очки"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
axes[0, 1].set_title("Среднее число рук"); axes[0, 1].grid(alpha=0.3)
axes[1, 0].set_title("Среднее число множителей"); axes[1, 0].grid(alpha=0.3)
axes[1, 1].set_title("Средняя эпоха (прогресс метацикла)"); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("layered_results.png", dpi=120)
plt.show()