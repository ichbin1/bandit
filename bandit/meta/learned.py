"""Обучаемая мета-политика: мета-бандит с отложенной наградой (P4).

Постановка
----------
Политика решает не «какую руку тянуть», а «инвестировать сейчас или
добывать». Решение принимается в любом шаге, где нет незакрытого окна
измерения; в остальные шаги политика возвращает ``None`` и делегирует выбор
руки агенту. Опции решения — мета-действия, которые среда сейчас разрешает
(``Observation.can_*``), поэтому политика никогда не тратит шаг на заведомо
неэффективное действие.

Воздержание — не отдельная опция, а *базовый уровень*: награда 0 означает
«доходность не изменилась», то есть инвестиция ничего не дала. Политика
инвестирует, только если лучшая UCB-оценка выше нуля. Это важно для
экономики шага: если бы воздержание занимало окно измерения, политика
упускала бы инвестиции просто из-за расписания решений, а не из-за их
ценности. Дополнительно ведутся счётчики «визитов» в контекст, чтобы бонус
исследования рос со временем и политика не застревала в воздержании.

Отложенная награда
------------------
Мета-действие не приносит очков в момент применения: выгода появляется
позже, в виде выросшего множителя или новых рук. Поэтому награда решения —
сравнение *доходности* соседних окон:

``reward = log((y_next + eps) / (y_prev + eps))``

где ``y`` — очки за шаг в окне. Почему именно лог-отношение:

* окно одной длины у всех решений делает награды сравнимыми — иначе
  мета-шаг (0 очков) всегда проигрывал бы шагу добычи;
* доходность растёт мультипликативно, поэтому абсолютные разности
  «плывут» вместе с множителем и с эпохой, а лог-отношение от масштаба
  не зависит;
* сравнение с *соседним* окном, а не со средним по всей истории, убирает
  общий тренд роста: иначе воздержание в поздней игре выглядело бы
  «прибыльным» просто потому, что средняя доходность за прошлое ниже
  текущей, и политика выучила бы «никогда не инвестировать».

Цена честная: мета-шаг занимает один шаг из окна и уменьшает его
доходность — это входит в сигнал и не вычитается отдельно.

Контекст
--------
Состояние сжимается до ``(эпоха, класс доступных инвестиций)``:

* класс 2 — доступны «ворота» или амплификатор (инвестиции в потолок);
* класс 1 — доступны только множители и покупка рук.

Статистика UCB ведётся отдельно по контексту: ценность «ворот» в эпохе 0 и
в эпохе 2 разная, и усреднять их нельзя.

Почему UCB и почему без RNG
---------------------------
Выбор детерминирован: неисследованные опции перебираются в каноническом
порядке (по id действия), ничьи разрешаются меньшим id. Значит, прогон
воспроизводим без отдельного сида и совпадает при ``--jobs 1`` и
``--jobs 2`` (важно для Windows/spawn).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from ..envs.base import MetaAction, Observation, meta_action_id
from .base import BasePolicy

ABSTAIN = -1
"""Id опции «воздержаться»: политика возвращает ``None`` и выбирает агент."""

CONTEXT_LABELS: Dict[int, str] = {
    1: "множители/руки",
    2: "технологии",
}
"""Человекочитаемые названия классов доступных инвестиций."""

_YIELD_FLOOR = 1e-3
"""Порог в логарифме: окно без очков не должно давать ``log(0)``."""

_OPTION_KINDS: Tuple[Tuple[MetaAction, str], ...] = (
    (MetaAction.INFO_TO_MULTIPLIER, "can_convert_info"),
    (MetaAction.BUY_ARM, "can_buy_arm"),
    (MetaAction.CRAFT_GATE, "can_craft_gate"),
    (MetaAction.CRAFT_AMPLIFIER, "can_craft_amplifier"),
)


class LearnedMeta(BasePolicy):
    """UCB-мета-бандит: учится, когда инвестировать и каким способом.

    Parameters
    ----------
    c:
        Коэффициент исследования в бонусе UCB.
    window:
        Горизонт измерения награды: через сколько шагов после инвестиции
        сравнивается доходность окон.
    decay:
        Вес забывания: ``1.0`` — обычное среднее, меньше — забываем старое.
    min_trials:
        Сколько наблюдений нужно опции, прежде чем сравнивать её по UCB.
    """

    name = "learned"

    def __init__(
        self,
        c: float = 1.0,
        window: int = 5,
        decay: float = 1.0,
        min_trials: int = 1,
    ) -> None:
        if int(window) < 1:
            raise ValueError("window должен быть положительным")
        if float(c) < 0.0:
            raise ValueError("c не может быть отрицательным")
        if not 0.0 < float(decay) <= 1.0:
            raise ValueError("decay должен лежать в (0, 1]")
        self.c = float(c)
        self.window = int(window)
        self.decay = float(decay)
        self.min_trials = max(1, int(min_trials))
        #: контекст → опция → [число наблюдений, средняя награда]
        self._stats: Dict[Tuple[int, int], Dict[int, List[float]]] = {}
        #: контекст → число рассмотренных решений (для бонуса исследования)
        self._visits: Dict[Tuple[int, int], int] = {}
        #: (опция, контекст, шаг решения, очки, базовая доходность до решения)
        self._pending: Optional[
            Tuple[int, Tuple[int, int], int, float, Optional[float]]
        ] = None
        #: последние ``(шаг, очки)`` — локальная база для лог-отношения
        self._history: Deque[Tuple[int, float]] = deque(maxlen=self.window + 2)
        #: скользящее среднее доходности окон (только диагностика)
        self._yield_mean: float = 0.0
        self._yield_n: int = 0
        self.n_decisions: int = 0
        self.n_invest: int = 0

    def __call__(self, obs: Observation) -> Optional[int]:
        self._settle(obs)
        t = int(obs.t)
        points = float(obs.points)
        self._history.append((t, points))
        if self._pending is not None:
            return None
        options = self._options(obs)
        if not options:
            return None
        context = self._context(obs)
        option = self._select(context, options)
        self.n_decisions += 1
        if option == ABSTAIN:
            return None
        self._pending = (option, context, t, points, self._yield_before(t, points))
        self.n_invest += 1
        return int(option)

    # ------------------------------------------------------------------
    # Награда: лог-отношение доходности соседних окон
    # ------------------------------------------------------------------
    def _settle(self, obs: Observation) -> None:
        """Закрыть прошлое окно и начислить награду его решению."""
        if self._pending is None:
            return
        option, context, start_t, start_points, baseline = self._pending
        elapsed = int(obs.t) - start_t
        if elapsed < self.window:
            return
        self._pending = None
        current = (float(obs.points) - start_points) / float(elapsed)
        self._yield_n += 1
        self._yield_mean += (current - self._yield_mean) / float(self._yield_n)
        if baseline is None:
            reward = 0.0  # базы ещё нет: окно калибровочное
        else:
            reward = math.log((current + _YIELD_FLOOR) / (baseline + _YIELD_FLOOR))
        self._update(context, option, reward)

    def _yield_before(self, t: int, points: float) -> Optional[float]:
        """Доходность окна непосредственно *до* решения (локальная база)."""
        for step, old_points in self._history:
            if step <= t - self.window:
                return (points - old_points) / float(t - step)
        return None

    def _update(self, context: Tuple[int, int], option: int, reward: float) -> None:
        table = self._stats.setdefault(context, {})
        entry = table.get(option)
        if entry is None:
            table[option] = [1.0, float(reward)]
        elif self.decay < 1.0:
            entry[1] = self.decay * entry[1] + (1.0 - self.decay) * float(reward)
            entry[0] += 1.0
        else:
            entry[1] += (float(reward) - entry[1]) / (entry[0] + 1.0)
            entry[0] += 1.0

    # ------------------------------------------------------------------
    # Выбор
    # ------------------------------------------------------------------
    def _select(self, context: Tuple[int, int], options: List[int]) -> int:
        table = self._stats.setdefault(context, {})
        visits = self._visits.get(context, 0) + 1
        self._visits[context] = visits
        for option in sorted(options):
            entry = table.get(option)
            if entry is None or entry[0] < self.min_trials:
                return int(option)
        best_option = ABSTAIN
        best_score = 0.0  # ценность воздержания: доходность не меняется
        for option in options:
            n, mean = table[option]
            bonus = self.c * math.sqrt(math.log(visits + 1.0) / max(1.0, n))
            score = mean + bonus
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and best_option != ABSTAIN
                and option < best_option
            ):
                best_option, best_score = option, score
        return int(best_option)

    @staticmethod
    def _context(obs: Observation) -> Tuple[int, int]:
        """Сжать состояние до ``(эпоха, класс доступных инвестиций)``."""
        if obs.can_craft_gate or obs.can_craft_amplifier:
            bucket = 2
        else:
            bucket = 1
        return (int(obs.epoch), bucket)

    @staticmethod
    def _options(obs: Observation) -> List[int]:
        """Только разрешённые средой мета-действия (воздержание — база)."""
        options = []
        for kind, flag in _OPTION_KINDS:
            if getattr(obs, flag):
                options.append(meta_action_id(kind))
        return options

    # ------------------------------------------------------------------
    # Диагностика (тесты, отчёты)
    # ------------------------------------------------------------------
    def table(self) -> Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]:
        """Статистика в читаемом виде: ``(эпоха, класс) → опция → (n, mean)``."""
        result: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
        for (epoch, bucket), entries in sorted(self._stats.items()):
            key = (f"epoch {epoch}", CONTEXT_LABELS[bucket])
            result[key] = {
                ("abstain" if option == ABSTAIN else f"meta {option}"): (
                    float(n),
                    float(mean),
                )
                for option, (n, mean) in sorted(entries.items())
            }
        return result

    def summary(self) -> Dict[str, float]:
        """Сводка обучения — одинаковая по ключам для любых прогонов."""
        return {
            "decisions": float(self.n_decisions),
            "invested": float(self.n_invest),
            "invest_share": (
                float(self.n_invest) / float(self.n_decisions) if self.n_decisions else 0.0
            ),
            "contexts": float(len(self._stats)),
            "yield_mean": float(self._yield_mean),
        }


__all__ = ["ABSTAIN", "CONTEXT_LABELS", "LearnedMeta"]
