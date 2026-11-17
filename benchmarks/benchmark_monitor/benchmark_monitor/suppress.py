"""Alert suppression applied AFTER the gate (the gate is never changed). Pure and stateless.

Everything is derived from the per-night verdict history, so a missed night or a rerun cannot desync state.
It reports new episodes only (a flag that continues is "sustained"), folds a storm into one digest line per
family, confirms an episode after ``confirm_nights`` nights, holds back flags of a partial run, and mutes
flip-flopping series. Every knob is configurable; 0 (1 for ``confirm_nights``) turns the matching rule off.

A "source" is ``SeriesKey.source``. A "data night" of a source is a night with at least one non-missing
verdict, so a night with no report never makes a standing flag new again. A series' own ``missing`` nights
are skipped when following its episode.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from benchmark_monitor.model import SeriesKey, Status, Verdict

MIN_BASELINE_NIGHTS = 3  # earlier data nights needed before the partial-run guard trusts its median


def identity(benchmark: str) -> str:
    """Default family mapping: every benchmark is its own family."""
    return benchmark


@dataclass(frozen=True)
class SuppressConfig:  # pylint: disable=too-many-instance-attributes
    """Suppression knobs. ``confirm_nights`` has a minimum of 1 (confirm on the first night)."""

    storm_threshold: int = 10  # live new episodes of ONE family on one night
    confirm_nights: int = 2
    min_coverage: float = 0.7  # partial-run ratio: tonight's count / recent median
    coverage_history: int = 10
    max_gap_days: int = 3
    flip_episodes: int = 3
    flip_days: int = 28
    flip_short_share: float = 0.5
    family_of: Callable[[str], str] = identity  # Catalog.family_of in the pipeline


@dataclass(frozen=True)
class StormDigest:
    """One storm line: how many new episodes one family contributed."""

    family: str
    count: int
    median_drop: float  # percent, positive = worse
    worst_drop: float


@dataclass(frozen=True)
class Confirmed:
    """A confirmed episode. Issue-link prefilling is a consumer concern and hangs off this record."""

    key: SeriesKey
    verdict: Verdict
    streak: int
    first_night: date


@dataclass(frozen=True)
class SuppressionResult:  # pylint: disable=too-many-instance-attributes
    """Tonight's suppressed view. ``new`` is what to alert; the rest is context."""

    new: tuple[Verdict, ...] = ()  # shown new episodes (empty during a storm / for suspect sources)
    confirmed: tuple[Confirmed, ...] = ()
    sustained: tuple[Verdict, ...] = ()
    muted: tuple[Verdict, ...] = ()  # new AND sustained flip-flop series
    storms: tuple[StormDigest, ...] = ()
    storm_nights: frozenset[date] = frozenset()
    suspect: Mapping[str, str] = field(default_factory=dict)  # source -> reason its flags are held
    held: Mapping[str, int] = field(default_factory=dict)  # source -> new flags held back
    warnings: tuple[str, ...] = ()

    @property
    def item_count(self) -> int:
        """Things worth a message: shown rows + digest lines + confirmed series."""
        keys = {v.key for v in self.new} | {c.key for c in self.confirmed}
        return len(keys) + len(self.storms)


class _History:
    """Flags, data nights and the per-source "previous night" derived from the verdict history."""

    def __init__(self, history: Mapping[date, Sequence[Verdict]], tonight: date) -> None:
        self.tonight = tonight
        self.nights = sorted(d for d in history if d <= tonight)
        self.flags: dict[date, dict[SeriesKey, Verdict]] = {
            d: {v.key: v
                for v in history[d]
                if v.status is Status.REGRESSION}
            for d in self.nights
        }
        self.missing: dict[date, set[SeriesKey]] = {
            d: {v.key
                for v in history[d]
                if v.status is Status.MISSING}
            for d in self.nights
        }
        self.checked: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
        for d in self.nights:
            for v in history[d]:
                if v.status is not Status.MISSING:
                    self.checked[v.key.source][d] += 1
        self._data = {
            s:
            sorted({d
                    for d, n in c.items()
                    if n > 0} | {d
                                 for d in self.nights
                                 if any(k.source == s
                                        for k in self.flags[d])})
            for s, c in self.checked.items()
        }

    def previous(self, source: str, night: date, key: SeriesKey | None = None) -> date:
        """Previous data night of *source* before *night* (calendar day when it has no data).

        With *key*, nights where that series' own verdict is ``missing`` are skipped too.
        """
        earlier = [n for n in self._data.get(source, ()) if n < night and (key is None or key not in self.missing[n])]
        return earlier[-1] if earlier else night - timedelta(days=1)

    def is_flagged(self, key: SeriesKey, night: date) -> bool:
        return key in self.flags.get(night, {})

    def streak(self, key: SeriesKey) -> int:
        n, cursor = 1, self.previous(key.source, self.tonight, key)
        while self.is_flagged(key, cursor):
            n, cursor = n + 1, self.previous(key.source, cursor, key)
        return n

    def episode(self, key: SeriesKey) -> list[date]:
        """Data nights of the current episode, ascending, ending tonight."""
        nights = [self.tonight]
        while self.is_flagged(key, prev := self.previous(key.source, nights[-1], key)):
            nights.append(prev)
        return nights[::-1]

    def split(self) -> tuple[list[Verdict], list[Verdict]]:
        """Tonight's (new, sustained) flags, worst drop first."""
        new, sustained = [], []
        for v in self.flags.get(self.tonight, {}).values():
            (sustained if self.is_flagged(v.key, self.previous(v.key.source, self.tonight, v.key)) else new).append(v)
        order = lambda v: (-(v.drop_pct or 0.0), v.key.benchmark, v.key.params)  # noqa: E731
        return sorted(new, key=order), sorted(sustained, key=order)


def _episodes(h: _History, key: SeriesKey, nights: list[date]) -> list[int]:
    """Lengths of the runs of consecutive data nights in *nights* (sorted)."""
    lengths: list[int] = []
    prev: date | None = None
    for n in nights:
        if prev is not None and h.previous(key.source, n, key) == prev:
            lengths[-1] += 1
        else:
            lengths.append(1)
        prev = n
    return lengths


def _prior_episodes(h: _History, cfg: SuppressConfig, keys: Sequence[SeriesKey]) -> dict[SeriesKey, list[int]]:
    """Per series flagged tonight: lengths of its episodes that ENDED before tonight's began (in ``flip_days``)."""
    first = h.tonight - timedelta(days=cfg.flip_days - 1)
    return {
        k: _episodes(h, k, [d
                            for d in h.nights
                            if d >= first and h.is_flagged(k, d)])[:-1]  # last run = tonight
        for k in keys
    }


def is_flip_flop(prior: list[int], streak: int, cfg: SuppressConfig) -> bool:
    """Whether a series with earlier episodes *prior* and a current *streak* is muted."""
    if cfg.flip_episodes <= 0 or len(prior) < cfg.flip_episodes:
        return False
    if sum(1 for n in prior if n == 1) / len(prior) < cfg.flip_short_share:
        return False
    return streak <= max(prior) and streak <= cfg.confirm_nights


def suspect_reason(h: _History, source: str, night: date, cfg: SuppressConfig) -> str | None:
    """Why *source*'s flags for *night* should not be trusted (partial run or data gap), or None."""
    if cfg.min_coverage <= 0 and cfg.max_gap_days <= 0:
        return None
    counts = h.checked.get(source, {})
    tonight = counts.get(night, 0)
    if tonight <= 0:
        return None
    earlier = sorted(d for d, n in counts.items() if d < night and n > 0)
    if cfg.max_gap_days > 0 and earlier and (gap := (night - earlier[-1]).days) > cfg.max_gap_days:
        return f"previous data night is {gap} days back"
    recent = [counts[d] for d in earlier[-cfg.coverage_history:]]
    if cfg.min_coverage > 0 and len(recent) >= MIN_BASELINE_NIGHTS:
        median = statistics.median(recent)
        if tonight < cfg.min_coverage * median:
            return f"only {tonight} series ran tonight vs a recent median of {median:.0f} (partial run?)"
    return None


def _digest(flags: Sequence[Verdict], cfg: SuppressConfig) -> tuple[StormDigest, ...]:
    groups: dict[str, list[Verdict]] = defaultdict(list)
    for v in flags:
        groups[cfg.family_of(v.key.benchmark)].append(v)
    lines = [
        StormDigest(fam, len(vs), statistics.median(v.drop_pct or 0.0
                                                    for v in vs), max(v.drop_pct or 0.0
                                                                      for v in vs))
        for fam, vs in groups.items()
    ]
    return tuple(sorted(lines, key=lambda s: (-s.count, s.family)))


@dataclass
class _Night:  # pylint: disable=too-many-instance-attributes
    """One night's view: its flags split into new and sustained, and which new flags are live."""

    h: _History
    new: list[Verdict]
    today: list[Verdict]
    streaks: dict[SeriesKey, int]
    prior: dict[SeriesKey, list[int]]
    flip: set[SeriesKey]
    suspect: dict[str, str]
    live: list[Verdict]  # new, not muted, from a source that is not suspect
    storm_families: frozenset[str]  # families with >= storm_threshold live new flags


def _night(history: Mapping[date, Sequence[Verdict]], cfg: SuppressConfig, night: date) -> _Night:
    h = _History(history, night)
    new, sustained = h.split()
    today = [*new, *sustained]
    streaks = {v.key: h.streak(v.key) for v in today}
    prior = _prior_episodes(h, cfg, [v.key for v in today])
    flip = {k for k in streaks if is_flip_flop(prior[k], streaks[k], cfg)}
    suspect = {s: r for s in h.checked if (r := suspect_reason(h, s, night, cfg))}
    live = [v for v in new if v.key not in flip and v.key.source not in suspect]
    counts = Counter(cfg.family_of(v.key.benchmark) for v in live)
    storming = frozenset(f for f, n in counts.items() if cfg.storm_threshold > 0 and n >= cfg.storm_threshold)
    return _Night(h, new, today, streaks, prior, flip, suspect, live, storming)


def apply(history: Mapping[date, Sequence[Verdict]], cfg: SuppressConfig | None, tonight: date) -> SuppressionResult:
    """Suppress and group the verdicts of *tonight* given the per-night verdict *history*."""
    cfg = replace(cfg or SuppressConfig(), confirm_nights=max(1, (cfg or SuppressConfig()).confirm_nights))
    n = _night(history, cfg, tonight)
    h, suspect = n.h, n.suspect
    # One storm definition everywhere: >= storm_threshold LIVE new flags of one family on one night.
    storms_by_night = {d: _night(history, cfg, d).storm_families for d in h.nights[1:]}
    storms_by_night[tonight] = n.storm_families
    storm_nights = frozenset(d for d, fams in storms_by_night.items() if fams)
    shown = [v for v in n.live if cfg.family_of(v.key.benchmark) not in n.storm_families]
    folded = [v for v in n.live if cfg.family_of(v.key.benchmark) in n.storm_families]
    held = {s: c for s in suspect if (c := sum(1 for v in n.new if v.key.source == s))}

    confirmed = []
    for v in n.today:
        if n.streaks[v.key] < cfg.confirm_nights:
            continue
        ep = h.episode(v.key)
        fam = cfg.family_of(v.key.benchmark)
        ok = lambda i, v=v, ep=ep, fam=fam: (  # noqa: E731
            fam not in storms_by_night.get(ep[0], ()) and suspect_reason(h, v.key.source, ep[i], cfg) is None and
            not is_flip_flop(n.prior[v.key], i + 1, cfg))
        last = len(ep) - 1
        if ok(last) and not any(ok(i) for i in range(cfg.confirm_nights - 1, last)):
            confirmed.append(Confirmed(v.key, v, n.streaks[v.key], ep[0]))

    warnings = [
        f"{s}: latest data night is {max(d)}, not {tonight} (no report for that night)"
        for s, c in sorted(h.checked.items())
        if (d := [k for k, c2 in c.items() if c2 > 0]) and max(d) < tonight
    ]
    warnings += [f"{s}: {held[s]} new flag(s) held back, {r}" for s, r in sorted(suspect.items()) if s in held]
    linked = {c.key for c in confirmed}
    sustained = tuple(v for v in n.today if v not in n.new and v.key not in linked)
    return SuppressionResult(new=tuple(shown), confirmed=tuple(confirmed), sustained=sustained,
                             muted=tuple(v for v in n.today if v.key in n.flip), storms=_digest(folded, cfg),
                             storm_nights=storm_nights, suspect=suspect, held=held, warnings=tuple(warnings))
