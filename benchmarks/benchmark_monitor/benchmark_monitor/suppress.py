"""Alert suppression applied AFTER the gate (the gate is never changed). Pure and stateless.

Everything is derived from the per-night verdict history, so a missed night or a rerun cannot desync state.
It reports new episodes only (a flag that continues is "sustained"), folds a storm into one digest line per
family, confirms an episode after ``confirm_nights`` nights, holds back flags of a partial run, and mutes
flip-flopping series. Every knob is configurable. 0 turns the matching rule off for ``storm_threshold``,
``min_coverage``, ``coverage_history`` (no baseline, so no partial-run check), ``max_gap_days``,
``flip_episodes`` and ``flip_days`` (no window, so no earlier episodes); ``confirm_nights`` has a minimum of 1 and
``flip_short_share`` has no off value. A source that does not run every night (a catalog scope with
``expected_nightly: false``) belongs in ``sparse_sources``, so that its data gaps are not read as a bad run.

Episodes that start in a storm are not confirmed one by one. The family is confirmed once, as a
``ConfirmedFamily``, on the first later data night on which its streak reaches ``confirm_nights`` and at least
one of its storm-started cases is still regressing (live, not muted). Like a case confirmation, it survives a
missed or partial night. A storm counts live flags only: if a broad regression first appears on a partial-run or
data-gap night, its flags are held, that night is not a storm night, and its cases are confirmed one by one.

A "source" is ``SeriesKey.source``. A "data night" of a source is a night with at least one non-missing
verdict, so a night with no report never makes a standing flag new again. A series' own ``missing`` nights
are skipped when following its episode.
"""

from __future__ import annotations

import functools
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import NamedTuple

from benchmark_monitor.model import SeriesKey, Status, Verdict

MIN_BASELINE_NIGHTS = 3  # earlier data nights needed before the partial-run guard trusts its median


def identity(benchmark: str) -> str:
    """Default family mapping: every benchmark is its own family."""
    return benchmark


@dataclass(frozen=True)
class SuppressConfig:  # pylint: disable=too-many-instance-attributes
    """Suppression knobs, validated on creation. ``confirm_nights`` below 1 is clamped to 1 (confirm on the
    first night); a negative count or a ratio outside 0..1 raises ``ValueError``."""

    storm_threshold: int = 10  # live new episodes of ONE family on one night
    confirm_nights: int = 2
    min_coverage: float = 0.7  # partial-run ratio: tonight's count / recent median
    coverage_history: int = 10
    max_gap_days: int = 3
    flip_episodes: int = 3
    flip_days: int = 28
    flip_short_share: float = 0.5
    family_of: Callable[[str], str] = identity  # Catalog.family_of in the pipeline
    sparse_sources: frozenset[str] = frozenset()  # sources not expected every night: no data-gap check

    def __post_init__(self) -> None:
        for name in ("storm_threshold", "coverage_history", "max_gap_days", "flip_episodes", "flip_days"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("min_coverage", "flip_short_share"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within 0..1, got {getattr(self, name)}")
        if self.confirm_nights < 1:
            object.__setattr__(self, "confirm_nights", 1)


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
class ConfirmedFamily:
    """A storm-started family confirmed once, on the first later night it is still regressing."""

    source: str
    family: str
    night: date  # the night the family is confirmed
    first_night: date  # the storm night
    keys: tuple[SeriesKey, ...]  # storm-started cases still regressing (live, not muted) on `night`


@dataclass(frozen=True)
class SuppressionResult:  # pylint: disable=too-many-instance-attributes
    """Tonight's suppressed view. ``new`` is what to alert; the rest is context."""

    new: tuple[Verdict, ...] = ()  # shown new episodes (storm families go to ``storms``, suspect sources to ``held``)
    confirmed: tuple[Confirmed, ...] = ()
    sustained: tuple[Verdict, ...] = ()
    muted: tuple[Verdict, ...] = ()  # new AND sustained flip-flop series (not also in ``sustained``)
    storms: tuple[StormDigest, ...] = ()
    storm_nights: frozenset[date] = frozenset()
    suspect: Mapping[str, str] = field(default_factory=dict)  # source -> reason its flags are held
    held: Mapping[str, int] = field(default_factory=dict)  # source -> new flags held back
    warnings: tuple[str, ...] = ()
    confirmed_families: tuple[ConfirmedFamily, ...] = ()

    @property
    def item_count(self) -> int:
        """Things worth a message: shown rows + digest lines + confirmed series + confirmed families."""
        keys = {v.key for v in self.new} | {c.key for c in self.confirmed}
        return len(keys) + len(self.storms) + len(self.confirmed_families)


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
        # every flagged night is also a checked night, so the data nights are the keys of `checked`
        self._data = {s: sorted(c) for s, c in self.checked.items()}

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


def _prior_episodes(h: _History, cfg: SuppressConfig, keys: Sequence[SeriesKey],
                    night: date | None = None) -> dict[SeriesKey, list[int]]:
    """Per series flagged on *night* (default tonight): lengths of its episodes that ENDED before that night's
    began, within the ``flip_days`` window ending on *night*."""
    night = night or h.tonight
    first = night - timedelta(days=cfg.flip_days - 1)
    return {
        k: _episodes(h, k, [d
                            for d in h.nights
                            if first <= d <= night and h.is_flagged(k, d)])[:-1]  # last run = `night`
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
    if cfg.max_gap_days > 0 and source not in cfg.sparse_sources and earlier and (gap := (
            night - earlier[-1]).days) > cfg.max_gap_days:
        return f"previous data night is {gap} days back"
    recent = [counts[d] for d in earlier[-cfg.coverage_history:]] if cfg.coverage_history > 0 else []
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
    return _Night(h, new, today, streaks, flip, suspect, live, storming)


class StormGroup(NamedTuple):
    """Cases of one family whose episodes began in the same storm."""

    source: str
    family: str
    night: date  # the storm night


def _storm_started(nt: _Night, cfg: SuppressConfig, storms: Mapping[date,
                                                                    frozenset[str]]) -> dict[StormGroup, list[Verdict]]:
    """Cases regressing live on *nt*'s night whose episode began, on an earlier night, in a storm of their family."""
    out: dict[StormGroup, list[Verdict]] = defaultdict(list)
    for v in nt.today:
        if v.key in nt.flip or v.key.source in nt.suspect:
            continue
        start = nt.h.episode(v.key)[0]
        fam = cfg.family_of(v.key.benchmark)
        if start < nt.h.tonight and fam in storms.get(start, ()):
            out[StormGroup(v.key.source, fam, start)].append(v)
    return out


def _qualifies(nt: _Night, cfg: SuppressConfig, vs: Sequence[Verdict]) -> bool:
    return max(nt.streaks[v.key] for v in vs) >= cfg.confirm_nights


def _confirmable(h: _History, cfg: SuppressConfig, nights: Mapping[date, _Night], storms: Mapping[date, frozenset[str]],
                 key: SeriesKey, ep: tuple[date, ...], i: int) -> bool:
    """Whether night ``ep[i]`` of *key*'s episode could have confirmed it, judged on that night's own view."""
    fam = cfg.family_of(key.benchmark)
    return (fam not in storms[ep[0]] and key.source not in nights[ep[i]].suspect
            and not is_flip_flop(_prior_episodes(h, cfg, [key], ep[i])[key], i + 1, cfg))


def apply(history: Mapping[date, Sequence[Verdict]], cfg: SuppressConfig | None, tonight: date) -> SuppressionResult:
    """Suppress and group the verdicts of *tonight* given the per-night verdict *history*."""
    cfg = cfg or SuppressConfig()
    n = _night(history, cfg, tonight)
    h, suspect = n.h, n.suspect
    # One storm definition everywhere: >= storm_threshold LIVE new flags of one family on one night.
    nights = {d: _night(history, cfg, d) for d in h.nights if d != tonight}
    nights[tonight] = n
    storms_by_night = {d: nt.storm_families for d, nt in nights.items()}
    storm_nights = frozenset(d for d, fams in storms_by_night.items() if fams)
    shown = [v for v in n.live if cfg.family_of(v.key.benchmark) not in n.storm_families]
    folded = [v for v in n.live if cfg.family_of(v.key.benchmark) in n.storm_families]
    held = {s: c for s in suspect if (c := sum(1 for v in n.new if v.key.source == s))}

    confirmed = []
    for v in n.today:
        if n.streaks[v.key] < cfg.confirm_nights:
            continue
        ep = h.episode(v.key)

        ok = functools.partial(_confirmable, h, cfg, nights, storms_by_night, v.key, tuple(ep))
        last = len(ep) - 1
        if ok(last) and not any(ok(i) for i in range(cfg.confirm_nights - 1, last)):
            confirmed.append(Confirmed(v.key, v, n.streaks[v.key], ep[0]))

    # A storm-started family is confirmed once: tonight qualifies and no earlier night did.
    families = []
    seen = {  # (source, family, storm night) groups that already qualified on an earlier night
        g
        for d, nt in nights.items() if d < tonight for g, vs in _storm_started(nt, cfg, storms_by_night).items()
        if _qualifies(nt, cfg, vs)
    }
    for group, vs in sorted(_storm_started(n, cfg, storms_by_night).items(), key=lambda kv: kv[0]):
        if group in seen or not _qualifies(n, cfg, vs):
            continue
        keys = tuple(
            sorted((v.key for v in vs), key=lambda k: (k.benchmark, k.params, k.platform, k.tag, k.compiler, k.metric)))
        families.append(ConfirmedFamily(group.source, group.family, tonight, group.night, keys))

    warnings = [
        f"{s}: latest data night is {max(c)}, not {tonight} (no report for that night)"
        for s, c in sorted(h.checked.items())
        if c and max(c) < tonight
    ]
    warnings += [f"{s}: {held[s]} new flag(s) held back, {r}" for s, r in sorted(suspect.items()) if s in held]
    linked = {c.key for c in confirmed}
    sustained = tuple(v for v in n.today if v not in n.new and v.key not in linked and v.key not in n.flip)
    return SuppressionResult(new=tuple(shown), confirmed=tuple(confirmed), sustained=sustained,
                             muted=tuple(v for v in n.today
                                         if v.key in n.flip), storms=_digest(folded, cfg), storm_nights=storm_nights,
                             suspect=suspect, held=held, warnings=tuple(warnings), confirmed_families=tuple(families))
