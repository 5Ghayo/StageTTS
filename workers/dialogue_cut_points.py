"""Select dialogue cuts from DiariZen's frame-level activity."""

from statistics import median

ACOUSTIC_SEARCH_SECONDS = 1.0
MAX_VOTE_SPREAD_SECONDS = 0.06

# Events are timestamped halfway between consecutive frame centers. The
# active slot numbers are meaningful only within their own inference window.
ActivityEvent = tuple[float, list[int], list[int]]


def select_acoustic_cut(
    windows: list[list[ActivityEvent]], target: float, start: float, end: float,
) -> tuple[float | None, str | None, list[float]]:
    """Use a pause or speaker-change midpoint supported by multiple windows."""
    votes = []
    for events in windows:
        candidates = []
        for (left, before, silent), (right, still_silent, after) in zip(events, events[1:]):
            if before and not silent and not still_silent and after and start < left < right < end:
                distance = max(left - target, target - right, 0.0)
                candidates.append((distance, (left + right) / 2, "pause"))
        for time, before, after in events:
            if start < time < end and len(before) == len(after) == 1 and before[0] != after[0]:
                candidates.append((abs(time - target), time, "speaker_change"))
        if candidates:
            distance, time, kind = min(candidates, key=lambda item: (item[0], abs(item[1] - target)))
            if distance <= ACOUSTIC_SEARCH_SECONDS:
                votes.append((round(time, 4), kind))

    for kind in ("pause", "speaker_change"):
        times = [time for time, vote_kind in votes if vote_kind == kind]
        if len(times) >= 2 and max(times) - min(times) <= MAX_VOTE_SPREAD_SECONDS:
            center = median(times)
            # Stay on a model-observed midpoint even when the vote count is even.
            return min(times, key=lambda time: abs(time - center)), kind, times
    return None, None, [time for time, _ in votes]
