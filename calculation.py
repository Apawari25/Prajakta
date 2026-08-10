def gate_count(events: list) -> dict:
    """Count vehicles gate wise."""

    counts = {}

    for vehicle, gate, tat in events:
        counts[gate] = counts.get(gate, 0) + 1

    return counts


def average_tat(events: list) -> float:
    """Calculate average TAT."""

    total = sum(tat for _, _, tat in events)

    return total / len(events)