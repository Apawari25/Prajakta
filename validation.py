def validate_events(events: list) -> list:
    """Return only valid vehicle events."""

    valid = []

    for vehicle, gate, tat in events:
        if vehicle != "":
            valid.append((vehicle, gate, tat))

    return valid


    if __name__ == "__main__":
        sample = [
        ("MH12AB1234", "Gate A", 15),
        ("", "Gate B", 20),
    ]

    print(validate_events(sample))