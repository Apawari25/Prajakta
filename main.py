from validation import validate_events
from calculation import gate_count, average_tat
from report import print_report

vehicle_events = [
    ("MH12AB1234", "Gate A", 15),
    ("MH14CD5678", "Gate B", 20),
    ("MH12AB1234", "Gate A", 10),
    ("", "Gate C", 12),
    ("MH20EF9876", "Gate B", 18),
]


def main():
    valid = validate_events(vehicle_events)

    counts = gate_count(valid)

    avg = average_tat(valid)

    print_report(counts, avg)


if __name__ == "__main__":
    main()