def print_report(counts: dict, avg: float) -> None:
    """Print final report."""

    print("\nVehicle Count by Gate")

    for gate, count in counts.items():
        print(f"{gate}: {count}")

    print(f"\nAverage TAT: {avg:.2f}")