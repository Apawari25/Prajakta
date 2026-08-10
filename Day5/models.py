from dataclasses import dataclass


@dataclass
class VehicleEvent:
    vehicle_number: str
    gate: str
    tat: int

    def is_valid(self) -> bool:
        return bool(
            self.vehicle_number
            and self.gate
            and self.tat >= 0
        )


@dataclass
class DetectionResult:
    vehicle_number: str
    confidence: float

    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.90