from models import VehicleEvent, DetectionResult

event = VehicleEvent("MH12AB1234", "Gate A", 15)

print(event)

print(event.is_valid())

result = DetectionResult("MH12AB1234", 0.95)

print(result)

print(result.is_high_confidence())