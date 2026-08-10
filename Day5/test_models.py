import pytest

from models import DetectionResult, VehicleEvent


@pytest.fixture
def valid_vehicle():
    return VehicleEvent("MH12AB1234", "Gate A", 15)


def test_vehicle_valid(valid_vehicle):
    assert valid_vehicle.is_valid() is True


def test_vehicle_number_missing():
    vehicle = VehicleEvent("", "Gate A", 15)
    assert vehicle.is_valid() is False


def test_gate_missing():
    vehicle = VehicleEvent("MH12AB1234", "", 15)
    assert vehicle.is_valid() is False


def test_negative_tat():
    vehicle = VehicleEvent("MH12AB1234", "Gate A", -1)
    assert vehicle.is_valid() is False


def test_high_confidence():
    result = DetectionResult("MH12AB1234", 0.95)
    assert result.is_high_confidence() is True


def test_low_confidence():
    result = DetectionResult("MH12AB1234", 0.40)
    assert result.is_high_confidence() is False


def test_boundary_confidence():
    result = DetectionResult("MH12AB1234", 0.90)
    assert result.is_high_confidence() is True