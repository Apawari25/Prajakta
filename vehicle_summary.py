vehicle_events = [
    ("MH12AB1234", "Gate A", 15),
    ("MH14CD5678", "Gate B", 20),
    ("MH12AB1234", "Gate A", 10),
    ("", "Gate C", 12),          # Invalid
    ("MH20EF9876", "Gate B", 18),
]

valid_events = []

for vehicle, gate, tat in vehicle_events:
    if vehicle != "":
        valid_events.append((vehicle, gate, tat))

print("Valid Events:")
for event in valid_events:
    print(event)


#-----Vehicles by Gate Count----

    gate_count = {}

for vehicle, gate, tat in valid_events:
    if gate in gate_count:
        gate_count[gate] += 1
    else:
        gate_count[gate] = 1

print("\nVehicle Count by Gate")
for gate, count in gate_count.items():
    print(gate, ":", count)


# ---- Average TAT Calculation---

    total = 0

for vehicle, gate, tat in valid_events:
    total += tat

average = total / len(valid_events)

print("\nAverage TAT =", average)

#----List Comprehension----

vehicles = [vehicle for vehicle, gate, tat in valid_events]

print(vehicles)


# ---- Set---
unique_vehicles = set(vehicles)

print(unique_vehicles)


#---- enumerate() ----
for index, vehicle in enumerate(unique_vehicles, start=1):
    print(index, vehicle)


#---- zip()---
gates = ["Gate A", "Gate B"]
counts = [2, 2]

for gate, count in zip(gates, counts):
    print(gate, count)
