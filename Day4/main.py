import csv
import json
import logging
from pathlib import Path

logging.basicConfig(
    filename="rejected.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

clean_records = []

try:
    with open("vehicle_events.csv", "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["Vehicle"] and row["Gate"] and row["TAT"]:
                clean_records.append(row)
            else:
                logging.info(f"Rejected Row: {row}")

except FileNotFoundError:
    print("CSV file not found.")

else:
    print("CSV loaded successfully.")

finally:
    print("CSV processing completed.")

with open("clean_report.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["Vehicle", "Gate", "TAT"]
    )

    writer.writeheader()
    writer.writerows(clean_records)

print("Clean report exported.")

with open("vehicle_events.json", "r") as file:
    data = json.load(file)

print("\nJSON Records:")
for record in data:
    print(record)

print("\nCurrent Folder:")
print(Path.cwd())