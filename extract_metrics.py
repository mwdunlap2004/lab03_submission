import re
import csv
import os

log_file = "train.log"
csv_file = "metrics.csv"

# Updated pattern to match: "step 04025/05000 (80.50%) loss: 2.901849"
pattern = re.compile(r"step\s+(\d+)/\d+.*?loss:\s*([\d.]+)")

metrics = []

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                metrics.append({"step": step, "loss": loss})
                
    if metrics:
        # Optional: If there are thousands of points, you might want to only save 
        # a subset (e.g., every 100th step) to keep the CSV small. 
        # For now, this saves everything it finds.
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss"])
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Successfully extracted {len(metrics)} data points to {csv_file}")
    else:
        print("Log file found, but no metric lines matched. If training just started, wait a minute for logs to flush.")
else:
    print(f"Error: {log_file} not found. Ensure the training job has initiated.")
