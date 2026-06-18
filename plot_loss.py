import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps, losses = [], []
for line in open("train.log"):
    m = re.search(r"step\s+(\d+)/\d+.*?loss:\s*([\d.]+)", line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))

plt.figure(figsize=(7, 4))
plt.plot(steps, losses, lw=1)
plt.xlabel("step")
plt.ylabel("loss")
plt.title(f"base_train depth 10 - {len(steps)} steps")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss.png", dpi=120)
print(f"wrote loss.png ({len(steps)} points)")
