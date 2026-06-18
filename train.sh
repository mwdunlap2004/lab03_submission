Create an interactive VS code instance
Field	Value for this lab	
Number of hours	4	
Number of cores	2	
Memory Request	30 GB	
Allocation	ds6042
Working Directory	/scratch/$USER	



Then in VSCode server, when your machine is spun up open the terminal


5.1 Clone into /scratch (not $HOME)
$ cd /scratch/$USER && mkdir -p lab03 && cd lab03
$ git clone https://github.com/researcher111/nanochat.git
Cloning into 'nanochat'...
Receiving objects: 100% (1,247/1,247), 4.8 MiB | 9.2 MiB/s, done.
$ cd nanochat




5.2 Load the right modules and create the environment

$ module spider gcc
# lists every available gcc version — pick the highest 13.x if there is one, else 11.x or 12.x is fine
$ module spider cuda
# do the same for CUDA — pick the highest 12.x
$ module load cuda/12.4 gcc/11.4.0
# above is a current-as-of-spring-2026 default; if Lmod errors with
# "The following module(s) are unknown", re-run module spider and
# substitute the version it prints.


For this lab I ran module load cuda/12.8.0 gcc/11.4.0



$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ source ~/.local/bin/env
$ cd /scratch/$USER/lab03/nanochat
$ uv sync --extra gpu
Using CPython 3.10.20
Creating virtual environment at: .venv
Resolved 130 packages in 2ms
Prepared 81 packages in 37.89s
Installed 81 packages in 3.47s
$ source .venv/bin/activate



With a lot of these steps if you have already done these steps they do not need to be repeated
In that case you just need these three lines
cd /scratch/$USER
cd lab03/nanochat
source .venv/bin/activate


In the original run we used depth =6, this time I tried Depth = 10
export DEPTH=10
export DEVICE_BATCH_SIZE=8         # MIG slice has limited VRAM; lower further if you OOM
export NCHAT_DATA=/scratch/$USER/lab03/data
mkdir -p "$NCHAT_DATA"


This should quickly run if the data is already in your cache
DOWNLOAD THE CORPUS
$ bash runs/speedrun.sh --download-only





Train the tokenizer
$ python -m scripts.tok_train --vocab-size 8192


With multiple users working on this same project we ran into issues with everyone trying the same port
If that error occurs you can swap the port number
Pretrain the base model
torchrun --master_port=29521 --nproc_per_node=1 -m scripts.base_train --depth $DEPTH --device-batch-size $DEVICE_BATCH_SIZE --num-iterations 2000 2>&1 | tee train.log


nano plot_loss.py
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

uv pip install matplotlib

python plot_loss.py

nano extract_metrics.py
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
	
python extract_metrics.py


6.3 Supervised fine-tune for chat

$ python -m scripts.chat_sft --num-iterations 800


6.4 (optional) Reinforcement learning

$ python -m scripts.chat_rl --model data/sft_model.pt

7.1 Chat in the terminal

$ python -m scripts.chat_cli 

7.2 (Optional) The browser chat UI

$ python -m scripts.chat_web --port 8000

