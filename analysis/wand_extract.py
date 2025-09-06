import wandb
from datetime import datetime, timezone

# Connect to wandb API
api = wandb.Api()

# Parameters
entity = "sayantan0013-epfl"        # e.g. "my-username" or org name
project = "fish-marl-cur-experiment"      # e.g. "my-project"

start_time_str = "2025-09-05 20:42:30"
end_time_str   = "2025-09-05 21:48:09"

time_start = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
time_end   = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

# Fetch runs
runs = api.runs(f"{entity}/{project}")

print(len(runs))
# Filter and sort
filtered_runs = sorted(
    [run for run in runs if time_start <=  datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")  <= time_end],
    key=lambda r: r.created_at
)

# Extract IDs
run_ids = [run.id for run in filtered_runs]

print(run_ids)
