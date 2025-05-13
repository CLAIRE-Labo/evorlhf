import os, sys

# ─── ensure 'src' is on PYTHONPATH ───────────────────────────
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
# ──────────────────────────────────────────────────────────────


import resource
from packing.parallel.memory_watcher import MemoryWatcher
import os, sys, logging

# 1) Set a hard cap on the total virtual address space.
#    Any malloc beyond this will raise MemoryError.
MEM_LIMIT_GB = 8
mem_bytes    = MEM_LIMIT_GB * 1024**3
# get the existing hard limit so we don’t shrink it below system defaults
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))

# callback when we exceed the limit
def _on_oom():
    logging.error("❌ Resource limit reached, exiting.")
    sys.stderr.write("❌ Resource limit reached, exiting.\n")
    # flush logs/streams
    for h in logging.root.handlers:
        h.flush()
    logging.shutdown()
    sys.stderr.flush()
    sys.exit(1)

# start watching (optional fallback)
watcher = MemoryWatcher(
    task_id="train",
    pid=os.getpid(),
    mem_limit_bytes=mem_bytes,
    callback_on_exceed=_on_oom
)
watcher.start()

# your Hydra entrypoint…
from hydra import main as _hydra_main

@_hydra_main(config_path="../../configs", config_name="config")
def wrapped(cfg):
    from src.experiments.main import main as real_main
    try:
        return real_main(cfg)
    except MemoryError:
        _on_oom()

if __name__ == "__main__":
    wrapped()