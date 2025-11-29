import json
import os


def dump_node_debug(state: dict, node_name: str, payload: dict) -> None:
    """Persist lightweight debug info for each node inside the data_dir."""
    data_dir = state.get("data_dir")
    if not data_dir:
        return
    path = os.path.join(data_dir, f"{node_name}_debug.json")
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            
        # Also append to a master log file for debugging flow
        log_path = os.path.join(data_dir, "debug_log.txt")
        with open(log_path, "a") as f:
            f.write(f"Node {node_name} completed. Keys: {list(payload.keys())}\n")
            
    except Exception:
        # Debug writing must not break the pipeline.
        pass

# Import node modules after helper is defined to avoid circular import issues
from .A_nodes import *
from .C_nodes import *
from .E_nodes import *
from .V_nodes import *
from . import lr_node
