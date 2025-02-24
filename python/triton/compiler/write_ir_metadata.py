import subprocess
import os
import hashlib
import json
import inspect
from ..runtime.cache import default_dump_dir


def get_commit_hash():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=script_dir).decode().strip()
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_path).decode().strip()
    return commit_hash


def write_metadata(fn_cache_manager_key, src):
    """
    Write metadata about IRs generated in triton
    """
    metadata_dir = default_dump_dir() + "/IR_metadata"
    constants = src.constants
    kernel_src = src.fn.src
    kernel_name = src.name
    func = src.fn.fn
    src_file = inspect.getfile(func)
    line_no = inspect.getsourcelines(func)[1]
    commit_hash = get_commit_hash()
    metadata_filename = metadata_dir + "/" + commit_hash + ".json"
    if not os.path.exists(metadata_filename):
        os.makedirs(metadata_dir, exist_ok=True)
        data = {}
    else:
        # Load existing data from the JSON file
        try:
            with open(metadata_filename, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {metadata_filename} is corrupted. Creating a new file.")
            data = {}
    constants = str(constants)
    metadata_key = kernel_src + constants
    hashed_key = hashlib.sha256(metadata_key.encode("utf-8")).hexdigest()
    value_dict = {
        "fn_cache_manager_key": fn_cache_manager_key, "kernel_name": kernel_name, "definition_location":
        f"{src_file}:{line_no}", "constants": constants
    }
    data[hashed_key] = value_dict
    try:
        with open(metadata_filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error writing to {metadata_filename}: {e}")
