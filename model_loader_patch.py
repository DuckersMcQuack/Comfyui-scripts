import comfy.model_management
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import folder_paths
import os

def thread_load_model_files(model_type):
    paths = folder_paths.get_folder_paths(model_type)
    files = []
    for path in paths:
        if not os.path.isdir(path):
            continue
        for name in os.listdir(path):
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path):
                files.append(full_path)

    def load_file(path):
        try:
            comfy.model_management.load_model(path, model_type)
            return (path, True)
        except Exception as e:
            return (path, False, str(e))

    logging.info(f"[THREAD] Loading {len(files)} {model_type} models...")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(load_file, f): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            if result[1]:
                logging.info(f"[{model_type.upper()}] Loaded {result[0]}")
            else:
                logging.warning(f"[{model_type.upper()}] Failed {result[0]} - {result[2]}")

def patch_model_loading():
    for model_type in ["checkpoints", "vae", "loras", "clip"]:
        thread_load_model_files(model_type)