import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
from comfy.cli_args import args
from app.logger import setup_logger
import itertools
import utils.extra_config
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

def apply_custom_paths():
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models", os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)

def execute_prestartup_script():
    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return (script_path, True)
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
            return (script_path, False)

    if args.disable_all_custom_nodes:
        return

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    script_paths = []
    for custom_node_path in node_paths:
        for possible_module in os.listdir(custom_node_path):
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue
            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                script_paths.append(script_path)

    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(execute_script, script_paths))

    if results:
        logging.info("\nPrestartup times for custom nodes:")
        for script_path, success in results:
            import_message = "" if success else " (PRESTARTUP FAILED)"
            logging.info(f"{script_path}{import_message}")

apply_custom_paths()
execute_prestartup_script()

import asyncio
import shutil
import threading
import gc

if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    if args.deterministic and 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

if args.windows_standalone_build:
    try:
        from fix_torch import fix_pytorch_libomp
        fix_pytorch_libomp()
    except:
        pass

import comfy.utils
import execution
import server
from server import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
import hook_breaker_ac10a0

# Import threaded model loader patch
import model_loader_patch
model_loader_patch.patch_model_loading()

def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    if "cudaMallocAsync" in device_name and any(b in device_name for b in cuda_malloc.blacklist):
        logging.warning("\nWARNING: this card most likely does not support cuda-malloc. Run with: --disable-cuda-malloc\n")

def prompt_worker(q, server_instance):
    current_time = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.DEPENDENCY_AWARE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_size=args.cache_lru)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0 if not need_gc else max(gc_collect_interval - (current_time - last_gc_collect), 0.0)
        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id
            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id, e.history_result, status=execution.PromptQueue.ExecutionStatus(
                status_str='success' if e.success else 'error',
                completed=e.success,
                messages=e.status_messages))

            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            logging.info("Prompt executed in {:.2f} seconds".format(current_time - execution_start_time))

        flags = q.get_flags()
        if flags.get("unload_models", flags.get("free_memory", False)):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if flags.get("free_memory"):
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc and (time.perf_counter() - last_gc_collect) > gc_collect_interval:
            gc.collect()
            comfy.model_management.soft_empty_cache()
            last_gc_collect = time.perf_counter()
            need_gc = False
            hook_breaker_ac10a0.restore_functions()

async def run(server_instance, address='', port=8189, verbose=True, call_on_start=None):
    addresses = [(addr, port) for addr in address.split(",")]
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose),
        server_instance.publish_loop()
    )

def hijack_progress(server_instance):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server_instance.last_prompt_id, "node": server_instance.last_node_id}
        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            server_instance.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server_instance.client_id)
    comfy.utils.set_progress_bar_global_hook(hook)

def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

def start_comfyui(asyncio_loop=None):
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)

    prompt_server = server.PromptServer(asyncio_loop)
    q = execution.PromptQueue(prompt_server)

    hook_breaker_ac10a0.save_functions()
    nodes.init_extra_nodes(init_custom_nodes=not args.disable_all_custom_nodes)
    hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()
    prompt_server.add_routes()
    hijack_progress(prompt_server)

    threading.Thread(target=prompt_worker, daemon=True, args=(q, prompt_server)).start()

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = f"[{address}]"
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    return asyncio_loop, prompt_server, start_all

if __name__ == "__main__":
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))
    event_loop, _, start_all_func = start_comfyui()
    try:
        x = start_all_func()
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        logging.info("\nStopped server")
    cleanup_temp()
