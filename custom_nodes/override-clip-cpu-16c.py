# Force model to always use specified device
# Place in `ComfyUI\custom_nodes` to use
#  City96 [Apache2]
#
import types
import torch
import comfy.model_management
import os
import time

# Maximize CPU usage
torch.set_num_threads(32)

class OverrideDevice:
    @classmethod
    def INPUT_TYPES(s):
        devices = ["cpu",]
        for k in range(0, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            "required": {
                "device": (devices, {"default": "cpu"}),
            }
        }

    FUNCTION = "patch"
    CATEGORY = "other"

    def override(self, model, model_attr, device):
        model.device = device
        patcher = getattr(model, "patcher", model)
        for name in ["device", "load_device", "offload_device", "current_device", "output_device"]:
            setattr(patcher, name, device)

        py_model = getattr(model, model_attr)
        py_model.to = types.MethodType(torch.nn.Module.to, py_model)

        start = time.time()
        with torch.inference_mode():
            py_model.to(device)
        print(f"[override-clip-cpu] Moved model to {device} in {time.time() - start:.2f}s")

        # Disable further transfers
        def to(*args, **kwargs):
            pass
        py_model.to = types.MethodType(to, py_model)
        return (model,)

    def patch(self, *args, **kwargs):
        raise NotImplementedError

class OverrideCLIPDevice(OverrideDevice):
    @classmethod
    def INPUT_TYPES(s):
        k = super().INPUT_TYPES()
        k["required"]["clip"] = ("CLIP",)
        return k

    RETURN_TYPES = ("CLIP",)
    TITLE = "Force/Set CLIP Device"

    def patch(self, clip, device):
        return self.override(clip, "cond_stage_model", torch.device(device))

class OverrideVAEDevice(OverrideDevice):
    @classmethod
    def INPUT_TYPES(s):
        k = super().INPUT_TYPES()
        k["required"]["vae"] = ("VAE",)
        return k

    RETURN_TYPES = ("VAE",)
    TITLE = "Force/Set VAE Device"

    def patch(self, vae, device):
        return self.override(vae, "first_stage_model", torch.device(device))


NODE_CLASS_MAPPINGS = {
    "OverrideCLIPDevice": OverrideCLIPDevice,
    "OverrideVAEDevice": OverrideVAEDevice,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
