import os
import torch
import importlib
import numpy as np

from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from einops import rearrange
from contextlib import nullcontext

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def main():
    prompt = "a professional photograph of an astronaut riding a horse"

    # device = torch.device("cuda")
    device = torch.device("cpu")
    config = OmegaConf.load("./moskize_sd/config/v2-inference-v-mac.yaml")
    ckpt = torch.load("./moskize_sd/download/checkpoint/v2-1_768-nonema-pruned.ckpt", map_location="cpu")
    model = load_model_from_config(config, ckpt, device)

    outputs_path = "./outputs"
    os.makedirs(outputs_path, exist_ok=True)

    sample_path = os.path.join(outputs_path, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # sampler = PLMSSampler(model, device=device)
    # sampler = DPMSolverSampler(model, device=device)
    sampler = DDIMSampler(model, device=device)

    # precision_scope = autocast
    precision_scope = nullcontext

    batch_size = 1
    n_iter = 1

    steps = 25
    start_code = None
    ddim_eta = 0.0
    scale = 9.0

    C = 4
    H = 768
    W = 768
    f = 8

    with torch.no_grad(), precision_scope(device), model.ema_scope():
        all_samples = list()
        base_count = len(os.listdir(sample_path))
        sample_count = 1

        for n in range(n_iter):
            c = model.get_learned_conditioning(batch_size * [prompt])
            uc = None

            if scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])

            shape = [C, H // f, W // f]
            samples, _ = sampler.sample(S=steps,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=ddim_eta,
                                        x_T=start_code)

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            for x_sample in x_samples:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1
                sample_count += 1

            all_samples.append(x_samples)

    print("complete")


def load_model_from_config(config, ckpt, device):
    sd = ckpt["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


if __name__ == "__main__":
    main()