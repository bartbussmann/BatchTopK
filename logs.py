import wandb
import torch
from functools import partial
import os
import json

def init_wandb(cfg):
    run = wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)
    return run

def log_wandb(output, step, wandb_run, index=None):
    to_log = ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "l2_loss_aux", "l2_loss_softmax", "num_dead_features"]
    log_dict = {k: output[k].item() for k in to_log if k in output}
    n_dead_in_batch = (output["feature_acts"].sum(0) == 0).sum().item()
    log_dict["n_dead_in_batch"] = n_dead_in_batch
    if index is not None:
        for k, v in log_dict.items():
            wandb_run.log({f"{k}_{index}": v}, step=step)
    else:
        wandb_run.log(log_dict, step=step)
    return None


def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    activation[:] = activation.mean([0, 1])
    return activation

def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None):
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()
    batch = activations_store.get_activations(batch_tokens).reshape(-1, sae.cfg["act_size"])
    with torch.no_grad():
        sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

        original_loss = model(batch_tokens, return_type="loss").item()
        reconstr_loss = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(sae.cfg["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
            return_type="loss",
        ).item()

        zero_loss = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(sae.cfg["hook_point"], zero_abl_hook)],
            return_type="loss",
        ).item()

        mean_loss = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(sae.cfg["hook_point"], mean_abl_hook)],
            return_type="loss",
        ).item()


        ce_degradation = original_loss - reconstr_loss
        zero_degradation = original_loss - zero_loss
        mean_degradation = original_loss - mean_loss

        recovery_from_zero = (reconstr_loss - zero_loss) / zero_degradation
        recovery_from_mean = (reconstr_loss - mean_loss) / mean_degradation


        log_dict = {
            "performance/ce_degradation": ce_degradation,
            "performance/recovery_from_zero": recovery_from_zero,
            "performance/recovery_from_mean": recovery_from_mean,
        }

        if index is not None:
            for k, v in log_dict.items():
                wandb_run.log({f"{k}_{index}": v}, step=step)
        else:
            wandb_run.log(log_dict, step=step)



def save_checkpoint(wandb_run, sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, torch.dtype):
            json_safe_cfg[key] = str(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        else:
            json_safe_cfg[key] = str(value)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    artifact = wandb.Artifact(
        name=f"{cfg['name']}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )

    artifact.add_file(sae_path)
    artifact.add_file(config_path)

    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")