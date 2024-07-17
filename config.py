import transformer_lens.utils as utils
import torch 

def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 1e-4,
        "num_tokens": int(5e8),
        "l1_coeff": 1e-2,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "dtype": torch.float32,
        "model_name": "gpt2-small",
        "site": "resid_pre",
        "layer": 8,
        "act_size": 768,
        "dict_size": 4096,
        "device": "cuda:0",
        "model_batch_size": 512,
        "num_batches_in_buffer": 10,
        "dataset_path": "Skylion007/openwebtext",
        "l1_warmup_frac": 0.01,
        "wandb_project": "sparse_autoencoders",
        "activation": "relu",
        "tied_weights": False,
        "input_unit_norm": False,
        "perf_log_freq": 100,
        "sae_type": "topk",
        "checkpoint_freq": 10000,

        # TopKSAE specific
        "top_k": 50,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        "n_batches_to_dead": 5,

        # MaxPoolSAE specific
        "pool_size": 4,
        "softmax_aux_penalty": 0,

        # Quantile specific
        "quantile": 0.99,
        "quantile_lr": 1e-3,
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["name"] = f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg