#%%
from training import train_sae_group_separate_wandb
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE, GlobalTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
import torch
import copy

def get_sae_type(cfg):  
    if cfg["sae_type"] == "vanilla":
        sae = VanillaSAE(cfg)
    elif cfg["sae_type"] == "topk":
        sae = TopKSAE(cfg)
    elif cfg["sae_type"] == "batchtopk":
        sae = BatchTopKSAE(cfg)
    elif cfg["sae_type"] == 'jumprelu':
        sae = JumpReLUSAE(cfg)
    elif cfg["sae_type"] == 'globaltopk':
        sae = GlobalTopKSAE(cfg)
    return sae

saes = []
cfgs = []
for sae_type in ['topk']:
    for top_k in [32, 128]:
        cfg = get_default_cfg()
        cfg["model_name"] = "gpt2-small"
        cfg["layer"] = 8
        cfg["site"] = "resid_post"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg["aux_penalty"] = (1/32)
        cfg["lr"] = 3e-4
        cfg["model_dtype"] = torch.bfloat16
        cfg["input_unit_norm"] = True
        cfg["dict_size"] = 2**14
        cfg['wandb_project'] = 'global_topk_sweep10'
        cfg['l1_coeff'] = 0.
        cfg['act_size'] = 768
        cfg['device'] = 'cuda'
        cfg["sae_type"] = sae_type
        cfg['top_k'] = top_k
        cfg = post_init_cfg(cfg)
        sae = get_sae_type(cfg) 
        saes.append(copy.deepcopy(sae))
        cfgs.append(copy.deepcopy(cfg))

    for dict_size in [2**13, 2**14, 2**15]:
        cfg = get_default_cfg()
        cfg["model_name"] = "gpt2-small"
        cfg["layer"] = 8
        cfg["site"] = "resid_post"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg["aux_penalty"] = (1/32)
        cfg["lr"] = 3e-4
        cfg["model_dtype"] = torch.bfloat16
        cfg["input_unit_norm"] = True
        cfg['wandb_project'] = 'global_topk_sweep10'
        cfg['l1_coeff'] = 0.
        cfg['act_size'] = 768
        cfg['device'] = 'cuda'
        cfg["sae_type"] = sae_type
        cfg['top_k'] = 64
        cfg['dict_size'] = dict_size
        cfg = post_init_cfg(cfg)
        sae = get_sae_type(cfg) 
        saes.append(copy.deepcopy(sae))
        cfgs.append(copy.deepcopy(cfg))


model = HookedTransformer.from_pretrained_no_processing(cfg["model_name"]).to(cfg["model_dtype"]).to(cfg["device"])
activations_store = ActivationsStore(model, cfg)
train_sae_group_separate_wandb(saes, activations_store, model, cfgs)
