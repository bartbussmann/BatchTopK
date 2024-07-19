#%%
from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer

cfg = get_default_cfg()
cfg["sae_type"] = "batchtopk" # "vanilla", "topk", "batchtopk"
cfg["model_name"] = "gpt2-small"
cfg["layer"] = 8
cfg["site"] = "resid_pre"
cfg["dataset_path"] = "Skylion007/openwebtext"
cfg["aux_penalty"] = (1/32)
cfg["lr"] = 3e-4
cfg["input_unit_norm"] = True
cfg["top_k"] = 32
cfg["dict_size"] = 768 * 16
cfg = post_init_cfg(cfg)

if cfg["sae_type"] == "vanilla":
    sae = VanillaSAE(cfg)
elif cfg["sae_type"] == "topk":
    sae = TopKSAE(cfg)
elif cfg["sae_type"] == "batchtopk":
    sae = BatchTopKSAE(cfg)
            
model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
activations_store = ActivationsStore(model, cfg)
train_sae(sae, activations_store, model, cfg)
# %%
