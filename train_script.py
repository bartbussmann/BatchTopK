#%%
from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, MaxPoolSAE, QuantileSAE, BatchTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
import copy

cfg = get_default_cfg()
cfg["l1_coeff"] = 0 
cfg["aux_penalty"] = (1/32)
cfg["lr"] = 3e-4
cfg["use_encoder_bias"] = False
cfg["activation"] = "relu"
cfg["tied_weights"] = False
cfg["input_unit_norm"] = True
cfg["perf_log_freq"] = 1000
cfg["project"] = "batch_topk_sweep"
cfg["top_k"] = 16
cfg["sae_type"] = "topk"
cfg["dict_size"] = 768 * 16
cfg = post_init_cfg(cfg)

if cfg["sae_type"] == "topk":
    sae = TopKSAE(cfg)
elif cfg["sae_type"] == "batchtopk":
    sae = BatchTopKSAE(cfg)
        
model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
activations_store = ActivationsStore(model, cfg)
train_sae(sae, activations_store, model, cfg)
# %%
