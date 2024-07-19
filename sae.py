import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.cfg["act_size"], self.cfg["dict_size"])))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.cfg["dict_size"], self.cfg["act_size"])))
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.cfg["dict_size"],)).to(cfg["device"])
        
        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"]*x.shape[0], dim=-1)
        acts_topk = torch.zeros_like(acts.flatten()).scatter(-1, acts_topk.indices, acts_topk.values).reshape(acts.shape)
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {"sae_out": sae_out, 
                  "feature_acts": acts,
                  "num_dead_features": num_dead_features,
                  "loss": loss, 
                  "l1_loss": l1_loss, 
                  "l2_loss": l2_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm,
                  "aux_loss": aux_loss}
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(acts[:, dead_features], min(self.cfg["top_k_aux"], dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features] + self.b_dec
            l2_loss_aux = self.cfg["aux_penalty"] * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {"sae_out": sae_out, 
                  "feature_acts": acts,
                  "num_dead_features": num_dead_features,
                  "loss": loss, 
                  "l1_loss": l1_loss, 
                  "l2_loss": l2_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm,
                  "aux_loss": aux_loss}
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(acts[:, dead_features], min(self.cfg["top_k_aux"], dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features] + self.b_dec
            l2_loss_aux = self.cfg["aux_penalty"] * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)


class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (self.num_batches_not_active > self.cfg["n_batches_to_dead"]).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {"sae_out": sae_out, 
                  "feature_acts": acts,
                  "num_dead_features": num_dead_features,
                  "loss": loss, 
                  "l1_loss": l1_loss, 
                  "l2_loss": l2_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm}
        return output



