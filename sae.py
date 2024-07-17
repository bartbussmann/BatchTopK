import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg["dtype"]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], cfg["dict_size"], dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["dict_size"], cfg["act_size"], dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(cfg["dict_size"], dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.to(cfg["device"])
        self.cfg = cfg
    
    def forward(self, x, fraction=1):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        current_l1_coefficient = self.get_l1_coeff(fraction)
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = current_l1_coefficient * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        output = {"sae_out": x_reconstruct, 
                  "feature_acts": acts, 
                  "loss": loss, 
                  "l2_loss": l2_loss, 
                  "l1_loss": l1_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm}
        return output
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def get_l1_coeff(self, fraction):
        if fraction < self.cfg["l1_warmup_frac"]:
            return self.cfg["l1_coeff"] * (fraction / self.cfg["l1_warmup_frac"])
        else:
            return self.cfg["l1_coeff"]
    


class TopKSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg["dtype"]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], cfg["dict_size"], dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["dict_size"], cfg["act_size"], dtype=dtype)))
        self.W_dec.data[:] = self.W_enc.t().data
        self.b_enc = nn.Parameter(torch.zeros(cfg["dict_size"], dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros(cfg["dict_size"], dtype=dtype).to(cfg["device"])

        self.to(cfg["device"])
        self.cfg = cfg
    
    def forward(self, x, fraction=1):

        if self.cfg["input_unit_norm"]:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)

        x_cent = x - self.b_dec
        acts_pre = x_cent @ self.W_enc
        if self.cfg["use_encoder_bias"]:
            acts_pre += self.b_enc

        if self.cfg["activation"] == "relu":
            acts = F.relu(acts_pre)
        elif self.cfg["activation"] == "abs":
            acts = acts_pre.abs()
        elif self.cfg["activation"] == "quadratic":
            acts = acts_pre.pow(2)
        elif self.cfg["activation"] == "softplus":
            acts = F.softplus(acts_pre)
        elif self.cfg["activation"] == "gelu":
            acts = F.gelu(acts_pre)

        W_dec = self.W_enc.t() if self.cfg["tied_weights"] else self.W_dec
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)
        x_reconstruct = acts_topk @ W_dec + self.b_dec


        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(acts[:, dead_features], min(self.cfg["top_k_aux"], dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)
            x_reconstruct_aux = acts_aux @ W_dec[dead_features] + self.b_dec
            l2_loss_aux = self.cfg["aux_penalty"] * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_aux = torch.tensor(0, dtype=x.dtype, device=x.device)


        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_softmax_aux = acts_pre * F.softmax(acts_pre, dim=-1) * 64
            x_reconstruct_softmax = acts_softmax_aux[:, dead_features] @ W_dec[dead_features] + self.b_dec
            l2_loss_softmax = self.cfg["softmax_aux_penalty"] * (x_reconstruct_softmax.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_softmax = torch.tensor(0, dtype=x.dtype, device=x.device)

        

        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        current_l1_coefficient = self.get_l1_coeff(fraction)
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = current_l1_coefficient * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        self.num_batches_not_active += (acts_topk.sum(0) == 0).float()
        self.num_batches_not_active[acts_topk.sum(0) > 0] = 0
        loss = l2_loss + l1_loss + l2_loss_aux + l2_loss_softmax

        if self.cfg["input_unit_norm"]:
            x_reconstruct = x_reconstruct * x_std + x

        output = {"sae_out": x_reconstruct, 
                  "feature_acts": acts_topk, 
                  "loss": loss, 
                  "l2_loss": l2_loss, 
                  "l1_loss": l1_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm,
                  "l2_loss_aux": l2_loss_aux,
                  "l2_loss_softmax": l2_loss_softmax,
                  "num_dead_features": dead_features.sum()}
        return output

    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def get_l1_coeff(self, fraction):
        if fraction < self.cfg["l1_warmup_frac"]:
            return self.cfg["l1_coeff"] * (fraction / self.cfg["l1_warmup_frac"])
        else:
            return self.cfg["l1_coeff"]


class BatchTopKSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg["dtype"]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], cfg["dict_size"], dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["dict_size"], cfg["act_size"], dtype=dtype)))
        self.W_dec.data[:] = self.W_enc.t().data
        self.b_enc = nn.Parameter(torch.zeros(cfg["dict_size"], dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros(cfg["dict_size"], dtype=dtype).to(cfg["device"])

        self.to(cfg["device"])
        self.cfg = cfg
    
    def forward(self, x, fraction=1):

        if self.cfg["input_unit_norm"]:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)

        x_cent = x - self.b_dec
        acts_pre = x_cent @ self.W_enc
        if self.cfg["use_encoder_bias"]:
            acts_pre += self.b_enc

        if self.cfg["activation"] == "relu":
            acts = F.relu(acts_pre)
        elif self.cfg["activation"] == "abs":
            acts = acts_pre.abs()
        elif self.cfg["activation"] == "quadratic":
            acts = acts_pre.pow(2)
        elif self.cfg["activation"] == "softplus":
            acts = F.softplus(acts_pre)
        elif self.cfg["activation"] == "gelu":
            acts = F.gelu(acts_pre)

        W_dec = self.W_enc.t() if self.cfg["tied_weights"] else self.W_dec
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"]*x.shape[0], dim=-1)
        acts_topk = torch.zeros_like(acts.flatten()).scatter(-1, acts_topk.indices, acts_topk.values).reshape(acts.shape)
        x_reconstruct = acts_topk @ W_dec + self.b_dec


        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(acts[:, dead_features], min(self.cfg["top_k_aux"], dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)
            x_reconstruct_aux = acts_aux @ W_dec[dead_features] + self.b_dec
            l2_loss_aux = self.cfg["aux_penalty"] * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_aux = torch.tensor(0, dtype=x.dtype, device=x.device)


        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_softmax_aux = acts_pre * F.softmax(acts_pre, dim=-1) * 64
            x_reconstruct_softmax = acts_softmax_aux[:, dead_features] @ W_dec[dead_features] + self.b_dec
            l2_loss_softmax = self.cfg["softmax_aux_penalty"] * (x_reconstruct_softmax.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_softmax = torch.tensor(0, dtype=x.dtype, device=x.device)

        

        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        current_l1_coefficient = self.get_l1_coeff(fraction)
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = current_l1_coefficient * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        self.num_batches_not_active += (acts_topk.sum(0) == 0).float()
        self.num_batches_not_active[acts_topk.sum(0) > 0] = 0
        loss = l2_loss + l1_loss + l2_loss_aux + l2_loss_softmax

        if self.cfg["input_unit_norm"]:
            x_reconstruct = x_reconstruct * x_std + x

        output = {"sae_out": x_reconstruct, 
                  "feature_acts": acts_topk, 
                  "loss": loss, 
                  "l2_loss": l2_loss, 
                  "l1_loss": l1_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm,
                  "l2_loss_aux": l2_loss_aux,
                  "l2_loss_softmax": l2_loss_softmax,
                  "num_dead_features": dead_features.sum()}
        return output

    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def get_l1_coeff(self, fraction):
        if fraction < self.cfg["l1_warmup_frac"]:
            return self.cfg["l1_coeff"] * (fraction / self.cfg["l1_warmup_frac"])
        else:
            return self.cfg["l1_coeff"]




class QuantileSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg["dtype"]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], cfg["dict_size"], dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["dict_size"], cfg["act_size"], dtype=dtype)))
        self.W_dec.data[:] = self.W_enc.t().data

        self.b_enc = nn.Parameter(torch.zeros(cfg["dict_size"], dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros(cfg["dict_size"], dtype=dtype).to(cfg["device"])
        self.quantile_threshold = torch.ones((1,), dtype=dtype).to(cfg["device"])
        # self.quantile_threshold = torch.ones((cfg["dict_size"],), dtype=dtype).to(cfg["device"])

        self.to(cfg["device"])
        self.cfg = cfg
    
    def forward(self, x, fraction=1):


        if self.cfg["input_unit_norm"]:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)


        x_cent = x - self.b_dec
        acts_pre = x_cent @ self.W_enc
        if self.cfg["use_encoder_bias"]:
            acts_pre += self.b_enc
        if self.cfg["activation"] == "relu":
            acts = F.relu(acts_pre)
        elif self.cfg["activation"] == "abs":
            acts = acts_pre.abs()
        elif self.cfg["activation"] == "quadratic":
            acts = acts_pre.pow(2)
        elif self.cfg["activation"] == "softplus":
            acts = F.softplus(acts_pre)
        elif self.cfg["activation"] == "gelu":
            acts = F.gelu(acts_pre)

        W_dec = self.W_enc.t() if self.cfg["tied_weights"] else self.W_dec
        acts_mask = acts > self.quantile_threshold
        acts_thresholded = acts * acts_mask
        x_reconstruct = acts_thresholded @ W_dec + self.b_dec


        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(acts[:, dead_features], min(self.cfg["top_k_aux"], dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)
            x_reconstruct_aux = acts_aux @ W_dec[dead_features] + self.b_dec
            l2_loss_aux = self.cfg["aux_penalty"] * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_aux = torch.tensor(0, dtype=x.dtype, device=x.device)


        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_softmax_aux = acts_pre * F.softmax(acts_pre, dim=-1) * 64
            x_reconstruct_softmax = acts_softmax_aux[:, dead_features] @ W_dec[dead_features] + self.b_dec
            l2_loss_softmax = self.cfg["softmax_aux_penalty"] * (x_reconstruct_softmax.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_softmax = torch.tensor(0, dtype=x.dtype, device=x.device)

        

        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        current_l1_coefficient = self.get_l1_coeff(fraction)
        l1_norm = acts_thresholded.float().abs().sum(-1).mean()
        l1_loss = current_l1_coefficient * l1_norm
        l0_norm = (acts_thresholded > 0).float().sum(-1).mean()
        self.num_batches_not_active += (acts_thresholded.sum(0) == 0).float()
        self.num_batches_not_active[acts_thresholded.sum(0) > 0] = 0
        loss = l2_loss + l1_loss + l2_loss_aux + l2_loss_softmax

        self.update_quantile_thresholds(acts_thresholded)

        if self.cfg["input_unit_norm"]:
            x_reconstruct = x_reconstruct * x_std + x


        output = {"sae_out": x_reconstruct, 
                  "feature_acts": acts_thresholded, 
                  "loss": loss, 
                  "l2_loss": l2_loss, 
                  "l1_loss": l1_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm,
                  "l2_loss_aux": l2_loss_aux,
                "l2_loss_softmax": l2_loss_softmax,
                  "num_dead_features": dead_features.sum()}
        return output

    @torch.no_grad()
    def update_quantile_thresholds(self, acts_thresholded):
        smaller_than_threshold = (acts_thresholded == 0).float().sum()
        larger_than_threshold = (acts_thresholded > 0).float().sum()
        error_smaller = smaller_than_threshold * (self.cfg["quantile"] - 1)
        error_larger = larger_than_threshold * self.cfg["quantile"]
        total_error = error_smaller + error_larger
        self.quantile_threshold += total_error * self.cfg["quantile_lr"]


    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def get_l1_coeff(self, fraction):
        if fraction < self.cfg["l1_warmup_frac"]:
            return self.cfg["l1_coeff"] * (fraction / self.cfg["l1_warmup_frac"])
        else:
            return self.cfg["l1_coeff"]





class MaxPoolSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg["dtype"]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["act_size"], cfg["dict_size"], dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg["dict_size"], cfg["act_size"], dtype=dtype)))
        self.W_dec.data[:] = self.W_enc.t().data

        self.b_enc = nn.Parameter(torch.zeros(cfg["dict_size"], dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros(cfg["dict_size"], dtype=dtype).to(cfg["device"])
        self.to(cfg["device"])
        self.cfg = cfg
    
    def forward(self, x, fraction=1):

        if self.cfg["input_unit_norm"]:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)

        x_cent = x - self.b_dec
        acts_pre = x_cent @ self.W_enc
        if self.cfg["use_encoder_bias"]:
            acts_pre += self.b_enc
        if self.cfg["activation"] == "relu":
            acts = F.relu(acts_pre)
        elif self.cfg["activation"] == "abs":
            acts = acts_pre.abs()
        elif self.cfg["activation"] == "quadratic":
            acts = acts_pre.pow(2)
        elif self.cfg["activation"] == "softplus":
            acts = F.softplus(acts_pre)
        elif self.cfg["activation"] == "gelu":
            acts = F.gelu(acts_pre)
            
        acts_reshaped = acts.view(acts.shape[0], -1, self.cfg["pool_size"])
        max_vals, _ = torch.max(acts_reshaped, dim=-1, keepdim=True)
        mask = (acts_reshaped == max_vals)
        acts_pooled = (acts_reshaped * mask).view(acts.shape)

        acts_pooled_topk = torch.topk(acts_pooled, self.cfg["top_k"], dim=-1)
        acts_pooled_topk = torch.zeros_like(acts).scatter(-1, acts_pooled_topk.indices, acts_pooled_topk.values)
        acts_topk_aux = torch.topk(acts, self.cfg["top_k_aux"], dim=-1)
        acts_aux = torch.zeros_like(acts).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)

        x_reconstruct = acts_pooled_topk @ self.W_dec + self.b_dec
        x_reconstruct_aux = acts_aux @ self.W_dec + self.b_dec

        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(acts[:, dead_features], min(self.cfg["top_k_aux"], dead_features.sum()), dim=-1)
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(-1, acts_topk_aux.indices, acts_topk_aux.values)
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features] + self.b_dec
            l2_loss_aux = self.cfg["aux_penalty"] * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_aux = torch.tensor(0, dtype=x.dtype, device=x.device)

        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_softmax_aux = acts * F.softmax(acts, dim=-1)
            acts_softmax_aux = acts_softmax_aux
            x_reconstruct_softmax = acts_softmax_aux @ self.W_dec + self.b_dec
            l2_loss_softmax = self.cfg["softmax_aux_penalty"] * (x_reconstruct_softmax.float() - residual.float()).pow(2).mean()
        else:
            l2_loss_softmax = torch.tensor(0, dtype=x.dtype, device=x.device)


        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        current_l1_coefficient = self.get_l1_coeff(fraction)
        l1_norm = acts_pooled_topk.float().abs().sum(-1).mean()
        l1_loss = current_l1_coefficient * l1_norm
        l0_norm = (acts_pooled_topk > 0).float().sum(-1).mean()
        self.num_batches_not_active += (acts_pooled_topk.sum(0) == 0).float()
        self.num_batches_not_active[acts_pooled_topk.sum(0) > 0] = 0
        loss = l2_loss + l1_loss + l2_loss_aux + l2_loss_softmax

        if self.cfg["input_unit_norm"]:
            x_reconstruct = x_reconstruct * x_std + x


        output = {"sae_out": x_reconstruct, 
                  "feature_acts": acts_pooled_topk, 
                  "loss": loss, 
                  "l2_loss": l2_loss, 
                  "l1_loss": l1_loss, 
                  "l0_norm": l0_norm, 
                  "l1_norm": l1_norm,
                  "l2_loss_aux": l2_loss_aux,
                  "l2_loss_softmax": l2_loss_softmax,
                  "num_dead_features": dead_features.sum()}
        return output
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def get_l1_coeff(self, fraction):
        if fraction < self.cfg["l1_warmup_frac"]:
            return self.cfg["l1_coeff"] * (fraction / self.cfg["l1_warmup_frac"])
        else:
            return self.cfg["l1_coeff"]


