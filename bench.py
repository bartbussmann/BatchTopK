#%%
import torch, matplotlib.pyplot as plt, numpy as np, matplotlib.colors as mcolors
torch.manual_seed(0)
device, dtype = "cuda", torch.float        # change dtype if you prefer BF16

k        = 100
seq_len  = 1

# --- sweep ranges -----------------------------------------------------------
batch_powers = range(10, 15)                 # B = 2^10 … 2^14
dims         = [2**i for i in range(12, 17)] # 1,024 … 32,768

# how many timing samples per (B, dim, mode)?
repeats = 20

# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def topk_mask(acts, k=64):
    acts_topk = torch.topk(acts, k, dim=-1)
    return torch.zeros_like(acts).scatter(-1, acts_topk.indices, acts_topk.values)

def batchtopk_mask(acts, k=64):
    flat = acts.flatten()
    acts_topk = torch.topk(flat, k * acts.shape[0], dim=-1)
    return torch.zeros_like(flat).scatter(-1, acts_topk.indices, acts_topk.values).reshape(acts.shape)

def threshold_uniform_k(acts, k=64):
    """Analytic threshold for Uniform(0,1) input → expected k survivors per row."""
    dim = acts.shape[-1]
    t = 1.0 - k / dim
    return acts * (acts > t).to(acts.dtype)

# --------------------------------------------------------------------------- #
# low-level timed run (single measurement)
# --------------------------------------------------------------------------- #
def bench_once(B, dim, mode="topk", k=64, t=0.99):
    x = torch.rand(B, seq_len, dim, device=device, dtype=dtype, requires_grad=False)

    start = torch.cuda.Event(enable_timing=True)
    stop  = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    if mode == "topk":
        v = topk_mask(x, k)
    elif mode == "batchtopk":
        v = batchtopk_mask(x, k)
    elif mode == "uniformk":      # analytic expected-k threshold
        v = threshold_uniform_k(x, k)
    elif mode == "fixedt":        # legacy: global > t
        v = threshold_fixed(x, t)
    else:
        raise ValueError(f"unknown mode {mode}")

    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / 1e3    # seconds


# --------------------------------------------------------------------------- #
# helper that repeats and aggregates
# --------------------------------------------------------------------------- #
def bench(B, dim, mode, k=64, t=0.99, repeats=5):
    times = [bench_once(B, dim, mode, k=k, t=t) for _ in range(repeats)]
    arr = np.array(times, dtype=np.float32)
    return {
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }


# warm-up (not recorded)
for _ in range(3):
    _ = bench_once(64, 4096, mode="topk", k=k)

# ---------- run grid ----------
modes = {
    "TopK"       : "topk",        # was "per-row topk"
    "BatchTopK"  : "batchtopk",   # was "flatten topk"
    "GlobalTopK" : "uniformk",    # was "uniform exp-k"
}

shape = (len(dims), len(batch_powers))
results_mean   = {name: np.zeros(shape, np.float32) for name in modes}
results_median = {name: np.zeros(shape, np.float32) for name in modes}
results_std    = {name: np.zeros(shape, np.float32) for name in modes}

for di, dim in enumerate(dims):
    for pi, p in enumerate(batch_powers):
        B = 1 << p
        for name, mode in modes.items():
            stats = bench(B, dim, mode, k=k, t=0.99, repeats=repeats)
            results_mean[name][di, pi]   = stats["mean"]
            results_median[name][di, pi] = stats["median"]
            results_std[name][di, pi]    = stats["std"]

# ---------- common colour scale (use medians for plotting) ----------
vmin = min(mat.min() for mat in results_median.values())
vmax = max(mat.max() for mat in results_median.values())

use_log = True
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if use_log else None

# ---------- plot medians ----------
fig, axs = plt.subplots(1, len(modes), figsize=(6.5*len(modes), 5), constrained_layout=True)
imshow_kwargs = dict(origin="lower", aspect="auto", interpolation="nearest")

for ax, (name, mat) in zip(np.atleast_1d(axs), results_median.items()):
    if norm is None:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, **imshow_kwargs)
    else:
        im = ax.imshow(mat, norm=norm, **imshow_kwargs)
    ax.set_xticks(range(len(batch_powers)), [f"2^{p}" for p in batch_powers])
    ax.set_yticks(range(len(dims)), [str(d) for d in dims])
    ax.set_xlabel("batch size (B)")
    ax.set_ylabel("embedding dim")
    ax.set_title(name)

fig.colorbar(im, ax=axs, label="sec median (fwd+bwd)", shrink=0.78)
plt.suptitle(f"Top-k variants – FP16, CUDA – {repeats}× median (shared colour scale)", y=1.04)
plt.show()

# ---------- optional variability plot (coef of variation) ----------
#   cv = std / mean  (dimensionless; shows stability)
fig, axs = plt.subplots(1, len(modes), figsize=(6.5*len(modes), 5), constrained_layout=True)
for ax, (name, _) in zip(np.atleast_1d(axs), modes.items()):
    mean_mat = results_mean[name]
    std_mat  = results_std[name]
    cv = np.where(mean_mat > 0, std_mat / mean_mat, 0.0)
    im = ax.imshow(cv, origin="lower", aspect="auto", interpolation="nearest", vmin=0.0, vmax=np.nanmax(cv))
    ax.set_xticks(range(len(batch_powers)), [f"2^{p}" for p in batch_powers])
    ax.set_yticks(range(len(dims)), [str(d) for d in dims])
    ax.set_xlabel("batch size (B)")
    ax.set_ylabel("embedding dim")
    ax.set_title(f"{name} CV")

fig.colorbar(im, ax=axs, label="std / mean", shrink=0.78)
plt.suptitle(f"Runtime variability over {repeats} runs", y=1.04)
plt.show()


# --------------------------------------------------------------------------- #
#  Line‑of‑sight plots for easier 1‑D comparison
# --------------------------------------------------------------------------- #
import itertools

# ▸ choose which slices you want to visualise
fixed_batch_power = 12        # B = 2^12 = 4096
fixed_dim_power   = 14        # dim = 2^14 = 16384

try:
    b_idx = batch_powers.index(fixed_batch_power)
    d_idx = [i for i,d in enumerate(dims) if d == (1<<fixed_dim_power)][0]
except ValueError:
    raise RuntimeError("chosen fixed_* values are outside sweep range")

color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# —————————————————————— 1) vary dim, fixed batch —————————————————————— #
plt.figure(figsize=(7,5))
for name, mat in results_median.items():
    plt.plot(dims, mat[:, b_idx], marker="o", label=name, color=next(color_cycle))
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("embedding dim")
plt.ylabel("sec  (fwd+bwd median)")
plt.title(f"Fixed batch size  B = 2^{fixed_batch_power}")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()

# —————————————————————— 2) vary batch, fixed dim —————————————————————— #
color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

plt.figure(figsize=(7,5))
for name, mat in results_median.items():
    plt.plot([1<<p for p in batch_powers],
             mat[d_idx, :], marker="o", label=name, color=next(color_cycle))
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("batch size  B")
plt.ylabel("sec  (fwd+bwd median)")
plt.title(f"Fixed embedding dim = 2^{fixed_dim_power}")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()

# %%
