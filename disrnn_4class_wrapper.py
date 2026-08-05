# full_script_4class_disrnn_final.py
"""
Final cleaned training + analysis script:
- Uses repo DisRNN core
- latent_size = 6
- KL scaling beta = 0.01 (NeurIPS-style): loss = CE + beta * mean_per_trial(sum_timestep(penalty_t))
- Uses generative means from MAT column 23 (index 22) to center stimuli
- Kernels computed WITHOUT multiplying by 2 (user request)
- Network inputs ARE scaled by 2 when passed into the RNN (keeps repo conventions)
- Plots: kernels, bottleneck sigmas, latent trajectories by selected/non-selected × conf
"""
import time
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Any

from disentangled_rnns.library import disrnn, plotting  # requires repo on PYTHONPATH

# sklearn for kernels
from sklearn.linear_model import LogisticRegression

# ------------------------
# === USER: path to your .mat
# ------------------------
MAT_PATH = "./sbj10.mat"   # <-- adjust

# ------------------------
# === HYPERPARAMS
# ------------------------
latent_size = 5          # as requested
n_steps = 3000           # you used 3000 previously
lr = 1e-3
beta = 0.0005              # NeurIPS-style scaling
seed = 0
min_sigma = 1e-5         # reparam used in repo: abs(raw) + min_sigma

# ------------------------
# === LOAD DATA
# ------------------------
mat = scipy.io.loadmat(MAT_PATH)
if 'data' in mat:
    Xmat = mat['data']
else:
    raise RuntimeError("MAT file doesn't contain key 'data' — check structure.")

# Columns recap (as you reported earlier):
# choice in column 0, right stimuli in 1:11, left in 11:21, conf in 21, delta/generative mean in column 23 (index 22)
choice_all = Xmat[:, 0]
stim_left_all = Xmat[:, 1:11]
stim_right_all  = Xmat[:, 11:21]
conf_all = Xmat[:, 21]

# generative delta stored in column 23 (1-based in your note) -> index 22
if Xmat.shape[1] > 22:
    delta_gen_all = Xmat[:, 22]
else:
    # if not present, fallback to zeros and warn
    delta_gen_all = np.zeros(choice_all.shape)
    print("Warning: delta/generative column (index 22) not present — using zeros.")

print("Raw unique choices:", np.unique(choice_all))
print("Raw unique conf:", np.unique(conf_all))

# filter trials that have NaN in either choice or conf
nan_mask = (~np.isnan(choice_all)) & (~np.isnan(conf_all))
print("Valid trials (non-NaN in choice & conf):", np.sum(nan_mask))

choice = choice_all[nan_mask].astype(int)
conf = conf_all[nan_mask].astype(int)
stim_right = stim_right_all[nan_mask]
stim_left  = stim_left_all[nan_mask]
delta_gen = delta_gen_all[nan_mask]  # Δ = mu_R - mu_L

N_trials = stim_right.shape[0]
timesteps = 10
features_per_side = 1

# reshape stimuli -> (trials, timesteps, 1)
stim_right = stim_right.reshape(N_trials, timesteps, features_per_side)
stim_left  = stim_left.reshape(N_trials, timesteps, features_per_side)

# Compute generative means per trial:
# mu_R = 0.5 + delta/2 ; mu_L = 0.5 - delta/2
mu_R = 0.5 + (delta_gen.astype(float) / 2.0)
mu_L = 0.5 - (delta_gen.astype(float) / 2.0)
# Expand to (trials, timesteps, 1) to allow subtracting
mu_R_tiled = mu_R[:, None, None] * np.ones((1, timesteps, 1))
mu_L_tiled = mu_L[:, None, None] * np.ones((1, timesteps, 1))

# Centered stimuli for kernel computation (NO multiply by 2 here per your request)
right_centered_for_kernel = stim_right - mu_R_tiled  # (trials, timesteps, 1)
left_centered_for_kernel = stim_left - mu_L_tiled

# For network input, follow repo convention: center by generative means THEN multiply by 2
right_for_net = 2.0 * (stim_right - mu_R_tiled)
left_for_net = 2.0 * (stim_left - mu_L_tiled)

# concat -> (trials, timesteps, 2) with (right, left) channels
x_concat = np.concatenate([right_for_net, left_for_net], axis=2)  # used for RNN
# transpose -> (timesteps, trials, features)
x_reshaped = np.transpose(x_concat, (1, 0, 2)).astype(float)  # (T, N, 2)
print("x_reshaped shape (T, N, features):", x_reshaped.shape)

# 4-class combined labels: class = choice*2 + conf
combined_label = (choice * 2 + conf).astype(int)  # shape (trials,)
print("Human class counts:", np.unique(combined_label, return_counts=True))

y_labels = jnp.array(combined_label)  # (N,)

# ------------------------
# === DisRNN config (latent penalties set to 1.0 so core returns sum-of-KLs weighted by internal multipliers)
# We will scale that core penalty by beta in the loss.
# ------------------------
disrnn_config = disrnn.DisRnnConfig(
    obs_size=x_reshaped.shape[2],
    output_size=4,                 # 4-way output
    x_names=['right', 'left'],
    y_names=['choice_conf'],
    latent_size=latent_size,
    update_net_n_units_per_layer=16,
    update_net_n_layers=4,
    choice_net_n_units_per_layer=4,
    choice_net_n_layers=2,
    activation='leaky_relu',
    noiseless_mode=False,
    # set internal multipliers to 1 so core's penalty equals the sum of all bottleneck KL terms
    latent_penalty=1.0,
    choice_net_latent_penalty=1.0,
    update_net_obs_penalty=1.0,
    update_net_latent_penalty=1.0,
)

# ------------------------
# === Haiku transform: unroll the core and attach a final 4-class Linear head
# The core returns a per-timestep penalty in its output. We'll sum that across timesteps per trial.
# ------------------------
def unroll_and_head(xs):
    core = disrnn.HkDisentangledRNN(disrnn_config)
    head = hk.Linear(4, name="final_4class_head")
    batch_size = xs.shape[1]
    state = core.initial_state(batch_size)  # (batch, latent)
    logits_list = []
    latents_list = []
    penalty_list = []
    for t in range(xs.shape[0]):
        core_out, state = core(xs[t], state)
        # core_out: (batch, output_size + 1) where last element is penalty per trial at that timestep
        logits_from_core = core_out[:, :-1]  # not used; we use head applied to latent
        pen_t = core_out[:, -1]              # shape (batch,)
        latents_list.append(state)
        logits_list.append(head(state))
        penalty_list.append(pen_t)
    outputs = jnp.stack(logits_list, axis=0)      # (T, N, 4)
    latents = jnp.stack(latents_list, axis=0)     # (T, N, latent)
    penalties = jnp.stack(penalty_list, axis=0)   # (T, N)
    # total penalty per trial = sum over timesteps, shape (N,)
    total_penalty_per_trial = jnp.sum(penalties, axis=0)
    return outputs, latents, total_penalty_per_trial

unroll_t = hk.transform(unroll_and_head)

# ------------------------
# === Initialize params & optimizer
# ------------------------
rng = jax.random.PRNGKey(seed)
rng, init_key = jax.random.split(rng)
sample_x = jnp.array(x_reshaped)  # (T,N,features)
params = unroll_t.init(init_key, sample_x)

# quick sanity print of param keys (optional)
def print_param_structure(p, indent=0):
    if not isinstance(p, dict):
        return
    for k in sorted(p.keys()):
        print(" " * indent + str(k))
        if isinstance(p[k], dict):
            print_param_structure(p[k], indent+2)

print("Top-level param keys:")
print_param_structure(params)

# Print sigma leaves shapes if available
hk_core = params.get('hk_disentangled_rnn', params)
print("\nFound sigma parameter leaves (if present):")
for key in ('latent_sigma_params',
            'update_net_obs_sigma_params',
            'update_net_latent_sigma_params',
            'choice_net_sigma_params'):
    if isinstance(hk_core, dict) and (key in hk_core):
        arr = np.array(hk_core[key])
        print(f"  {key}: shape {arr.shape}")
    else:
        print(f"  {key}: NOT FOUND")

optimizer = optax.adam(lr)
opt_state = optimizer.init(params)

# ------------------------
# === Loss & update (use core's penalty and scale by beta)
# ------------------------
@jax.jit
def compute_loss_and_aux(params, rng, xs, y_labels):
    outputs, latents, total_pen = unroll_t.apply(params, rng, xs)  # total_pen: (N,)
    final_logits = outputs[-1]  # (N,4)

    # cross-entropy (mean across batch)
    log_probs = jax.nn.log_softmax(final_logits, axis=-1)
    one_hot = jax.nn.one_hot(y_labels, num_classes=4)
    ce = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    # KL penalty: mean across trials of summed per-timestep penalties (this matches repo output)
    kl_penalty = jnp.mean(total_pen)

    loss = ce + beta * kl_penalty

    preds = jnp.argmax(final_logits, axis=-1)
    return loss, (ce, kl_penalty, final_logits, preds, latents)

@jax.jit
def update(params, opt_state, rng, xs, y_labels):
    (loss, aux), grads = jax.value_and_grad(compute_loss_and_aux, has_aux=True)(params, rng, xs, y_labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# ------------------------
# === Training loop (full-batch)
# ------------------------
print("\nStarting training (full-batch). steps =", n_steps)
start = time.time()
for step in range(1, n_steps + 1):
    rng, step_key = jax.random.split(rng)
    params, opt_state, loss, aux = update(params, opt_state, step_key, jnp.array(x_reshaped), y_labels)
    if step % 100 == 0 or step == 1 or step == n_steps:
        ce, kl_penalty, final_logits, preds, latents = aux
        final_logits_np = np.array(final_logits)
        preds_np = np.array(preds)
        acc = np.mean(preds_np == np.array(combined_label))
        print(f"step {step:5d} | loss {loss:.6f} | ce {ce:.6f} | KL(mean trial sum) {kl_penalty:.6e} | acc {acc*100:.2f}%")
end = time.time()
print("Training finished, time:", end - start, "s")

# ------------------------
# === After training: compute outputs/latents + total penalty per trial
# ------------------------
outputs_all, latents_all, total_pen_all = unroll_t.apply(params, jax.random.PRNGKey(0), jnp.array(x_reshaped))
outputs_all = np.array(outputs_all)   # (T,N,4)
latents_all = np.array(latents_all)   # (T,N,latent)
total_pen_all = np.array(total_pen_all)  # (N,)

print("Latents shape:", latents_all.shape)
print("outputs_all shape:", outputs_all.shape)
print("total_pen_all shape:", total_pen_all.shape)

final_logits = np.array(outputs_all[-1])

# stable probabilities
max_l = np.max(final_logits, axis=1, keepdims=True)
exp_l = np.exp(final_logits - max_l)
probs4 = exp_l / np.sum(exp_l, axis=1, keepdims=True)
probs4 = np.nan_to_num(probs4, nan=1.0 / probs4.shape[1])

pred_class = np.argmax(probs4, axis=1)
pred_choice = (pred_class >= 2).astype(int)
pred_conf = (pred_class % 2).astype(int)

print("Predicted class unique:", np.unique(pred_class))
print("Human class counts:", np.unique(combined_label, return_counts=True))
print("RNN class counts:", np.unique(pred_class, return_counts=True))
print("Human high-conf rate: {:.2f}%".format(100*np.mean(conf)))
print("RNN high-conf rate (pred): {:.2f}%".format(100*np.mean(pred_conf)))

# ------------------------
# === Sigma/bottleneck summaries (reparameterize as repo: abs(raw)+min_sigma)
# ------------------------
def reparam_sigma(raw):
    return np.abs(np.array(raw)) + min_sigma

if isinstance(hk_core, dict):
    latent_sig = reparam_sigma(hk_core['latent_sigma_params']) if 'latent_sigma_params' in hk_core else None
    choice_sig = reparam_sigma(hk_core['choice_net_sigma_params']) if 'choice_net_sigma_params' in hk_core else None
    update_obs_sig = reparam_sigma(hk_core['update_net_obs_sigma_params']).T if 'update_net_obs_sigma_params' in hk_core else None
    update_latent_sig = reparam_sigma(hk_core['update_net_latent_sigma_params']) if 'update_net_latent_sigma_params' in hk_core else None
else:
    latent_sig = choice_sig = update_obs_sig = update_latent_sig = None

print("\nSigma summary (reparam):")
if latent_sig is not None:
    print(f"  latent_sig: shape={latent_sig.shape} min={latent_sig.min():.6e} max={latent_sig.max():.6e} mean={latent_sig.mean():.6e}")
if choice_sig is not None:
    print(f"  choice_sig: shape={choice_sig.shape} min={choice_sig.min():.6e} max={choice_sig.max():.6e} mean={choice_sig.mean():.6e}")
if update_obs_sig is not None:
    print(f"  update_obs_sig: shape={update_obs_sig.shape} min={update_obs_sig.min():.6e} max={update_obs_sig.max():.6e} mean={update_obs_sig.mean():.6e}")
if update_latent_sig is not None:
    print(f"  update_latent_sig: shape={update_latent_sig.shape} min={update_latent_sig.min():.6e} max={update_latent_sig.max():.6e} mean={update_latent_sig.mean():.6e}")

# Also call repo plotting helpers (they may expect standard keys)
try:
    _ = plotting.plot_bottlenecks(params, disrnn_config)
except Exception as e:
    print("plot_bottlenecks failed:", e, file=sys.stderr)
try:
    _ = plotting.plot_choice_rule(params, disrnn_config)
except Exception as e:
    print("plot_choice_rule failed:", e, file=sys.stderr)
try:
    _ = plotting.plot_update_rules(params, disrnn_config)
except Exception as e:
    print("plot_update_rules failed:", e, file=sys.stderr)

# ------------------------
# === Psychophysical kernels (LOGISTIC) using centered-by-generative-mean stimuli WITHOUT multiplying by 2
# Use R-L differences per timestep as regressors: X shape (trials, timesteps)
stim_diff_for_kernel = (right_centered_for_kernel[:, :, 0] - left_centered_for_kernel[:, :, 0])  # (trials, timesteps)
X_kernel = stim_diff_for_kernel  # already (trials, timesteps)

y_choice_data = choice
y_choice_rnn = pred_choice

def compute_kernel_stable(X_, y_):
    clf = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=2000)
    clf.fit(X_, y_)
    return clf.coef_.flatten(), clf

# Only compute RNN kernel if >1 unique class in predicted choice
if len(np.unique(y_choice_rnn)) > 1:
    kernel_data_choice, clf_data = compute_kernel_stable(X_kernel, y_choice_data)
    kernel_rnn_choice, clf_rnn = compute_kernel_stable(X_kernel, y_choice_rnn)

    plt.figure(figsize=(8,5))
    plt.plot(kernel_data_choice, label='Behavior choice kernel', lw=3)
    plt.plot(kernel_rnn_choice, label='RNN choice kernel', lw=3)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel("Timestep")
    plt.ylabel("Kernel weight (logit coef)")
    plt.title("Choice psychophysical kernels (logistic, generative-mean centered, NO ×2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("RNN produced a single predicted choice class -> skipping logistic kernel fit for RNN.")

# ------------------------
# === Averaging-based kernels split by selected/non-selected × high/low confidence
# For these we use the centered-by-generative-mean stimuli (no *2), and compute selected vs non-selected
# ------------------------
right_stim_kernel = right_centered_for_kernel[:, :, 0]  # (trials, timesteps)
left_stim_kernel = left_centered_for_kernel[:, :, 0]

# selected / non-selected defined by predicted choice (RNN)
selected_rnn = np.where(pred_choice[:, None] == 1, right_stim_kernel, left_stim_kernel)
nonselected_rnn = np.where(pred_choice[:, None] == 1, left_stim_kernel, right_stim_kernel)

rnn_high_mask = pred_conf == 1
rnn_low_mask  = pred_conf == 0

def avg_kernel(contrasts, mask):
    if np.sum(mask) == 0:
        return np.zeros(contrasts.shape[1])
    return np.nanmean(contrasts[mask], axis=0)

kernel_sel_high_rnn = avg_kernel(selected_rnn, rnn_high_mask)
kernel_sel_low_rnn  = avg_kernel(selected_rnn, rnn_low_mask)
kernel_nsel_high_rnn = avg_kernel(nonselected_rnn, rnn_high_mask)
kernel_nsel_low_rnn  = avg_kernel(nonselected_rnn, rnn_low_mask)

# Human kernels using human choice/conf
selected_data = np.where(choice[:, None] == 1, right_stim_kernel, left_stim_kernel)
nonselected_data = np.where(choice[:, None] == 1, left_stim_kernel, right_stim_kernel)
human_high_mask = conf == 1
human_low_mask  = conf == 0

kernel_sel_high_data = avg_kernel(selected_data, human_high_mask)
kernel_sel_low_data  = avg_kernel(selected_data, human_low_mask)
kernel_nsel_high_data = avg_kernel(nonselected_data, human_high_mask)
kernel_nsel_low_data  = avg_kernel(nonselected_data, human_low_mask)

timesteps_idx = np.arange(kernel_sel_low_rnn.shape[0]) + 1
plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(timesteps_idx, kernel_sel_high_rnn, label='Selected - High (RNN)', lw=3)
plt.plot(timesteps_idx, kernel_sel_low_rnn,  label='Selected - Low  (RNN)', lw=3)
plt.plot(timesteps_idx, kernel_nsel_high_rnn, label='NonSel - High (RNN)', lw=2, linestyle='--')
plt.plot(timesteps_idx, kernel_nsel_low_rnn,  label='NonSel - Low  (RNN)', lw=2, linestyle='--')
plt.title("RNN kernels (selected vs non-selected) by predicted conf — centered by generative mean (NO ×2)")
plt.xlabel("Sample")
plt.ylabel("Avg centered contrast")
plt.legend(); plt.grid(True)

plt.subplot(2,1,2)
plt.plot(timesteps_idx, kernel_sel_high_data, label='Selected - High (Human)', lw=3)
plt.plot(timesteps_idx, kernel_sel_low_data,  label='Selected - Low  (Human)', lw=3)
plt.plot(timesteps_idx, kernel_nsel_high_data, label='NonSel - High (Human)', lw=2, linestyle='--')
plt.plot(timesteps_idx, kernel_nsel_low_data,  label='NonSel - Low  (Human)', lw=2, linestyle='--')
plt.title("Human kernels (selected vs non-selected) by reported conf — centered by generative mean")
plt.xlabel("Sample")
plt.ylabel("Avg centered contrast")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()
quit()
# ------------------------
# === Latent plotting: average latent trajectories by selected/non-selected × high/low confidence
# (user requested selected vs non-selected instead of left/right)
# ------------------------
latents_np = latents_all  # (T, N, latent)
timesteps_l, trials_l, latent_sz = latents_np.shape
print("Latents shape for plotting:", latents_np.shape)

# define masks by RNN predicted selected / nonselected and predicted confidence
# selected = predicted choice == 1 (right selected); non-selected conversely
sel_mask = pred_choice == 1
nsel_mask = pred_choice == 0

sel_high = sel_mask & (pred_conf == 1)
sel_low  = sel_mask & (pred_conf == 0)
nsel_high = nsel_mask & (pred_conf == 1)
nsel_low  = nsel_mask & (pred_conf == 0)

masks = [(sel_low, "Selected Low"), (sel_high, "Selected High"),
         (nsel_low, "NonSel Low"), (nsel_high, "NonSel High")]
colors = ["C0","C2","C1","C3"]

print("Trials per predicted selected/conf groups:")
for m,lbl in masks:
    print(lbl, np.sum(m))

# compute global y-limits across latents and groups
global_min = np.inf
global_max = -np.inf
for li in range(latent_sz):
    for m,_ in masks:
        if np.sum(m) > 0:
            avg_curve = np.mean(latents_np[:, m, li], axis=1)
            global_min = min(global_min, np.min(avg_curve))
            global_max = max(global_max, np.max(avg_curve))
if global_min == np.inf:
    global_min, global_max = -1, 1
pad = 0.05*(global_max-global_min)
global_min -= pad; global_max += pad

fig, axes = plt.subplots(latent_sz, 1, figsize=(10, 3*latent_sz), sharex=True, sharey=True)
if latent_sz == 1:
    axes = [axes]

for li in range(latent_sz):
    ax = axes[li]
    for (m, lbl), c in zip(masks, colors):
        if np.sum(m) > 0:
            mean_curve = np.mean(latents_np[:, m, li], axis=1)
            ax.plot(mean_curve, label=lbl, color=c, lw=2)
        else:
            ax.plot(np.zeros(timesteps_l), '--', color=c, alpha=0.4, label=lbl + " (0 trials)")
    sigma_text = f"{latent_sig[li]:.3f}" if (latent_sig is not None and li < len(latent_sig)) else "nan"
    ax.set_title(f"Latent {li} (σ={sigma_text})")
    ax.set_ylim(global_min, global_max)
    ax.grid(True)

axes[-1].set_xlabel("Timestep")
fig.suptitle("Average Latent Activity by Selected/NonSel × Predicted Confidence", fontsize=16)
fig.legend(loc='upper right', bbox_to_anchor=(0.92, 0.98))
plt.tight_layout(rect=[0,0,0.9,0.96])
plt.show()

print("Done — model trained & analyses plotted.")
