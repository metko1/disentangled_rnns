import scipy.io
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import haiku as hk
from disentangled_rnns.library import rnn_utils, disrnn, plotting

# === Load your MATLAB data ===
mat = scipy.io.loadmat('/Users/tosot/Downloads/Data_forAle.mat')
x = mat['samp']          # samples in degrees
y = mat['choices']
nsamps = mat['nsamps']
dist_lr = mat['dist_lr']

# === Pad variable-length trials, aligning to the last sample ===
def pad_and_align_to_last(x, nsamps):
    """
    Pad each trial with zeros on the left so that the last sample is aligned.
    Args:
        x: array (n_trials, max_timesteps)
        nsamps: array (n_trials, 1) number of valid samples in each trial
    Returns:
        padded_x: array (n_trials, max_len) with left-zero padding
    """
    n_trials, max_len = x.shape
    padded = np.zeros_like(x)
    for i in range(n_trials):
        n = int(nsamps[i, 0])
        if n > 0:
            padded[i, -n:] = x[i, :n]
    return padded

# Find longest sequence
timesteps = int(np.max(nsamps))
print(f"Max number of samples across trials: {timesteps}")

# Pad all trials to equal length, aligned to last sample
x_padded = pad_and_align_to_last(x, nsamps)

# Use all trials
N_trials = len(y)
print("Total number of trials:", N_trials)

dominance = np.mean(dist_lr, axis=1)
n_left = np.sum(dominance < 0.5)
print(f"Left source dominated in {n_left} trials, right in {N_trials - n_left}")

# === Create two observation channels: left and right evidence ===
obs1 = np.clip(-x_padded, 0, None)  # leftward (negative) values only
obs2 = np.clip(x_padded, 0, None)   # rightward (positive) values only

x_reshaped = np.stack((obs1, obs2), axis=2)
x_reshaped = np.transpose(x_reshaped, (1, 0, 2))  # (timesteps, trials, features)

# Prepare y with shape (timesteps, trials, 1)
y_repeated = np.tile(y, (1, timesteps))  # (N_trials, timesteps)
y_reshaped = y_repeated.T[:, :, np.newaxis]  # (timesteps, trials, 1)

# Names
x_names = ['source1', 'source2']
y_names = ['choice']

# === Dataset class expected by disentangled_rnns ===
class MyDataset:
    def __init__(self, x, y, x_names=None, y_names=None, batch_size=16, batch_mode='single'):
        self.x = x
        self.y = y
        self.x_names = x_names or []
        self.y_names = y_names or []
        self.x_type = 'float'
        self.y_type = 'categorical'
        self.n_classes = len(np.unique(y))
        self.batch_size = batch_size
        self.batch_mode = batch_mode

    def get_all(self):
        return self.x, self.y

my_dataset = MyDataset(x_reshaped, y_reshaped, x_names, y_names, batch_size=16, batch_mode='single')

# Split dataset: every 2nd trial is eval
dataset_train, dataset_eval = rnn_utils.split_dataset(my_dataset, 2)

# === Configure Disentangled RNN ===
disrnn_config = disrnn.DisRnnConfig(
    obs_size=x_reshaped.shape[2],
    output_size=2,
    x_names=x_names,
    y_names=y_names,
    latent_size=5,
    update_net_n_units_per_layer=16,
    update_net_n_layers=4,
    choice_net_n_units_per_layer=4,
    choice_net_n_layers=2,
    activation='leaky_relu',
    noiseless_mode=False,
    latent_penalty=1e-2,
    choice_net_latent_penalty=1e-2,
    update_net_obs_penalty=1e-2,
    update_net_latent_penalty=1e-2,
)

# === Initialize network (no training yet) ===
params, opt_state, losses = rnn_utils.train_network(
    lambda: disrnn.HkDisentangledRNN(disrnn_config),
    dataset_train,
    dataset_eval,
    opt=optax.adam(1e-3),
    loss="penalized_categorical",
    n_steps=0
)

# === Train the network ===
params, opt_state, losses = rnn_utils.train_network(
    lambda: disrnn.HkDisentangledRNN(disrnn_config),
    dataset_train,
    dataset_eval,
    loss="penalized_categorical",
    params=params,
    opt_state=opt_state,
    opt=optax.adam(1e-3),
    loss_param=1,
    n_steps=2500,
    do_plot=True,
)

# === Plot bottlenecks and learned rules ===
_ = plotting.plot_bottlenecks(params, disrnn_config)
_ = plotting.plot_choice_rule(params, disrnn_config)
_ = plotting.plot_update_rules(params, disrnn_config)

# === Extract sigma values from params ===
def find_sigma_in_params(params):
    def recursive_search(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if 'sigma' in k.lower():
                    return v
                elif isinstance(v, dict):
                    found = recursive_search(v)
                    if found is not None:
                        return found
        return None
    return recursive_search(params)

sigmas = find_sigma_in_params(params)
if sigmas is None:
    raise ValueError("Could not find sigma parameters in params dict.")
sigmas_np = np.array(sigmas).flatten()
print(f"Extracted latent sigmas: {sigmas_np}")

# === Extract and plot latent activity ===
def forward_latents(x_seq):
    model = disrnn.HkDisentangledRNN(disrnn_config)
    state = model.initial_state(x_seq.shape[1])  # batch size = number of trials
    latents = []

    for t in range(x_seq.shape[0]):
        _, state = model(x_seq[t], state)
        latents.append(state)  # (trials, latent_size)

    latents = jnp.stack(latents, axis=0)  # (timesteps, trials, latent_size)
    return latents

forward_latents_t = hk.transform(forward_latents)
rng = jax.random.PRNGKey(42)
latents = forward_latents_t.apply(params, rng, jnp.array(x_reshaped))
latents = np.array(latents)

# === Plot latent activity for trials ending in left/right source ===
last_source = dist_lr[:, -1].astype(int)
timesteps, n_trials, latent_size = latents.shape
right_last_mask = last_source == 1
left_last_mask = last_source == 0

print(f"Trials with right as last source: {np.sum(right_last_mask)}")
print(f"Trials with left as last source:  {np.sum(left_last_mask)}")

avg_latent_rightlast = np.mean(latents[:, right_last_mask, :], axis=1)
avg_latent_leftlast  = np.mean(latents[:, left_last_mask, :], axis=1)

plt.figure(figsize=(12, 2.2 * latent_size))
for li in range(latent_size):
    plt.subplot(latent_size, 1, li + 1)
    plt.plot(avg_latent_rightlast[:, li], label='Right-last', linestyle='-')
    plt.plot(avg_latent_leftlast[:, li], label='Left-last', linestyle='--')
    plt.title(f'Latent {li} — avg activity (Right-last vs Left-last)')
    plt.xlabel('Timestep')
    plt.ylabel('Activity')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# === Plot the most stable latent (lowest sigma) ===
lowest_sigma_idx = np.argmin(sigmas_np)
plt.figure(figsize=(8, 4))
plt.plot(avg_latent_rightlast[:, lowest_sigma_idx], label='Right-last')
plt.plot(avg_latent_leftlast[:, lowest_sigma_idx], label='Left-last')
plt.title(f'Latent {lowest_sigma_idx} (σ={sigmas_np[lowest_sigma_idx]:.3f}) — avg by last source')
plt.xlabel('Timestep')
plt.ylabel('Activity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# === Evidence-based analysis ===
stim_diff = x_reshaped[:, :, 0] - x_reshaped[:, :, 1]
total_evidence_per_trial = np.sum(stim_diff, axis=0)
right_dominant_trials = total_evidence_per_trial > 0
left_dominant_trials = total_evidence_per_trial < 0

print(f"# Right-dominant trials: {np.sum(right_dominant_trials)}")
print(f"# Left-dominant trials: {np.sum(left_dominant_trials)}")

avg_latent_right = np.mean(latents[:, right_dominant_trials, lowest_sigma_idx], axis=1)
avg_latent_left = np.mean(latents[:, left_dominant_trials, lowest_sigma_idx], axis=1)

plt.figure(figsize=(8, 5))
plt.plot(avg_latent_right, label='Right-dominant', color='blue')
plt.plot(avg_latent_left, label='Left-dominant', color='red')
plt.title(f"Average Activity of Latent {lowest_sigma_idx} (σ={sigmas_np[lowest_sigma_idx]:.3f})")
plt.xlabel("Timestep")
plt.ylabel("Average Latent Activity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# === Correlation analysis ===
def analyze_latent_accumulation_over_time(latents, x_reshaped):
    timesteps, trials, latent_size = latents.shape
    stimulus_diff = x_reshaped[:, :, 0] - x_reshaped[:, :, 1]
    cumulative_evidence = np.cumsum(stimulus_diff, axis=0)
    correlations_over_time = np.zeros((timesteps, latent_size))
    for t in range(timesteps):
        for i in range(latent_size):
            latent_vals = latents[t, :, i]
            ev_vals = cumulative_evidence[t, :]
            if np.std(latent_vals) > 1e-5 and np.std(ev_vals) > 1e-5:
                corr = np.corrcoef(latent_vals, ev_vals)[0, 1]
            else:
                corr = 0.0
            correlations_over_time[t, i] = corr
    plt.figure(figsize=(10, 6))
    for i in range(latent_size):
        plt.plot(range(timesteps), correlations_over_time[:, i], label=f'Latent {i}')
    plt.xlabel("Timestep")
    plt.ylabel("Correlation with Cumulative Evidence")
    plt.title("Latent–Evidence Correlation Over Time")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    return correlations_over_time

correlations_over_time = analyze_latent_accumulation_over_time(latents, x_reshaped)

# === Plot latent activity for a single trial ===
trial_idx = 10
plt.figure(figsize=(10, 6))
for latent_idx in range(disrnn_config.latent_size):
    plt.plot(latents[:, trial_idx, latent_idx],
             label=f'Latent {latent_idx} (σ={sigmas_np[latent_idx]:.3f})')
plt.xlabel('Timestep')
plt.ylabel('Latent Activity')
plt.title(f'Latent Activity Over Time (Trial {trial_idx})')
plt.legend()
plt.tight_layout()
plt.show()

print("Training complete! You can continue training by rerunning the training step.")
