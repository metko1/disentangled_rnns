import scipy.io
import numpy as np
import optax
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import haiku as hk
import sys
import fcntl
from disentangled_rnns.library import rnn_utils, disrnn, plotting

# Number of trials to use
trials2use = 0
if len(sys.argv) > 1:
  trials2use = int(sys.argv[1])


# === Load your MATLAB data ===
mat = scipy.io.loadmat('./sbj1.mat')
Xmat = mat['data']  # Check keys with print(mat.keys()) if needed

# === Extract stimuli and choice columns ===
stim_right = Xmat[:, :10]   # Right stimulus, 10 timesteps per trial
stim_left = Xmat[:, 10:20]  # Left stimulus, 10 timesteps per trial
choice = Xmat[:, 20]        # Choices (one per trial)

# === Clean NaNs from choice and filter stim/choice ===
print("Unique choice values before cleaning:", np.unique(choice))
nan_mask = ~np.isnan(choice)
print("Number of NaNs in choice:", np.sum(~nan_mask))

stim_right = stim_right[nan_mask]
stim_left = stim_left[nan_mask]
choice = choice[nan_mask].astype(int)

N_trials = stim_right.shape[0]
if trials2use == 0:
    trials2use = N_trials # if script is called with now argument, use all trials
timesteps = 10
features_per_side = 1

# Reshape stimuli to (N_trials, timesteps, 1)
stim_right = stim_right.reshape(N_trials, timesteps, features_per_side)
stim_left = stim_left.reshape(N_trials, timesteps, features_per_side)

# Concatenate right and left features: (N_trials, timesteps, 2)
x_concat = np.concatenate([stim_right, stim_left], axis=2)

# Transpose to (timesteps, trials, features)
x_reshaped = np.transpose(x_concat, (1, 0, 2))  # IMPORTANT FIX HERE

# Prepare y with shape (timesteps, trials, 1)
y_repeated = np.tile(choice[:, np.newaxis], (1, timesteps))  # (N_trials, timesteps)
y_reshaped = y_repeated.T[:, :, np.newaxis]  # (timesteps, trials, 1)

# Names
x_names = ['right', 'left']
y_names = ['choice']

# Dataset class expected by disentangled_rnns
class MyDataset:
    def __init__(self, x, y, x_names=None, y_names=None, batch_size=16, batch_mode='single'):
        self.x = x
        self.y = y
        self.x_names = x_names or []
        self.y_names = y_names or []
        self.x_type = 'float'
        self.y_type = 'categorical'  # classification task
        self.n_classes = len(np.unique(choice))
        self.batch_size = batch_size
        self.batch_mode = batch_mode  # Must be one of 'single', 'rolling', or 'random'

    def get_all(self):
        return self.x, self.y

#idx = np.random.randint(0, N_trials, trials2use)
#x_reshaped = x_reshaped[:,idx,:]
#y_reshaped = y_reshaped[:,idx,:]
my_dataset = MyDataset(x_reshaped, y_reshaped, x_names, y_names, batch_size=16, batch_mode='single')

# Split dataset: every 2nd trial is eval
dataset_train, dataset_eval = rnn_utils.split_dataset(my_dataset, 2)

# Configure Disentangled RNN
disrnn_config = disrnn.DisRnnConfig(
    obs_size=x_reshaped.shape[2],  # 2 features (right, left)
    output_size=2,                 # binary classification
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

# Initialize network (no training yet)
params, opt_state, losses = rnn_utils.train_network(
    lambda: disrnn.HkDisentangledRNN(disrnn_config),
    dataset_train,
    dataset_eval,
    opt=optax.adam(1e-3),
    loss="penalized_categorical",
    n_steps=0
)

# Train the network
params, opt_state, losses = rnn_utils.train_network(
    lambda: disrnn.HkDisentangledRNN(disrnn_config),
    dataset_train,
    dataset_eval,
    loss="penalized_categorical",
    params=params,
    opt_state=opt_state,
    opt=optax.adam(1e-3),
    loss_param=1,
    n_steps=4000,
    do_plot=True,
)

# Plot bottlenecks and learned rules
_ = plotting.plot_bottlenecks(params, disrnn_config)
_ = plotting.plot_choice_rule(params, disrnn_config)
_ = plotting.plot_update_rules(params, disrnn_config)
plt.show()
quit()
# === Extract sigma values from params ===
# The sigma for each latent unit is stored in the update network parameters, usually under 'latent_sigma'
# Let's locate it safely:

def find_sigma_in_params(params):
    # Recursively search for any key containing 'sigma' (case insensitive)
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

ssigmas = np.sort(sigmas_np)
with open("results.txt", "a") as file:
    fcntl.flock(file.fileno(), fcntl.LOCK_EX)
    file.write(str(trials2use) + " " + str(ssigmas)[1:-1] + " " +
               str(np.mean(losses['training_loss'][-1])) + " " +
               str(np.mean(losses['validation_loss'][-1])) + "\n")
    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
quit()


# === Extract and plot latent activity ===
def forward_latents(x_seq):
    model = disrnn.HkDisentangledRNN(disrnn_config)
    state = model.initial_state(x_seq.shape[1])  # batch size = number of trials
    latents = []

    for t in range(x_seq.shape[0]):
        _, state = model(x_seq[t], state)
        latents.append(state)  # state is (trials, latent_size)

    latents = jnp.stack(latents, axis=0)  # (timesteps, trials, latent_size)
    return latents

# Transform forward function with Haiku
forward_latents_t = hk.transform(forward_latents)

# Prepare RNG key
rng = jax.random.PRNGKey(42)

# Apply the model with trained params to get latent states
latents = forward_latents_t.apply(params, rng, jnp.array(x_reshaped))
latents = np.array(latents)  # (timesteps, trials, latent_size)


# === Compute cumulative evidence per trial (total right - left stimulus) ===
stim_diff = x_reshaped[:, :, 0] - x_reshaped[:, :, 1]  # shape: (timesteps, trials)
total_evidence_per_trial = np.sum(stim_diff, axis=0)  # shape: (trials,)

# Split trials into those with more rightward or leftward evidence
right_dominant_trials = total_evidence_per_trial > 0
left_dominant_trials = total_evidence_per_trial < 0

print(f"# Right-dominant trials: {np.sum(right_dominant_trials)}")
print(f"# Left-dominant trials: {np.sum(left_dominant_trials)}")

# === Find latent with lowest sigma ===
lowest_sigma_idx = np.argmin(sigmas_np)
print(f"Latent with lowest sigma: {lowest_sigma_idx} (σ = {sigmas_np[lowest_sigma_idx]:.3f})")

# === Compute average latent activity for that latent ===
# Shape of latents: (timesteps, trials, latent_size)
avg_latent_right = np.mean(latents[:, right_dominant_trials, lowest_sigma_idx], axis=1)
avg_latent_left = np.mean(latents[:, left_dominant_trials, lowest_sigma_idx], axis=1)

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(avg_latent_right, label='Right-dominant Trials', color='blue')
plt.plot(avg_latent_left, label='Left-dominant Trials', color='red')
plt.title(f"Average Activity of Latent {lowest_sigma_idx} (σ={sigmas_np[lowest_sigma_idx]:.3f})")
plt.xlabel("Timestep")
plt.ylabel("Average Latent Activity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)



def analyze_latent_accumulation_over_time(latents, x_reshaped):
    """
    Analyze how well latent units track accumulated stimulus evidence (right - left) over time.
    """
    timesteps, trials, latent_size = latents.shape

    # Compute cumulative evidence over time: shape (timesteps, trials)
    stimulus_diff = x_reshaped[:, :, 0] - x_reshaped[:, :, 1]
    cumulative_evidence = np.cumsum(stimulus_diff, axis=0)  # (timesteps, trials)

    # Compute correlation at each timestep between latent and cumulative evidence
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

    # Plot correlation of each latent with cumulative evidence over time
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

# Plot latent activity for a single trial WITH sigma in legend
trial_idx = 10  # Change to visualize different trials
num_latents_to_plot = disrnn_config.latent_size

plt.figure(figsize=(10, 6))
for latent_idx in range(num_latents_to_plot):
    plt.plot(latents[:, trial_idx, latent_idx], label=f'Latent {latent_idx} (σ={sigmas_np[latent_idx]:.3f})')
plt.xlabel('Timestep')
plt.ylabel('Latent Activity')
plt.title(f'Latent Activity Over Time (Trial {trial_idx})')
plt.legend()
plt.tight_layout()
plt.show()

print("Training complete! You can continue training by rerunning the training step.")
