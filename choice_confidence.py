import pickle
import optax
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from disentangled_rnns.library import rnn_utils, two_output_disrnn
from disentangled_rnns.library import checkpoint_utils

#import jax
#jax.config.update("jax_disable_jit", True)

mat = scipy.io.loadmat('./sbj10.mat')
Xmat = mat['data']  # Check keys with print(mat.keys()) if needed

# for sbj10.mat
# choice in column 0, right stimuli in 1:11, left in 11:21,
# conf in 21, delta/generative mean in column 23 (index 22)
nan_mask = ~np.isnan(Xmat[:,0])
Xmat = Xmat[nan_mask]
choice = Xmat[:, 0]
stim_left = Xmat[:, 1:11]
stim_right  = Xmat[:, 11:21]
conf = Xmat[:, 21]
timesteps = 10
N_trials = stim_right.shape[0]

# Concatenate right and left features: (N_trials, timesteps, 2)
x_concat = np.stack([stim_right, stim_left], axis=2)
# Transpose to (timesteps, trials, features)
x_reshaped = np.transpose(x_concat, (1,0,2))

# Concatenate right and left features and choice from previous trial: (N_trials, timesteps, 2)
#c = np.tile(choice[:,np.newaxis], (1,timesteps))
#x_concat = np.stack([stim_right[1:,:], stim_left[1:,:], c[0:-1,:]])

# Prepare y with shape (timesteps, trials, 2)
y = np.full((timesteps, N_trials,2), -1)
y[-1,:,0] = choice
y[-1,:,1] = conf

x_names = ['right', 'left']
y_names = ['choice', 'confidence']

dataset = rnn_utils.DatasetRNN(x_reshaped, y, y_type='categorical',
                                  x_names=x_names, y_names=y_names,
                                  batch_size=16, batch_mode='single')

dataset_train, dataset_eval = rnn_utils.split_dataset(dataset, 2)
with open("datasets.pkl", "wb") as f:
  pickle.dump(dataset, f)
  pickle.dump(dataset_train, f)
  pickle.dump(dataset_eval, f)

config = two_output_disrnn.DisRnnWTwoOutputConfig(
  # Dataset related
      obs_size=2,  # left, right
      output_size=2,  # Choose left or right
      x_names=dataset.x_names,
      y_names=dataset.y_names,
      # Network architecture
      latent_size=5,
      update_net_n_units_per_layer=16,
      update_net_n_layers=4,
      choice_net_n_units_per_layer=4,
      choice_net_n_layers=2,
      confidence_net_n_units_per_layer=4,
      confidence_net_n_layers=2,
      activation='leaky_relu',
      # Penalties
      noiseless_mode=False,
      latent_penalty=1e-2,
      choice_net_latent_penalty=1e-2,
      update_net_obs_penalty=1e-2,
      update_net_latent_penalty=1e-2,
      confidence_net_latent_penalty=1e-2
)

# Initial training in noiseless mode
params, opt_state, losses = rnn_utils.train_network(
    lambda: two_output_disrnn.HkTwoOutputDisentangledRNN(config),
    dataset_train,
    dataset_eval,
    opt = optax.adam(1e-3),
    loss="penalized_categorical2",
    #loss_param={'likelihood_weight': 1.0, 'penalty_scale': 1.0},
    n_steps=0)

# Usually 20,000 steps in total should be sufficient.
n_steps = 20_000

params, opt_state, losses = rnn_utils.train_network(
    lambda: two_output_disrnn.HkTwoOutputDisentangledRNN(config),
    dataset_train,
    dataset_eval,
    loss="penalized_categorical2",
    params=params,
    opt_state=opt_state,
    opt = optax.adam(1e-3),
    loss_param = 1,
    #loss_param = {'likelihood_weight': 1.0, 'penalty_scale': 1.0},
    n_steps=n_steps,
    do_plot = True)

choice_sigmas = params['hk_two_output_disentangled_rnn']['choice_net_sigma_params']
print(f"choice bn sigmas:{choice_sigmas}")
confidence_sigmas = params['hk_two_output_disentangled_rnn']['confidence_net_sigma_params']
print(f"confidence bn sigmas: {confidence_sigmas}")

#with open("params.pkl", "wb") as f:
#  pickle.dump(config, f)
#  pickle.dump(params, f)

two_output_disrnn.plot_bottlenecks(params, config, sort_latents=False)
two_output_disrnn.plot_choice_rule(params, config)
two_output_disrnn.plot_confidence_rule(params, config)
#two_output_disrnn.plot_update_rules(params, disrnn_w_two_output_config)
plt.show()

