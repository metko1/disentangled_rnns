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
# confidence in 21, delta/generative mean in column 22,
# session no. (1..6) in column 24, block no. (1..6) in column 25
nan_mask = ~np.isnan(Xmat[:,0])
Xmat = Xmat[nan_mask]
choice = Xmat[:, 0]
stim_left = Xmat[:, 1:11]
stim_right  = Xmat[:, 11:21]
conf = Xmat[:, 21]
timesteps = 10
N_trials = stim_right.shape[0]
x_names = ['right', 'left']
y_names = ['choice', 'confidence']

# sessions = np.unique(Xmat[:, 24])
# blocks = np.unique(Xmat[:, 25])
# datasets = []
# for s in sessions:
#   for b in blocks:
#     selection = (Xmat[:, 24] == s) & (Xmat[:, 25] == b)
#     # Concatenate right and left features: (N_trials, timesteps, 2)
#     x_concat = np.stack([stim_right[selection,:], stim_left[selection,:]], axis=2)
#     # Transpose to (timesteps, trials, features)
#     x_reshaped = np.transpose(x_concat, (1,0,2))
#     N_trials = np.sum(selection)
#     # Prepare y with shape (timesteps, trials, 2)
#     y = np.full((timesteps, N_trials,2), -1)
#     y[-1,:,0] = choice[selection]
#     y[-1,:,1] = conf[selection]
#
#     # batch learning - no history
#     # ds = rnn_utils.DatasetRNN(x_reshaped, y, y_type='categorical',
#     #                           x_names=x_names, y_names=y_names,
#     #                           batch_mode='single') # batch_size=16 not relevant for batch_mode='single'
#     # train on one trial at a time, but with history across trials (i.e. hidden state is not reset between trials)
#     ds = rnn_utils.DatasetRNN(x_reshaped, y, y_type='categorical',
#                               x_names=x_names, y_names=y_names,
#                               batch_size=1, batch_mode='rolling') # history across trials, not time steps
#     datasets.append(ds)
# validation_ds = datasets.pop() # Last session vor validation

# Concatenate right and left features: (N_trials, timesteps, 2)
x_concat = np.stack([stim_right, stim_left], axis=2)
# Transpose to (timesteps, trials, features)
x_reshaped = np.transpose(x_concat, (1,0,2))
# Prepare y with shape (timesteps, trials, 2)
y = np.full((timesteps, N_trials,2), -1)
y[-1,:,0] = choice
y[-1,:,1] = conf

# select the validation block
b = 1
val_idx = Xmat[:, 25] == b
ds_train = rnn_utils.DatasetRNN(x_reshaped[:,np.logical_not(val_idx),:],
                                y[:,np.logical_not(val_idx),:], y_type='categorical',
                                x_names=x_names, y_names=y_names,
                                batch_size=1, batch_mode='rolling') # history across trials, not time steps
                                #batch_mode = 'single')  # history across trials, not time steps
ds_valid = rnn_utils.DatasetRNN(x_reshaped[:,val_idx,:],
                                y[:,val_idx,:], y_type='categorical',
                                x_names=x_names, y_names=y_names,
                                batch_size=1, batch_mode='rolling') # history across trials, not time steps
                                #batch_mode = 'single')  # history across trials, not time steps


#dataset_train, dataset_eval = rnn_utils.split_dataset(dataset, 2)

#with open("datasets.pkl", "wb") as f:
#  pickle.dump(dataset, f)
#  pickle.dump(dataset_train, f)
#  pickle.dump(dataset_eval, f)

config = two_output_disrnn.DisRnnWTwoOutputConfig(
  # Dataset related
  obs_size=2,  # left, right
  output_size=2,  # Choose left or right
  x_names=ds_train.x_names,
  y_names=ds_train.y_names,
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
  latent_penalty=1e-3,
  choice_net_latent_penalty=5e-4,
  update_net_obs_penalty=5e-4,
  update_net_latent_penalty=5e-4,
  confidence_net_latent_penalty=1e-3
  )

# Initial training in noiseless mode
params, opt_state, losses = rnn_utils.train_network(
  lambda: two_output_disrnn.HkTwoOutputDisentangledRNN(config),
  ds_train,
  ds_valid,
  opt = optax.adam(1e-3),
  loss="penalized_categorical2",
  #loss_param={'likelihood_weight': 1.0, 'penalty_scale': 1.0},
  n_steps=0)

params, opt_state, losses = rnn_utils.train_network(
  lambda: two_output_disrnn.HkTwoOutputDisentangledRNN(config),
  ds_train,
  ds_valid,
  loss="penalized_categorical2",
  params=params,
  opt_state=opt_state,
  opt = optax.adam(1e-3), # was 1e-3, but with history we need a smaller learning rate to prevent divergence
  loss_param = 1,
  #loss_param = {'likelihood_weight': 1.0, 'penalty_scale': 1.0},
  n_steps=20000,
  do_plot = True,
  log_losses_every=100)

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

