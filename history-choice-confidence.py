import pickle
import optax
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from disentangled_rnns.library import rnn_utils, two_output_disrnn
from disentangled_rnns.library import checkpoint_utils

#import jax
#jax.config.update("jax_disable_jit", True)

# considers a whole block as episodes of a single trial
# input delta of the generative mean and feedback from previous trial
mat = scipy.io.loadmat('data/All_data_feedback.mat')
data = mat['All_Data']  # Check keys with print(mat.keys()) if needed
timesteps = 10
datasets = []
x = np.empty((119,0,2))
y = np.empty((119,0,2))
for Xmat in data[:,1]: # placebo
  nan_mask = ~np.isnan(Xmat[:,0])
  #print(np.sum(nan_mask))
  Xmat = Xmat[nan_mask]
  delta_mean = Xmat[:,22]
  feedback = Xmat[:, 27]
  choice = Xmat[:, 0]
  conf = Xmat[:, 21]
  trans_prob = Xmat[:, 26]
  sessions = np.unique(Xmat[:, 24])
  blocks = np.unique(Xmat[:, 25])
  for s in sessions:
    for b in blocks:
      selection = (Xmat[:, 24] == s) & (Xmat[:, 25] == b)
      if np.sum(selection) < 2: # need at least 2 trials to have a history
        print(f"No trials in session {s}, block {b}")
        continue
      if trans_prob[selection][0] == 0.5: # only consider repetitive trials
        continue
      # Prepare x with shape (trials, 1, 2)
      xdelta_mean = delta_mean[selection]
      xdelta_mean = xdelta_mean[1:] # remove first trial because it has no feedback
      xfeedback = feedback[selection]
      xfeedback = xfeedback[:-1] # remove last trial because it has no next trial to predict
      row = np.stack([xdelta_mean, xfeedback], axis=1)
      if row.shape[0] < 119:
        missing = 119 - row.shape[0]
        print(f"block {b} session {s}: {missing} trials missing, padding with zeros")
        row = np.pad(row, ((0, missing), (0, 0)), mode='constant', constant_values=0)
      x = np.append(x, row[:,None,:], axis=1)
      # Prepare y with shape (trials, 1, 2)
      ychoice = choice[selection]
      ychoice = ychoice[1:] # remove first trial because it has no feedback
      yconf = conf[selection]
      yconf = yconf[1:] # remove first trial because it has no feedback
      row = np.stack([ychoice, yconf], axis=1)
      if row.shape[0] < 119:
        missing = 119 - row.shape[0]
        #print(f"block {b} session {s}: {missing} trials missing, padding with zeros")
        row = np.pad(row, ((0, missing), (0, 0)), mode='constant', constant_values=-1)
      y = np.append(y, row[:,None,:], axis=1)

N_trials = x.shape[1]
x_names = ['delta_contrast', 'feedback']
y_names = ['choice', 'confidence']

# select the validation block
dataset = rnn_utils.DatasetRNN(x, y, y_type='categorical',
                                x_names=x_names, y_names=y_names,
                                batch_mode = 'single')
ds_train, ds_valid = rnn_utils.split_dataset(dataset, eval_every_n=4)
#with open("datasets.pkl", "wb") as f:
#  pickle.dump(dataset, f)
#  pickle.dump(dataset_train, f)
#  pickle.dump(dataset_eval, f)

config = two_output_disrnn.DisRnnWTwoOutputConfig(
  # Dataset related
  obs_size=2,  # delta mean contrast, previous feedback
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
  choice_net_latent_penalty=1e-3,
  update_net_obs_penalty=1e-3,
  update_net_latent_penalty=1e-3,
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
  n_steps=20_000,
  do_plot = True,
  log_losses_every=100)

choice_sigmas = params['hk_two_output_disentangled_rnn']['choice_net_sigma_params']
print(f"choice bn sigmas:{choice_sigmas}")
confidence_sigmas = params['hk_two_output_disentangled_rnn']['confidence_net_sigma_params']
print(f"confidence bn sigmas: {confidence_sigmas}")

with open("params.pkl", "wb") as f:
  pickle.dump(config, f)
  pickle.dump(params, f)

two_output_disrnn.plot_bottlenecks(params, config, sort_latents=False)
two_output_disrnn.plot_choice_rule(params, config)
two_output_disrnn.plot_confidence_rule(params, config)
#two_output_disrnn.plot_update_rules(params, disrnn_w_two_output_config)
plt.show()

