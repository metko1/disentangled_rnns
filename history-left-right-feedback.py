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
# input is stim_left, stim right and feedback from previous trial
# doesn't work as long as disRNN allows strictly two inputs
mat = scipy.io.loadmat('data/All_data_feedback.mat')
data = mat['All_Data']  # Check keys with print(mat.keys()) if needed
timesteps = 10
datasets = []
x = np.empty((1200,0,2))
y = np.empty((1200,0,2))
for Xmat in data[:,1]: # placebo
  nan_mask = ~np.isnan(Xmat[:,0])
  Xmat = Xmat[nan_mask]
  choice = Xmat[:, 0]
  stim_left = Xmat[:, 1:11]
  stim_right  = Xmat[:, 11:21]
  conf = Xmat[:, 21]
  correct = Xmat[:, 27]
  trans_prob = Xmat[:, 26]
  sessions = np.unique(Xmat[:, 24])
  blocks = np.unique(Xmat[:, 25])
  for s in sessions:
    for b in blocks:
      selection = (Xmat[:, 24] == s) & (Xmat[:, 25] == b)
      if trans_prob[selection][0] == 0.5: # only consider repetitive trials
        continue
      xr = stim_right[selection,:]
      xr = xr[1:,:] # remove first trial of the block, since it doesn't have feedback
      xl = stim_left[selection,:]
      xl = xl[1:,:] # remove first trial of the block, since it doesn't have feedback
      xc = correct[selection]
      xc = xc[:-1] # remove last trial of the block, since it doesn't have a next trial to predict
      xc = np.repeat(xc[:,None], timesteps, axis=1).flatten()
      row = np.stack([xr.flatten(), xl.flatten(), xc.flatten()], axis=1)
      x = np.append(x, row[:,None,:], axis=1)
      N_trials = np.sum(selection)
      # Prepare y with shape (timesteps, trials, 2)
      y_choice = np.full((timesteps,N_trials), -1)
      y_choice[-1,:] = choice[selection]
      y_conf = np.full((timesteps,N_trials), -1)
      y_conf[-1,:] = conf[selection]
      row = np.stack([y_choice.flatten(), y_conf.flatten()], axis=1)
      y = np.append(y, row[:,None,:], axis=1)

N_trials = stim_right.shape[0]
x_names = ['right', 'left']
y_names = ['choice', 'confidence']

# select the validation block
ds_train = rnn_utils.DatasetRNN(x[:,:-1,:],
                                y[:,:-1,:], y_type='categorical',
                                x_names=x_names, y_names=y_names,
                                batch_mode = 'single')
ds_valid = rnn_utils.DatasetRNN(x[:,-1:,:],
                                y[:,-1:,:], y_type='categorical',
                                x_names=x_names, y_names=y_names,
                                batch_mode = 'single')

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
  latent_penalty=1e-5,
  choice_net_latent_penalty=1e-5,
  update_net_obs_penalty=1e-5,
  update_net_latent_penalty=1e-5,
  confidence_net_latent_penalty=1e-5
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

