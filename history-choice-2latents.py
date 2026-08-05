import argparse
from math import pi
import pickle
import optax
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from disentangled_rnns.library import rnn_utils, disrnn, plotting
#from disentangled_rnns.library import checkpoint_utils

#import jax
#jax.config.update("jax_disable_jit", True)

parser = argparse.ArgumentParser(description="Fine-tunes disRNN models.")
parser.add_argument('--result-path', type=str, default=".", help="Path to save results (params, configs).")
parser.add_argument('--beta', type=float, default=1e-3, help="general penalty")
parser.add_argument('--train-steps', type=int, default=1e5, help="training steps")
parser.add_argument('--learning-curve', type=bool, default=True, help="show learning curve")

args = parser.parse_args()

# Drugs: subjects(20) X sessions (6)
#  1.Lorazepam, 2.placebo, 3.Memantine
Drugs=np.array([[3, 1, 1, 2, 3, 2], 
                [3, 1, 3, 2, 1, 2], 
                [1, 3, 2, 1, 3, 2], 
                [2, 1, 1, 3, 2, 3], 
                [1, 3, 2, 3, 1, 2],
                [3, 1, 1, 2, 2, 3], 
                [1, 2, 2, 3, 3, 1], 
                [3, 1, 2, 1, 3, 2], 
                [1, 3, 3, 2, 2, 1], 
                [2, 1, 3, 2, 3, 1],
                [1, 2, 3, 2, 1, 3],
                [1, 2, 2, 3, 1, 3],
                [2, 1, 3, 1, 2, 3],
                [2, 3, 1, 2, 1, 3],
                [1, 2, 3, 1, 3, 2],
                [2, 1, 1, 2, 3, 3],
                [3, 1, 2, 2, 1, 3],
                [2, 3, 2, 1, 1, 3],
                [3, 1, 1, 2, 3, 2],
                [3, 2, 1, 3, 1, 2]])

# considers a whole block as episodes of a single trial
# input delta of the generative mean and feedback from previous trial
mat = scipy.io.loadmat('data/data2alex.mat')
Xmat = mat['All_Data_s']
mat = scipy.io.loadmat('data/Drug_id.mat')
drug = mat['All_Drug']
nan_mask = ~np.isnan(Xmat[:,0])
#print(np.sum(nan_mask))
Xmat = Xmat[nan_mask]
drug = drug[nan_mask]
Xmat = Xmat[drug[:,0] == 2,:] # only consider placebo sessions
delta_mean = Xmat[:,22]
feedback = Xmat[:, 27]
choice = Xmat[:, 0]
#conf = Xmat[:, 21]
trans_prob = Xmat[:, 26]
#sessions = np.unique(Xmat[:, 24]).astype(int)
#blocks = np.unique(Xmat[:, 25]).astype(int)

timesteps = 10
datasets = []
x = np.empty((119,0,2))
y = np.empty((119,0,1), dtype=int)
repeat = np.empty(0, dtype=bool)

for sub in range(1,21):
  #d = Drugs[sub-1] # drug condition for this subject
  for sess in range(1,7):
    for blk in range(1,8):
      #print(f"subject {sub}, session {sess}, block {blk}")
      selection = (Xmat[:,23] == sub) & (Xmat[:, 25] == sess) & (Xmat[:, 24] == blk)
      if np.sum(selection) < 2: # need at least 2 trials to have a history
        #print(f"No trials in subject {sub}, block {blk}, session {sess}.")
        continue
      if np.sum(selection) > 120: 
        #print(f"Too many trials ({np.sum(selection)}) for subject {sub} in block {blk} session {sess}, skipping")
        continue
      #if trans_prob[selection][0] == 0.5: # only consider repetitive trials
      #  continue
      # Prepare x with shape (trials, 1, 2)
      xdelta_mean = delta_mean[selection]
      xdelta_mean = xdelta_mean[1:] # remove first trial because it has no feedback
      xfeedback = feedback[selection]
      xfeedback = xfeedback[:-1] # remove last trial because it has no next trial to predict
      row = np.stack([xdelta_mean, xfeedback], axis=1)
      if row.shape[0] < 119:
        missing = 119 - row.shape[0]
        #print(f"Subject {sub}, session {sess}, block {blk}: {missing} trial(s) missing, padding with zeros")
        row = np.pad(row, ((0, missing), (0, 0)), mode='constant', constant_values=0)
      x = np.append(x, row[:,None,:], axis=1)
      # Prepare y with shape (trials, 1, 2)
      ychoice = choice[selection].astype(int)
      ychoice = ychoice[1:] # remove first trial because it has no feedback
      #yconf = conf[selection]
      #yconf = yconf[1:] # remove first trial because it has no feedback
      #row = np.stack([ychoice, yconf], axis=1)
      if ychoice.shape[0] < 119:
        missing = 119 - ychoice.shape[0]
        #print(f"block {b} session {s}: {missing} trials missing, padding with zeros")
        ychoice = np.pad(ychoice, ((0, missing)),
                       mode='constant', constant_values=-1)
      y = np.append(y, ychoice[:,None,None], axis=1)
      # Prepare repeat with shape (trials, 1)
      repeat = np.append(repeat, trans_prob[selection][0] != 0.5)
      #print(f"  trials: {Xmat[selection,28]}")

N_trials = x.shape[1]
print(f"Total number of trials: {N_trials}")
x_names = ['delta_contrast', 'feedback']
y_names = ['choice']

#with open("history/datasets_choice_all_trials.pkl", "rb") as f:
#  old_dataset = pickle.load(f)
#  ds_train = pickle.load(f)
#  ds_valid = pickle.load(f)

# select the validation block
dataset = rnn_utils.DatasetRNNCategorical(x, y,
                               x_names=x_names, y_names=y_names,
                               batch_mode = 'single')
ds_train, ds_valid = rnn_utils.split_dataset(dataset, eval_every_n=4)

if args.result_path:
  print(f"Saving dataset to {args.result_path}")
  os.makedirs(args.result_path, exist_ok=True)
  with open(os.path.join(args.result_path, "datasets.pkl"), "wb") as f:
    pickle.dump(dataset, f)
    pickle.dump(ds_train, f)
    pickle.dump(ds_valid, f)
    pickle.dump(repeat, f)

#with open("history/params_choice_all_trials.pkl", "rb") as f:
#  config = pickle.load(f)
#  #params = pickle.load(f)

config = disrnn.DisRnnConfig(
  # Dataset related
  obs_size=2,  # delta mean contrast, previous feedback
  output_size=2,  # choice
  x_names=ds_train.x_names,
  y_names=ds_train.y_names,
  # Network architecture
  latent_size=5,
  update_net_n_units_per_layer=16,
  update_net_n_layers=4,
  choice_net_n_units_per_layer=4,
  choice_net_n_layers=2,
  activation='leaky_relu',
  # Penalties
  noiseless_mode=False,
  latent_penalty=args.beta,
  choice_net_latent_penalty=args.beta,
  update_net_obs_penalty=args.beta,
  update_net_latent_penalty=args.beta,
  )

# Initial training in noiseless mode
params, opt_state, losses = rnn_utils.train_network(
  lambda: disrnn.HkDisentangledRNN(config),
  ds_train,
  ds_valid,
  opt = optax.adam(5e-4),
  loss="penalized_categorical",
  #loss_param={'likelihood_weight': 1.0, 'penalty_scale': 1.0},
  n_steps=0)

params, opt_state, losses = rnn_utils.train_network(
  lambda: disrnn.HkDisentangledRNN(config),
  ds_train,
  ds_valid,
  loss="penalized_categorical",
  params=params,
  opt_state=opt_state,
  opt = optax.adam(1e-3),
  loss_param = 1,
  #loss_param = {'likelihood_weight': 1.0, 'penalty_scale': 1.0},
  n_steps=args.train_steps,
  do_plot = args.learning_curve,
  log_losses_every=100)

choice_sigmas = params['hk_disentangled_rnn']['choice_net_sigma_params']
print(f"choice bn sigmas:{choice_sigmas}")

if args.result_path:
  with open(os.path.join(args.result_path, "params.pkl"), "wb") as f:
    pickle.dump(config, f)
    pickle.dump(params, f)

if args.learning_curve:
  plotting.plot_learning_curves(losses)
  plotting.plot_bottlenecks(params, config, sort_latents=False)
  plotting.plot_choice_rule(params, config)
  #plotting.plot_update_rules(params, config)
  plt.show()

