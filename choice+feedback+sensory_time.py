# Trains disRNN on inputs previous choice, feedback and 
# sensory input (time resolved) to predict next choice.
import argparse
import copy
from enum import unique
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
parser.add_argument('--sensory-noise', type=float, default=None, help="sensory noise level")
parser.add_argument('--train-steps', type=int, default=500_000, help="training steps")
parser.add_argument('--no-plot', action='store_true', help="hide result plots")
parser.add_argument('--test-trial', type=int, default=None, help="evaluate accuracy on this trial")
parser.add_argument('--drug-id', type=int, default=2, help="0 all, 1 Lorazepam, 2 placebo, 3 Memantine")

args = parser.parse_args()
print(f"Training with beta={args.beta}, train_steps={args.train_steps}, "
      f"sensory-noise={args.sensory_noise}, "
      f"test_trial={args.test_trial}, no_plot={args.no_plot}, drug_id={args.drug_id}")

# Load data and drug information
mat = scipy.io.loadmat('data/data2alex.mat')
Xmat = mat['All_Data_s']
nan_mask = ~np.isnan(Xmat[:,0])
Xmat = Xmat[nan_mask]

mat = scipy.io.loadmat('data/Drug_id.mat')
drug = mat['All_Drug']
drug = drug[nan_mask]
if args.drug_id != 0: # only consider sessions for the specified drug
  Xmat = Xmat[drug[:,0] == args.drug_id,:] 
  drug = drug[drug[:,0] == args.drug_id,:].astype(int)

# considers a whole block as episodes of a single trial
# input: delta of the generative mean of the current trial and choice and feedback
# from the previous trial
stim_right = Xmat[:, 1:11]   # Right stimulus, 10 timesteps per trial
stim_left = Xmat[:, 11:21]  # Left stimulus, 10 timesteps per trial
delta_mean = Xmat[:,22]
feedback = Xmat[:, 27]
choice = Xmat[:, 0]
#conf = Xmat[:, 21]
trans_prob = Xmat[:, 26]
#sessions = np.unique(Xmat[:, 24]).astype(int)
#blocks = np.unique(Xmat[:, 25]).astype(int)

timesteps = 10
datasets = []
# 11 time steps, 119 trials -> 1309
x = np.empty((1309,0,4)) # input: previous choice, feedback, stim_left, stim_right
#x_rep = np.empty((119,0,2))
y = np.empty((1309,0,1), dtype=int)
repeat = np.empty(0, dtype=bool)
drug_id = np.empty(0, dtype=int)

for sub in range(1,21):
  for sess in range(1,7):
    for blk in range(1,8):
      #print(f"subject {sub}, session {sess}, block {blk}")
      selection = (Xmat[:,23] == sub) & (Xmat[:, 25] == sess) & (Xmat[:, 24] == blk)
      if np.sum(selection) < 2: # need at least 2 trials to have a history
        #print(f"No trials in subject {sub}, block {blk}, session {sess}.")
        continue
      if np.sum(selection) > 120: 
        print(f"Too many trials ({np.sum(selection)}) for subject {sub} in block {blk} session {sess}, skipping")
        continue
      
      #if trans_prob[selection][0] == 0.5: # only consider repetitive trials
      #  continue
      #print(np.where(selection)[0][0])
      
      # Prepare x
      xstim_left = stim_left[selection]
      xstim_left = xstim_left[1:,:] # remove first trial because it has no feedback
      if args.sensory_noise is not None:
        xstim_left = xstim_left + np.random.normal(0, args.sensory_noise, size=xstim_left.shape)
      xstim_left = np.append(xstim_left, np.zeros((xstim_left.shape[0],1)), axis=1) # add a zero to the end of each trial to indicate the end of the trial
      xstim_right = stim_right[selection]
      xstim_right = xstim_right[1:,:] # remove first trial because it has no feedback
      if args.sensory_noise is not None:
        xstim_right = xstim_right + np.random.normal(0, args.sensory_noise, size=xstim_right.shape)
      xstim_right = np.append(xstim_right, np.zeros((xstim_right.shape[0],1)), axis=1) # add a zero to the end of each trial to indicate the end of the trial
      xdelta_mean = delta_mean[selection]
      xdelta_mean = xdelta_mean[1:] # remove first trial because it has no feedback
      #if args.sensory_noise is not None:
      #  scrambled = np.random.permutation(xdelta_mean)
      #  for i in range(len(xdelta_mean)):
      #    if np.random.rand() < args.sensory_noise:
      #      xdelta_mean[i] = scrambled[i]
      
      xfeedback = feedback[selection]
      xfeedback = xfeedback[:-1] # remove last trial because it has no next trial to predict
      xfeedback_time = np.tile(xfeedback, (timesteps,1)).T
      xfeedback_time = np.append(xfeedback_time, np.zeros((xfeedback_time.shape[0],1)), axis=1) # add a zero to the end of each trial to indicate the end of the trial
      xchoice = choice[selection].astype(int)
      xchoice = xchoice[:-1] # remove last trial because it has no next trial to predict
      xchoice_time = np.tile(xchoice, (timesteps,1)).T
      xchoice_time = np.append(xchoice_time, np.zeros((xchoice_time.shape[0],1)), axis=1) # add a zero to the end of each trial to indicate the end of the trial
      row = np.stack([xchoice_time.flatten(), xfeedback_time.flatten(), 
                      xstim_left.flatten(), xstim_right.flatten()], axis=1)
      if row.shape[0] < 1309:   
        missing = 1309 - row.shape[0]
        #print(f"Subject {sub}, session {sess}, block {blk}: {missing} trial(s) missing, padding with zeros")
        row = np.pad(row, ((0, missing), (0, 0)), mode='constant', constant_values=-1)

      x = np.append(x, row[:,None,:], axis=1)

      # if trans_prob[selection][0] == 0.5: # only consider repetitive trials
      #   x = np.append(x, row[:,None,:], axis=1)
      # else:
      #   x_rep = np.append(x_rep, row[:,None,:], axis=1)
      
      # Train to predict subject's choice
      ychoice = choice[selection].astype(int)
      ychoice = ychoice[1:] # remove first trial because it has no feedback

      # Train on task
      #ychoice = np.array(xdelta_mean>0).astype(int) # 1 if right, 0 if left
      #ychoice[xdelta_mean==0] = -1
      
      ychoice_time = np.full_like(xchoice_time, -1)
      ychoice_time[:,-1] = ychoice  

      #yconf = conf[selection]
      #yconf = yconf[1:] # remove first trial because it has no feedback
      #row = np.stack([ychoice, yconf], axis=1)
      if ychoice_time.shape[0] < 119:
        missing = 119 - ychoice_time.shape[0]
        #print(f"block {b} session {s}: {missing} trials missing, padding with zeros")
        ychoice_time = np.pad(ychoice_time, ((0, missing),(0,0)),
                       mode='constant', constant_values=-1)
      y = np.append(y, ychoice_time.flatten()[:,None,None], axis=1)
      
      # Prepare repeat with shape (trials, 1)
      repeat = np.append(repeat, trans_prob[selection][0] != 0.5)
      drug_id = np.append(drug_id, np.unique(drug[selection])[0])
      #print(f"  trials: {Xmat[selection,28]}")

N_trials = x.shape[1]
print(f"Total number of trials: {N_trials}")

if args.test_trial is not None:
  if args.test_trial < 0 or args.test_trial >= N_trials:
    raise ValueError(f"test_trial must be between 0 and {N_trials-1}, but got {args.test_trial}")
  # Extract test trial
  x_test = x[:, args.test_trial:args.test_trial+1, :]
  y_test = y[:, args.test_trial:args.test_trial+1, :]
  # Remove test trial from training data
  x = np.delete(x, args.test_trial, axis=1)
  y = np.delete(y, args.test_trial, axis=1)

x_names = ['choice', 'feedback', 'stim_left', 'stim_right']
y_names = ['choice']

dataset = rnn_utils.DatasetRNNMixed(x, y,
                               x_names=x_names, y_names=y_names,
                               batch_mode = 'single')
ds_train, ds_valid = rnn_utils.split_dataset(dataset, eval_every_n=4)

if args.test_trial is not None:
  test_dataset = rnn_utils.DatasetRNNMixed(x_test, y_test,
                               x_names=x_names, y_names=y_names,
                               batch_mode = 'single')

if args.result_path:
  print(f"Saving dataset to {args.result_path}")
  os.makedirs(args.result_path, exist_ok=True)
  with open(os.path.join(args.result_path, "datasets.pkl"), "wb") as f:
    pickle.dump(dataset, f)
    pickle.dump(ds_train, f)
    pickle.dump(ds_valid, f)
    pickle.dump(repeat, f)
    pickle.dump(drug_id, f)

config = disrnn.DisRnnConfig(
  # Dataset related
  obs_size=4,  # choice, feedback, stim_left, stim_right
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
  l2_scale=1e-2, # default is 1e-2, which is a pretty strong L2 penalty. You can set this to a smaller value if you want to have less regularization on the weights. Setting it to 0 will turn off L2 regularization entirely.
  # loss_param = 1, # default is 1, which means the loss is (likelihood + penalty). If you set this to a value other than 1, it will scale the whole loss function by that value. So for example if you set it to 0.5, the loss will be 0.5*likelihood + 0.5*penalty. This can be useful if you want to have more control over the relative weighting of the likelihood and penalty terms in the loss function.
  )

config_noiseless = copy.deepcopy(config)
config_noiseless.noiseless_mode = True
config_noiseless.latent_penalty = 0
config_noiseless.choice_net_latent_penalty = 0
config_noiseless.update_net_obs_penalty = 0
config_noiseless.update_net_latent_penalty = 0
config_noiseless.l2_scale = 0

# Initial training in noiseless mode
params, opt_state, losses = rnn_utils.train_network(
  lambda: disrnn.HkDisentangledRNN(config_noiseless),
  ds_train,
  ds_valid,
  opt = optax.adam(1e-3),
  loss="penalized_categorical",
  log_losses_every=100,
  n_steps=500)

opt_state = None # Reset the optimizer state. In the next cell we'll add the bottleneck penalties, which will change the loss function quite a bit.

params, opt_state, losses = rnn_utils.train_network(
  lambda: disrnn.HkDisentangledRNN(config),
  ds_train,
  ds_valid,
  loss="penalized_categorical",
  params=params,
  opt_state=opt_state,
  opt = optax.adam(1e-3),
  # loss_param = 1, default is 1, which means the loss is (likelihood + penalty). If you set this to a value other than 1, it will scale the whole loss function by that value. So for example if you set it to 0.5, the loss will be 0.5*likelihood + 0.5*penalty. This can be useful if you want to have more control over the relative weighting of the likelihood and penalty terms in the loss function.
  # loss_param = {'likelihood_weight': 1.0, 'penalty_scale': 1.0},
  n_steps=args.train_steps,
  do_plot = not args.no_plot,
  log_losses_every=100)

choice_sigmas = params['hk_disentangled_rnn']['choice_net_sigma_params']
print(f"choice bn sigmas:{choice_sigmas}")

if args.result_path:
  with open(os.path.join(args.result_path, "params.pkl"), "wb") as f:
    pickle.dump(config, f)
    pickle.dump(params, f)

if not args.no_plot:
  plotting.plot_bottlenecks(params, config, sort_latents=False)
  plotting.plot_choice_rule(params, config)
  #plotting.plot_update_rules(params, config)
  plt.show()

