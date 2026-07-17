# trains a disRNN to predict the next choice based on the previous choice and feedback,
# and evaluates the accuracy of the network's predictions on a held-out test trial. 
#
import argparse
import copy
import fcntl
from filelock import FileLock
from math import pi
import pickle
import jax
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
parser.add_argument('--result-path', type=str, default=None, help="Path to save results (params, configs).")
parser.add_argument('--beta', type=float, default=1e-3, help="general penalty")
parser.add_argument('--train-steps', type=int, default=500_000, help="training steps")
parser.add_argument('--no-plot', action='store_true', help="hide result plots")
parser.add_argument('--test-trial', type=int, default=None, help="evaluate accuracy on this trial")

args = parser.parse_args()
print(f"Training with beta={args.beta}, train_steps={args.train_steps}, no_plot={args.no_plot}, test_trial={args.test_trial}")

# input delta of the generative mean and feedback from previous trial
mat = scipy.io.loadmat('data/data2alex.mat')
Xmat = mat['All_Data_s']
mat = scipy.io.loadmat('data/Drug_id.mat')
drug = mat['All_Drug']
Xmat = Xmat[drug[:,0] == 2,:] # only consider placebo sessions
nan_mask = ~np.isnan(Xmat[:,0])
Xmat = Xmat[nan_mask]

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
#x_rep = np.empty((119,0,2))
y = np.empty((119,0,1), dtype=int)
repeat = np.empty(0, dtype=bool)

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
      # Prepare x with shape (trials, 1, 2)
      xdelta_mean = delta_mean[selection]
      xdelta_mean = xdelta_mean[1:] # remove first trial because it has no feedback
      xfeedback = feedback[selection]
      xfeedback = xfeedback[:-1] # remove last trial because it has no next trial to predict
      xchoice = choice[selection].astype(int)
      xchoice = xchoice[:-1] # remove last trial because it has no next trial to predict
      #row = np.stack([xdelta_mean, xfeedback], axis=1)
      row = np.stack([xchoice, xfeedback], axis=1)
      #row = xfeedback[:,None] # only use feedback as input
      if row.shape[0] < 119:
        missing = 119 - row.shape[0]
        #print(f"Subject {sub}, session {sess}, block {blk}: {missing} trial(s) missing, padding with zeros")
        row = np.pad(row, ((0, missing), (0, 0)), mode='constant', constant_values=0)
      x = np.append(x, row[:,None,:], axis=1)
      # if trans_prob[selection][0] == 0.5: # only consider repetitive trials
      #   x = np.append(x, row[:,None,:], axis=1)
      # else:
      #   x_rep = np.append(x_rep, row[:,None,:], axis=1)
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

# Verify transition probabilities
# lr = np.sign(x[:,:,0])
# lr_rep = np.sign(x_rep[:,:,0])
# lrdiff = np.diff(lr, axis=0)
# lrdiff_rep = np.diff(lr_rep, axis=0)
# trans_prob = np.mean(lrdiff != 0)
# trans_prob_rep = np.mean(lrdiff_rep != 0)
# print(f"Transition probability in all trials: {trans_prob:.3f}")
# print(f"Transition probability in repetitive trials: {trans_prob_rep:.3f}")
# quit()

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

#x_names = ['delta_contrast', 'feedback']
x_names = ['choice', 'feedback']
#x_names = ['feedback']
y_names = ['next choice']

#with open("history/datasets_choice_all_trials.pkl", "rb") as f:
#  old_dataset = pickle.load(f)
#  ds_train = pickle.load(f)
#  ds_valid = pickle.load(f)

# select the validation block

dataset = rnn_utils.DatasetRNNCategorical(x, y,
                               x_names=x_names, y_names=y_names,
                               batch_mode = 'single')
ds_train, ds_valid = rnn_utils.split_dataset_randomly(dataset, eval_fraction=0.2, 
                                                      rng=np.random.default_rng())

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
  obs_size=2, 
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
  random_key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), "big")),
  opt = optax.adam(1e-2),
  loss="penalized_categorical",
  log_losses_every=100,
  n_steps=1000)

opt_state = None # Reset the optimizer state. In the next cell we'll add the bottleneck penalties, which will change the loss function quite a bit.

params, opt_state, losses = rnn_utils.train_network(
  lambda: disrnn.HkDisentangledRNN(config),
  ds_train,
  ds_valid,
  random_key = jax.random.PRNGKey(int.from_bytes(os.urandom(4), "big")),
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
latent_sigmas = params['hk_disentangled_rnn']['latent_sigma_params']
print(f"choice bn sigmas:{choice_sigmas}")
print(f"latent bn sigmas:{latent_sigmas}")
#nbottlenecks = np.sum(choice_sigmas<0.1)

if args.result_path:
  with open(os.path.join(args.result_path, "params.pkl"), "wb") as f:
    pickle.dump(config, f)
    pickle.dump(params, f)

if not args.no_plot:
  plotting.plot_bottlenecks(params, config, sort_latents=False)
  plotting.plot_choice_rule(params, config)
  #plotting.plot_update_rules(params, config)
  plt.show(block=False)

if args.test_trial is not None:
  # Evaluate accuracy on test trial
  make_disrnn_warmup = lambda: disrnn.HkDisentangledRNN(config_noiseless)
  model_output, _ = rnn_utils.eval_network(
    make_disrnn_warmup, params, x_test)
  output_logits = model_output[:, :, :-1] # discard penalties
  log_probs = jax.nn.log_softmax(output_logits, axis=2)
  choice_prediction = jax.numpy.argmax(log_probs,axis=-1)
  
  next_choice = y_test.flatten()
  valid_mask = next_choice >= 0
  next_choice = next_choice[valid_mask]
  network_prediction = np.array(choice_prediction.flatten()[valid_mask])
  acc = np.mean(next_choice==network_prediction)
  #if nbottlenecks == 2:
  print(f"Network accuracy: {acc:.3f}")
  #else:
  #  print(f"bottlenecks: {nbottlenecks}")
  
  # with open("results.txt", "a") as file:
  #   fcntl.flock(file.fileno(), fcntl.LOCK_EX)
  #   file.write(str(args.test_trial) + " " + str(acc) + " " +
  #              str(latent_sigmas) + " "+ str(choice_sigmas) + "\n")
  #   #if nbottlenecks == 2:
  #   #  file.write(str(args.test_trial) + " " + str(acc) + "\n")
  #   #else:
  #   #  file.write(str(args.test_trial) + " " + str(nbottlenecks) + "\n")
  #   fcntl.flock(file.fileno(), fcntl.LOCK_UN)

  lock = FileLock("results.lock")
  with lock:
    with open("results.txt", "a") as f:
      f.write(str(args.test_trial) + " " + str(acc) + " " +
                 str(latent_sigmas) + " "+ str(choice_sigmas) + "\n")
