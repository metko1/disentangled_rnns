import optax
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from disentangled_rnns.library import rnn_utils, disrnn, plotting

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

# for sbj1.mat
# nan_mask = ~np.isnan(Xmat[:,20])
# Xmat = Xmat[nan_mask]
# choice = Xmat[:, 20]
# stim_right = Xmat[:, :10]
# stim_left  = Xmat[:, 10:20]

[N_trials, timesteps] = stim_right.shape

# Concatenate right and left features: (N_trials, timesteps, 2)
x_concat = np.stack([stim_right, stim_left], axis=2)
# Transpose to (timesteps, trials, features)
x_reshaped = np.transpose(x_concat, (1,0,2))

# Concatenate right and left features and choice from previous trial: (N_trials, timesteps, 2)
#c = np.tile(choice[:,np.newaxis], (1,timesteps))
#x_concat = np.stack([stim_right[1:,:], stim_left[1:,:], c[0:-1,:]])


# Prepare y with shape (timesteps, trials, 2)
y = np.full((timesteps, N_trials,1), -1)
y[-1,:,0] = choice
# y[-1,:,1] = conf

#y = np.tile(choice,[timesteps,1])[:,:,None]

x_names = ['right', 'left']
y_names = ['choice']

dataset = rnn_utils.DatasetRNN(x_reshaped, y, y_type='categorical',
                                  x_names=x_names, y_names=y_names,
                                  batch_size=16, batch_mode='single')

dataset_train, dataset_eval = rnn_utils.split_dataset(dataset, 2)

disrnn_config = disrnn.DisRnnConfig(
      # Dataset related
      obs_size=2,  # left, right
      output_size=2,  # choose
      x_names=dataset.x_names,
      y_names=dataset.y_names,
      # Network architecture
      latent_size=5,
      update_net_n_units_per_layer=16,
      update_net_n_layers=4,
      choice_net_n_units_per_layer=4,
      choice_net_n_layers=2,
      activation='leaky_relu',
      # Penalties
      noiseless_mode=False,
      latent_penalty=1e-2,
      choice_net_latent_penalty=1e-2,
      update_net_obs_penalty=1e-2,
      update_net_latent_penalty=1e-2
)

# Initial training in noiseless mode

params, opt_state, losses = rnn_utils.train_network(
   lambda: disrnn.HkDisentangledRNN(disrnn_config),
    dataset_train,
    dataset_eval,
    opt = optax.adam(1e-3),
    loss="penalized_categorical",
    #loss_param={'likelihood_weight': 1.0, 'penalty_scale': 1.0},
    n_steps=0)

# Usually 20,000 steps in total should be sufficient.
#n_steps = 20_000
n_steps = 3_000

params, opt_state, losses = rnn_utils.train_network(
    lambda: disrnn.HkDisentangledRNN(disrnn_config),
    dataset_train,
    dataset_eval,
    loss="penalized_categorical",
    params=params,
    opt_state=opt_state,
    opt = optax.adam(1e-3),
    loss_param = 1,
    n_steps=n_steps,
    do_plot = True)

plotting.plot_bottlenecks(params, disrnn_config, sort_latents=False)
plotting.plot_choice_rule(params, disrnn_config)
plotting.plot_update_rules(params, disrnn_config)
plt.show()
