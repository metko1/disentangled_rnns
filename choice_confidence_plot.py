import jax
import jax.numpy as jnp
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from disentangled_rnns.library import rnn_utils, two_output_disrnn

with open("params.pkl", "rb") as f:
  config = pickle.load(f)
  params = pickle.load(f)
with open("datasets.pkl", "rb") as f:
  dataset = pickle.load(f)
  dataset_train = pickle.load(f)
  dataset_eval = pickle.load(f)

#two_output_disrnn.plot_bottlenecks(params, config, sort_latents=False)
#two_output_disrnn.plot_choice_rule(params, config)
#two_output_disrnn.plot_confidence_rule(params, config)
#two_output_disrnn.plot_update_rules(params, disrnn_w_two_output_config)
#plt.show()

#----------------------------------
# Plot human confidence kernels
#----------------------------------
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
delta_gen = Xmat[:, 22]
timesteps = 10

# Compute generative means per trial:
# mu_R = 0.5 + delta/2 ; mu_L = 0.5 - delta/2
mu_R = 0.5 + (delta_gen.astype(float) / 2.0)
mu_L = 0.5 - (delta_gen.astype(float) / 2.0)
# Expand to (trials, timesteps, 1) to allow subtracting
mu_R_tiled = np.tile(mu_R,[timesteps,1]).transpose()
mu_L_tiled = np.tile(mu_L,[timesteps,1]).transpose()
# Centered stimuli for kernel computation (NO multiply by 2 here per your request)
right_centered_for_kernel = stim_right - mu_R_tiled  # (trials, timesteps, 1)
left_centered_for_kernel = stim_left - mu_L_tiled

# Averaging-based kernels split by selected/non-selected × high/low confidence
# For these we use the centered-by-generative-mean stimuli (no *2), and compute selected vs non-selected
# Human kernels using human choice/conf
selected_data = np.where(choice[:, None] == 1, right_centered_for_kernel, left_centered_for_kernel)
nonselected_data = np.where(choice[:, None] == 1, left_centered_for_kernel, right_centered_for_kernel)
high_mask = conf == 1
low_mask  = conf == 0
kernel_sel_high_data = np.mean(selected_data[high_mask],axis=0)
kernel_sel_low_data  = np.mean(selected_data[low_mask], axis=0)
kernel_nsel_high_data = np.mean(nonselected_data[high_mask], axis=0)
kernel_nsel_low_data  = np.mean(nonselected_data[low_mask], axis=0)

t = np.arange(timesteps)+1

plt.subplot(2,1,1)
plt.plot(t, kernel_sel_high_data, label='Selected - High (Human)', lw=3)
plt.plot(t, kernel_sel_low_data,  label='Selected - Low  (Human)', lw=3)
plt.plot(t, kernel_nsel_high_data, label='NonSel - High (Human)', lw=2, linestyle='--')
plt.plot(t, kernel_nsel_low_data,  label='NonSel - Low  (Human)', lw=2, linestyle='--')
plt.title("Human kernels (selected vs non-selected) \nby reported conf — centered by generative mean")
plt.xlabel("Sample")
plt.ylabel("Avg centered contrast")
plt.legend(); plt.grid(True)

#----------------------------------
# Plot RNN model confidence kernels
#----------------------------------
# Propagate data through the trained network
x_s, y_s = dataset.get_all()
model_output, opt_state = rnn_utils.eval_network(
    lambda: two_output_disrnn.HkTwoOutputDisentangledRNN(config),
    params, x_s)
output_logits = model_output[:, :, :-1] # discard penalties
shape = output_logits.shape
# reshape output into pairs of binaries
reshaped_output_logits = jnp.reshape(output_logits, (shape[0], shape[1], 2, 2))
log_probs = jax.nn.log_softmax(reshaped_output_logits, axis=3)
choice = jnp.argmax(log_probs[-1,:,:,0],axis=-1)
conf = jnp.argmax(log_probs[-1,:,:,1],axis=-1)

# Averaging-based kernels split by selected/non-selected × high/low confidence
# For these we use the centered-by-generative-mean stimuli (no *2), and compute selected vs non-selected
# Human kernels using human choice/conf
selected_data = np.where(choice[:, None] == 1, right_centered_for_kernel, left_centered_for_kernel)
nonselected_data = np.where(choice[:, None] == 1, left_centered_for_kernel, right_centered_for_kernel)
high_mask = conf == 1
low_mask  = conf == 0
kernel_sel_high_data = np.mean(selected_data[high_mask],axis=0)
kernel_sel_low_data  = np.mean(selected_data[low_mask], axis=0)
kernel_nsel_high_data = np.mean(nonselected_data[high_mask], axis=0)
kernel_nsel_low_data  = np.mean(nonselected_data[low_mask], axis=0)

plt.subplot(2,1,2)
plt.plot(t, kernel_sel_high_data, label='Selected - High (model)', lw=3)
plt.plot(t, kernel_sel_low_data,  label='Selected - Low  (model)', lw=3)
plt.plot(t, kernel_nsel_high_data, label='NonSel - High (model)', lw=2, linestyle='--')
plt.plot(t, kernel_nsel_low_data,  label='NonSel - Low  (model)', lw=2, linestyle='--')
plt.title("disRNN model kernels (selected vs non-selected) \nby reported conf — centered by generative mean")
plt.xlabel("Sample")
plt.ylabel("Avg centered contrast")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()



