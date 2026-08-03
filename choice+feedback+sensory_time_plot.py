# Plot bottlenecks and time courses of latent variables and outputs from 
# choice+feedback+sensory script
import copy
from haiku.pad import valid
import jax
import jax.numpy as jnp
from matplotlib import axis
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from disentangled_rnns.library import rnn_utils, disrnn, plotting
from disentangled_rnns.library import two_armed_bandits

#datapath = "choice+feedback+sensory/beta=1e-3_steps=2e4_noise=0.6"
datapath = "choice+feedback+sensory_time/task_beta=1e-3_steps=2e5"
os.makedirs(datapath, exist_ok=True)  # Create directory if it doesn't exist
with open(datapath+"/params.pkl", "rb") as f:
  config = pickle.load(f)
  params = pickle.load(f)
with open(datapath+"/datasets.pkl", "rb") as f:
  dataset = pickle.load(f)
  dataset_train = pickle.load(f)
  dataset_eval = pickle.load(f)
  repeat = pickle.load(f)
  #drug = pickle.load(f)
drug = np.full(repeat.shape, fill_value=2) # dummy drug array since drug info is not available for all trials
drug_name = ["all drugs", "lorazepam", "placebo", "memantine"]

#plotting.plot_bottlenecks(params, config, sort_latents=False)
#plotting.plot_choice_rule(params, config)
#plotting.plot_update_rules(params, config, observation_types = [[0,0,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,1]]) # choice, feedback, stim_left, stim_right 
#plt.show(block=True)

latent_sigmas = disrnn.reparameterize_sigma(
   params['hk_disentangled_rnn']['latent_sigma_params'])
print(f"latent sigmas:{latent_sigmas}")
choice_sigmas = disrnn.reparameterize_sigma(
   params['hk_disentangled_rnn']['choice_net_sigma_params'])
print(f"choice sigmas:{choice_sigmas}")
update_sigmas = disrnn.reparameterize_sigma(
   params['hk_disentangled_rnn']['update_net_latent_sigma_params'])
print(f"update sigmas: {update_sigmas}")

# Propagate data through the trained network
disrnn_config_warmup = copy.deepcopy(config)
disrnn_config_warmup.latent_penalty = 0
disrnn_config_warmup.choice_net_latent_penalty = 0
disrnn_config_warmup.update_net_obs_penalty = 0
disrnn_config_warmup.update_net_latent_penalty = 0
disrnn_config_warmup.l2_scale = 0
disrnn_config_warmup.noiseless_mode = True
make_disrnn_warmup = lambda: disrnn.HkDisentangledRNN(disrnn_config_warmup)

xs = dataset.get_all()['xs']
ys = dataset.get_all()['ys']
model_output, opt_state = rnn_utils.eval_network(
    make_disrnn_warmup, params, xs)

last_choice = xs[:, :, 0] # previous choice
last_feedback = xs[:, :, 1] # previous feedback
stim_left = xs[:, :, 2] # current sensory input
stim_right = xs[:, :, 3] # current sensory input left
choice = np.squeeze(ys)
valid_mask = choice >= 0

output_logits = model_output[:, :, :-1] # discard penalties
log_probs = jax.nn.log_softmax(output_logits, axis=2)
model_choice = jnp.argmax(log_probs,axis=-1)
network_prediction = np.array(model_choice[valid_mask])

# Evaluate network performance
correct = model_choice[valid_mask] == choice[valid_mask]
network_acc = np.mean(correct.flatten())
print(f"Network accuracy: {network_acc:.3f}")
#model_acc = np.mean(network_prediction == subj_choice)
#print(f"Model accuracy: {model_acc:.3f}")

# Reduce last_choice and last_feedback to one input at the beginning of a trial
# to better visualize trial limits
vis_last_choice = np.reshape(last_choice, shape=(11, -1, 241), order='F')
vis_last_choice[1:, :, :] = -1
vis_last_choice = np.reshape(vis_last_choice, shape=(-1,241), order='F')
vis_last_feedback = np.reshape(last_feedback, shape=(11, -1, 241), order='F')
vis_last_feedback[1:, :, :] = -1
vis_last_feedback = np.reshape(vis_last_feedback, shape=(-1, 241), order='F')

for drug_id in np.unique(drug):
   drug_mask = drug.flatten() == drug_id
   non_rep_trials = np.where(~repeat & drug_mask)[0] # take the first non-repetitive session as an example 
   if non_rep_trials.size > 0:
      for i in range(3):
         example_session = non_rep_trials[i]
         last_choice_session = vis_last_choice[:, example_session]
         last_feedback_session = vis_last_feedback[:, example_session]
         model_choice_session = model_choice[:, example_session,np.newaxis] - 0.5
         #correct_session = correct[:, example_session]
         states = np.squeeze(opt_state[:, example_session, :])
         #input_left = 5 * stim_left[:, example_session, np.newaxis]
         #input_right = 5 * stim_right[:, example_session, np.newaxis]
         #scalars = np.concat((states, input_left, input_right, 
         #                     model_choice_session), axis=1)
         scalars = np.concat((states,model_choice_session),axis=1)
         two_armed_bandits.plot_2ab_sessdata(last_choice_session,
                                          last_feedback_session,
                                          scalars=scalars,
                                          scalar_types='agent_states',
                                          show_legend=True)
         plt.suptitle(drug_name[drug_id] + ", non-repetitive block, trial "+str(example_session))
         plt.savefig(os.path.join(datapath, f"nonrep_trial{i}.png"), dpi=150, bbox_inches='tight')
#plt.show(block=True)

   rep_trials = np.where(repeat & drug_mask)[0] # take the first repetitive session as an example
   if rep_trials.size > 0:
      for i in range(3):
         example_session = rep_trials[i]
         last_choice_session = last_choice[:, example_session]
         last_feedback_session = last_feedback[:, example_session]
         model_choice_session = model_choice[:, example_session,np.newaxis] - 0.5
         #correct_session = correct[:, example_session]
         states = np.squeeze(opt_state[:, example_session, :])
         input_left = 5 * stim_left[:, example_session, np.newaxis]
         input_right = 5 * stim_right[:, example_session, np.newaxis]
         #scalars = np.concat((states, input_left, input_right, model_choice_session), axis=1)
         scalars = np.concat((states,model_choice_session),axis=1)
         two_armed_bandits.plot_2ab_sessdata(last_choice_session,
                                          last_feedback_session,
                                          scalars=scalars,
                                          scalar_types='agent_states',
                                          show_legend=True)
         plt.suptitle(drug_name[drug_id] + ", repetitive block, trial "+str(example_session))
         plt.savefig(os.path.join(datapath, f"rep_trial{i}.png"), dpi=150, bbox_inches='tight')
plt.show(block=True)
# quit()  # Commented out to allow remaining plots to be generated

# plot latent manifold
x = xs[:, :, 0].flatten() # delta contrast
f = xs[:, :, 1].flatten() # feedback
y = ys.flatten() # choice
delta_contrast = xs[:, :, 0].flatten()
latent_1 = opt_state[:,:,2].flatten()
latent_2 = opt_state[:,:,3].flatten()
plt.figure()
#sns.scatterplot(x=latent_1, y=latent_2, hue=delta_contrast, palette='Set1', alpha=0.7)
indices = np.argsort(np.abs(delta_contrast))
ax = sns.scatterplot(x=latent_1[indices], y=latent_2[indices], hue=delta_contrast[indices],
                palette='RdYlBu', edgecolor='none', alpha=0.7)
plt.xlabel("Latent 3")
plt.ylabel("Latent 4")
plt.title("Latent manifold colored by choice")
#plt.legend(title="Choice")
norm=plt.Normalize(delta_contrast.min(),delta_contrast.max())
sm=plt.cm.ScalarMappable(cmap="RdYlBu",norm=norm)
sm.set_array([])
ax.get_legend().remove()
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Delta contrast")
plt.grid(True)
plt.savefig(os.path.join(datapath, "latent_manifold.png"), dpi=150, bbox_inches='tight')
#plt.show(block=True)

# plt.figure()
# r = np.repeat(repeat, xs.shape[0])
# r = r.flatten()
# sns.scatterplot(x=latent_1, y=latent_2, hue=r, palette='Set1', alpha=0.7)
# plt.xlabel("Latent 1")
# plt.ylabel("Latent 2")
# plt.title("Latent manifold colored by repeat")
# plt.legend(title="Repeat")
# plt.grid(True)
# plt.show(block=True)

# plot time trace of the latent variables
fig, axs = plt.subplots(3,1, figsize=(8,6))
for i, line in enumerate(np.transpose(opt_state[0,:,:])):
    axs[0].plot(line, label=f'Latent {i+1}')  # Label each line with its number
axs[0].set_xlabel("Trial")
axs[0].set_ylabel("Activity")
axs[0].legend()
#plt.show(block=False)

# plot outputs
#axs[1].plot(choice[0,:], label='choice')
#axs[1].plot(5*xs[0, :, 0], label='delta contrast')
#axs[1].set_xlabel("Trial")
#axs[1].set_ylabel("Output logits")
#axs[1].legend()

axs[1].plot(opt_state[0,:,0], label='latent 1')
axs[1].plot(5*xs[0, :, 0], label='delta contrast')
axs[1].set_xlabel("Trial")
axs[1].set_ylabel("Activity")
axs[1].grid(True)
axs[1].legend()

feedback = xs[:, :, 1]
axs[2].plot(opt_state[0,:,2], label='latent 3')
axs[2].plot(feedback[0,:], label='feedback')
axs[2].set_xlabel("Trial")
axs[2].set_ylabel("Activity")
axs[2].grid(True)
axs[2].legend()
plt.grid(True)
plt.savefig(os.path.join(datapath, "timecourse_latents.png"), dpi=150, bbox_inches='tight')
#plt.show(block=True)

# Plot network output vs. delta contrast
x = choice.flatten()
y = xs[:, :, 0].flatten()
plt.figure()
sns.set_style("whitegrid")
sns.violinplot(x=x, y=y, hue=x)
plt.ylabel("Delta contrast")
plt.xlabel("Network output")
plt.savefig(os.path.join(datapath, "output_vs_contrast.png"), dpi=150, bbox_inches='tight')
#plt.legend()
#plt.show(block=False)

# Plot accuracy of network choice vs. subject choice
fig, axs = plt.subplots(1, 3, figsize=(15,5))
sns.barplot(ax=axs[0], x=['Subject', 'Network'], y=[subj_acc, network_acc])
axs[0].set_ylim(0, 1)
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Subject vs. network accuracy")

conf_mat = sklearn.metrics.confusion_matrix(subj_choice, network_choice)
sns.heatmap(ax=axs[1], data=conf_mat, annot=True, fmt='d')
axs[1].set_ylabel("Network choice")
axs[1].set_xlabel("Subject choice")
axs[1].set_title("Subject vs. network choice")

conf_mat = sklearn.metrics.confusion_matrix(bin_contrast, network_choice)
sns.heatmap(ax=axs[2], data=conf_mat, annot=True, fmt='d')
axs[2].set_ylabel("Network choice")
axs[2].set_xlabel("Delta contrast")
axs[2].set_title("Delta contrast vs. network choice")
plt.show()
quit()

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



