import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from disentangled_rnns.library import rnn_utils, disrnn, plotting

# Path to the folder with params.pkl files
data_path = "history/choice+feedback/stability"

latent_sigmas_all = []
choice_sigmas_all = []

# Iterate through all folders in the data path
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        params_file = os.path.join(folder_path, "params.pkl")
        
        if os.path.exists(params_file):
            with open(params_file, "rb") as f:
                _ = pickle.load(f) # ignore config
                params = pickle.load(f)
            
            # Extract and compute latent_sigmas
            latent_sigma_params = params['hk_disentangled_rnn']['latent_sigma_params']
            latent_sigmas = disrnn.reparameterize_sigma(latent_sigma_params)
            latent_sigmas_all.append(np.sort(latent_sigmas))
            
            # Extract and compute choice_sigmas
            choice_sigma_params = params['hk_disentangled_rnn']['choice_net_sigma_params']
            choice_sigmas = disrnn.reparameterize_sigma(choice_sigma_params)
            choice_sigmas_all.append(np.sort(choice_sigmas))

# Flatten arrays for plotting
latent_sigmas_flat = np.concatenate(latent_sigmas_all)
choice_sigmas_flat = np.concatenate(choice_sigmas_all)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot sorted latent_sigmas
#axes[0].plot(range(len(latent_sigmas_flat)), latent_sigmas_flat, linewidth=2)
sns.lineplot(data=np.array(latent_sigmas_all).transpose(), ax=axes[0], linewidth=2)
sns.set_style("darkgrid")
axes[0].set_xlabel("latent (sorted)")
axes[0].set_ylabel("sigma")
axes[0].set_title("Sorted latent sigmas")

# Plot sorted choice_sigmas
#axes[1].plot(range(len(choice_sigmas_flat)), choice_sigmas_flat, linewidth=2)
sns.lineplot(data=np.array(choice_sigmas_all).transpose(), ax=axes[1], linewidth=2)
axes[1].set_xlabel("latent (sorted)")
axes[1].set_ylabel("sigma")
axes[1].set_title("Sorted choice sigmas")

plt.tight_layout()
plt.show()
