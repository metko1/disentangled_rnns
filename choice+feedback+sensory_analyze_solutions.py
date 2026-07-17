# Reads results from repetitions of choice+feedback+sensory script and plots the 
# latent statistics
import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from disentangled_rnns.library import rnn_utils, disrnn, plotting

# Path to the folder with params.pkl files
data_path = "choice+feedback+sensory/beta=1e-3_steps=1e5/"

latent_sigmas = []
choice_sigmas = []

# Iterate through all folders in the data path
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        fname = os.path.join(folder_path, "params.pkl")
        with open(fname, "rb") as f:
            config = pickle.load(f)
            params = pickle.load(f)
        l = disrnn.reparameterize_sigma(
            params['hk_disentangled_rnn']['latent_sigma_params'])
        l = np.sort(l)
        latent_sigmas.append(disrnn.reparameterize_sigma(l)) 
        c = disrnn.reparameterize_sigma(
            params['hk_disentangled_rnn']['choice_net_sigma_params'])  
        c = np.sort(c)  # Assuming choice sigmas are in columns 6-10
        choice_sigmas.append(disrnn.reparameterize_sigma(c))

# Sort rows by the sum across columns for the sigma matrices
latent_sigmas_array = np.array(latent_sigmas)
choice_sigmas_array = np.array(choice_sigmas)

latent_order = np.argsort(latent_sigmas_array.sum(axis=1))
choice_order = np.argsort(choice_sigmas_array.sum(axis=1))

latent_sigmas_sorted = latent_sigmas_array[latent_order]
choice_sigmas_sorted = choice_sigmas_array[choice_order]

# Sort each row of the sorted arrays in ascending order
latent_sigmas_sorted = np.array([np.sort(row) for row in latent_sigmas_sorted])
choice_sigmas_sorted = np.array([np.sort(row) for row in choice_sigmas_sorted])

# Display sorted arrays as image plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(latent_sigmas_sorted, ax=axes[0], cmap="viridis", cbar=True)
axes[0].set_title("Sorted latent_sigmas by row sum")
axes[0].set_xlabel("Sigma index")
axes[0].set_ylabel("Run index")

sns.heatmap(choice_sigmas_sorted, ax=axes[1], cmap="viridis", cbar=True)
axes[1].set_title("Sorted choice_sigmas by row sum")
axes[1].set_xlabel("Sigma index")
axes[1].set_ylabel("Run index")

plt.tight_layout()
plt.show()
