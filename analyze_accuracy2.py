# Reads results from the history-choice-accuracy2.py script and plots the accuracy statistics
import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from disentangled_rnns.library import rnn_utils, disrnn, plotting

# Path to the folder with params.pkl files
data_path = "history/choice+feedback/accuracy_beta=1e-3"

acc = []
latent_sigmas = []
choice_sigmas = []

# Iterate through all folders in the data path
for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        fname = os.path.join(folder_path, "results.csv")
        data = pd.read_csv(fname, header=None)
        acc.append(data.iloc[0][0])  # Assuming accuracy is in the first column of the first row
        l = np.sort(data.iloc[0][1:6])  # Assuming latent sigmas are in columns 1-5
        latent_sigmas.append(disrnn.reparameterize_sigma(l)) 
        c = np.sort(data.iloc[0][6:])  # Assuming choice sigmas are in columns 6-10
        choice_sigmas.append(disrnn.reparameterize_sigma(c))

# Histogram of acc values
plt.figure(figsize=(6, 4))
sns.histplot(pd.Series(acc), bins=20, kde=True, color="steelblue")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.title("Distribution of accuracy across runs")
plt.tight_layout()
#plt.show()

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
#plt.show()

# line  plot of accuracy vs. sum of latent sigmas, sorted by sum of latent sigmas
latent_sum = latent_sigmas_array.sum(axis=1)
latent_sum_sorted = latent_sum[latent_order]
acc_sorted = np.array(acc)[latent_order]
plt.figure(figsize=(6, 4))
sns.lineplot(x=latent_sum_sorted, y=acc_sorted, linewidth=2)
sns.set_style("darkgrid")
axes[0].set_xlabel("latent (sorted)")
axes[0].set_ylabel("sigma")
axes[0].set_title("Sorted latent sigmas")
#plt.show()

# bar plot of accuracies in runs with 2 vs. 3 latents
acc_array = np.array(acc)

#two_latents_idx = np.where((latent_sum > 3) & (latent_sum < 3.5))
#three_latents_idx = np.where((latent_sum > 4) & (latent_sum < 4.5))
#df = pd.DataFrame({'num_latents': np.concatenate([np.full(len(two_latents_idx[0]), 2), 
#                                   np.full(len(three_latents_idx[0]), 3)]), 
#                    'accuracy': np.concatenate([acc_array[two_latents_idx], 
#                                    acc_array[three_latents_idx]])})

# Count, for each row, how many entries are below 0.2
latent_below_02_counts = np.sum(latent_sigmas_array < 0.2, axis=1)
choice_below_02_counts = np.sum(choice_sigmas_array < 0.2, axis=1)
df = pd.DataFrame({'num_latents': latent_below_02_counts, 'accuracy': acc_array})
plt.figure(figsize=(6, 4))
sns.violinplot(data=df, x='num_latents', y='accuracy', inner='box', palette="Set2")
plt.xlabel("Number of bottlenecks")
plt.ylabel("Mean accuracy")
plt.title("Accuracy by number of bottlenecks")
plt.tight_layout()
plt.show()  