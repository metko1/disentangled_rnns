import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from disentangled_rnns.library import rnn_utils


# load accuracy data
data = pd.read_csv('results.txt', sep='\s+', header=None,
                   usecols=[0, 1], names=['trial', 'acc'])

# remove duplicate trials, keeping only the first occurrence
#data = data.drop_duplicates(subset=['trial'], keep='first')

# count occurrences of each unique trial number
#trial_counts = data['trial'].value_counts().sort_index()
#print("Number of occurrences for each trial:")
#print(trial_counts)

# load trial information
with open("history/choice+feedback/placebo-all_trials/datasets.pkl", "rb") as f:
  _ = pickle.load(f) # dataset
  _ = pickle.load(f) # dataset_train
  _ = pickle.load(f) # dataset_eval
  repeat = pickle.load(f)

# order information about transition probabilities according to trial number in the accuracy file
repeat = repeat[data['trial'].values]
# Plot accuracy grouped by repeat
acc_df = pd.DataFrame({
    'accuracy': data.iloc[:, 1],
    'history': pd.Series(repeat, name='repeat').astype(bool)
})
plt.figure(figsize=(5, 5))
sns.barplot(data=acc_df, x='history', y='accuracy', ci=95, capsize=0.1, color="lavender", edgecolor="black")
plt.xlabel('history trial')
plt.ylabel('accuracy')
plt.title('Accuracy by block type')
plt.show()