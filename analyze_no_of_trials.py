import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# first column: number of trials, remaining columns: sigmas in ascending order
data = pd.read_csv('results.txt', sep='\s+', header=None,
                   names=['trials', 'sig1', 'sig2', 'sig3', 'sig4', 'sig5',
                          ' err_train', 'err_valid'])
# Plot sigmas
d = data.iloc[:,0:6].melt('trials', var_name='sigma', value_name='values')
plt.subplot(2,1,1)
sns.lineplot(d, x='trials', y='values', hue='sigma', err_style="bars",
             linewidth=2, marker='o')#, legend=False)
plt.gca().set_xticks(data['trials'].unique())
plt.xticks(rotation=45)
plt.legend(loc='center right')

# Plot errors
d = data.iloc[:,[0,6,7]].melt('trials', var_name='error', value_name='values')
plt.subplot(2,1,2)
sns.lineplot(d, x='trials', y='values', hue='error', err_style="bars",
             linewidth=2, marker='o')#, legend=False)
plt.gca().set_xticks(data['trials'].unique())
plt.xticks(rotation=45)
plt.legend(loc='center right')

plt.show()