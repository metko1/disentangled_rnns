import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# first column: number of trials, remaining columns: sigmas in ascending order
data = pd.read_csv('results.txt', delim_whitespace=True, header=None,
                   names=['trials', 'sig1', 'sig2', 'sig3', 'sig4', 'sig5'])
d = data.melt('trials', var_name='sigma', value_name='values')
sns.lineplot(d, x='trials', y='values', hue='sigma', err_style="bars",
             linewidth=2, marker='o')#, legend=False)
plt.gca().set_xticks(data['trials'].unique())
plt.xticks(rotation=45)
plt.legend(loc='center right')
plt.show()