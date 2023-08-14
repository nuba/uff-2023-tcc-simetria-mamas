import functools
import matplotlib.pyplot as plt

from scipy import stats
import scikit_posthocs as sp
import numpy as np

t1 = [9, 8, 7, 14, 4, 10]
t2 = [16, 13, 8, 15, 8, 14]
t3 = [10, 9, 10, 12, 9, 10]

t1r = [1, 1, 1, 2, 1, 1.5]
t2r = [3, 3, 2, 3, 2, 3]
t3r = [2, 2, 3, 1, 3, 1.5]

t1sum = functools.reduce(lambda a, b: a + b, t1r)
t2sum = functools.reduce(lambda a, b: a + b, t2r)
t3sum = functools.reduce(lambda a, b: a + b, t3r)

n = 6
k = 3
friedman_stats_na_unha = (12 / (n * k * (k + 1)) * (t1sum ** 2 + t2sum ** 2 + t3sum ** 2)) - (3 * n * (k + 1))


stats, p = stats.friedmanchisquare(t1, t2, t3)

df = sp.posthoc_nemenyi_friedman(np.array([t1, t2, t3]).T)
tratamentos_labels = ['A', 'B', 'C']
df.index = tratamentos_labels
df.columns = tratamentos_labels
output = df.to_dict(orient='index')

print('Friedman Stats NA UNHA:')
print(friedman_stats_na_unha)
print('Friedman Stats:')
print(stats)
print('Friedman p:')
print(p)
print('Nemenyi dataframe')
print(df)
print('Nemenyi output')
print(output)
print('----')

heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
sp.sign_plot(df, **heatmap_args)
plt.savefig("tcc-nemenyi.png")
# plt.show()