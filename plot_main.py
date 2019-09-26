import pandas as pd
import pdb
import sklearn
import sklearn.preprocessing
import numpy as np
import fancyimpute
import matplotlib.pyplot as plt
import pickle
import sys

all_Y = []
for i in range(48):
	with open(f'data/user_ifp1_{i}.pickle', 'rb') as fin:
		data_Y = pickle.load(fin)
		all_Y.append(data_Y)
for i in range(48):
	with open(f'data/user_ifp2_{i}.pickle', 'rb') as fin:
		data_Y = pickle.load(fin)
		all_Y.append(data_Y)

all_Y = np.asarray(all_Y)
plot_Y = np.mean(all_Y, axis=0)
plot_X = np.linspace(0.01, 0.85, 50)

plt.plot(plot_X, plot_Y[:, 0], label='SimpleFill')
plt.plot(plot_X, plot_Y[:, 1], label='KNN1')
plt.plot(plot_X, plot_Y[:, 2], label='KNN3')
plt.plot(plot_X, plot_Y[:, 3], label='KNN10')
plt.plot(plot_X, plot_Y[:, 4], label='KNN15')
plt.plot(plot_X, plot_Y[:, 5], label='SoftImpute')
plt.plot(plot_X, plot_Y[:, 6], label='IterativeImputer')
plt.plot(plot_X, plot_Y[:, 7], label='IterativeSVD')
plt.plot(plot_X, plot_Y[:, 8], label='MatrixFactorization')
plt.xlabel('Drop Probablity')
plt.ylabel('MSE of whether user make a forecast')
plt.legend()
plt.show()
pdb.set_trace()

print('Pause before exit')
