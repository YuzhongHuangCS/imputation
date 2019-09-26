import os

# uncomment to force CPU training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import pandas as pd
import pdb
import sklearn
import sklearn.preprocessing
import numpy as np
import fancyimpute
#import matplotlib.pyplot as plt
import heapq
import pickle

# Fixed random seed
# Even though we fixed the random seed, there are still some randomness in the results due to SGD based algorithms
np.random.seed(2019)
df1 = pd.read_csv('user_score.csv')
data = df1[['user_id', 'ifp_id']]
enc = sklearn.preprocessing.OrdinalEncoder()
df2 = enc.fit_transform(data).astype(int)

user_ids = df2[:, 0]
ifp_ids = df2[:, 1]
brier_scores = df1['brier_score']

X = np.zeros((max(user_ids)+1, max(ifp_ids)+1))
n_rows, n_columns = X.shape

for index, row in enumerate(df2):
	X[row[0], row[1]] = brier_scores[index]

df = pd.DataFrame(X)

def calculate_mse_ary(X_filled, missing_mask):
	assert X_filled.shape == X.shape
	return ((X_filled[missing_mask] - X[missing_mask]) ** 2).mean()

def run_imputation(drop_probablity):
	# using shuffle instead of generating random numbers as mask to ensure exact percentages of cells are masked
	n_total = int(np.prod(X.shape))
	n_drop = int(round(n_total * drop_probablity))
	n_keep = n_total - n_drop
	# True is drop, False is keep
	missing_mask = np.asarray([True] * n_drop + [False] * n_keep)
	np.random.shuffle(missing_mask)
	missing_mask = missing_mask.reshape(X.shape)

	# The probablity of all cells in a row are masked is (drop_probablity)^n_columns. If drop_probablity is large, this could happen.
	# If that happens, some algorithm will report error, and it's not reasonable to expect missing values could be recoverd in this case.
	# So add this logic to ensure at least one unmasked cell per line, while still keep the total number of masked cell unchanged.
	drop_count_heap = [(sum(row), index) for index, row in enumerate(missing_mask)]
	heapq.heapify(drop_count_heap)

	for row in missing_mask:
		if sum(row) == n_columns:
			keep_index = np.random.randint(0, n_columns)
			row[keep_index] = False

			most_keep_row_index = heapq.heappop(drop_count_heap)
			most_keep_row = missing_mask[most_keep_row_index[1]]
			drop_index = np.random.choice(np.where(most_keep_row==False)[0])
			most_keep_row[drop_index] = True
			heapq.heappush(drop_count_heap, (sum(most_keep_row), most_keep_row_index[1]))

	# verify the total number of masked cell unchanged
	assert np.sum(missing_mask) == n_drop

	X_incomplete = df.mask(missing_mask)
	#X_incomplete.to_csv('incomplete_{}.csv'.format(drop_probablity), index=False)
	X_incomplete = X_incomplete.values

	X_filled_simple = fancyimpute.SimpleFill().fit_transform(X_incomplete)
	simple_mse = calculate_mse_ary(X_filled_simple, missing_mask)

	X_filled_knn1 = fancyimpute.KNN(k=1).fit_transform(X_incomplete)
	knn_mse1 = calculate_mse_ary(X_filled_knn1, missing_mask)

	X_filled_knn3 = fancyimpute.KNN(k=3).fit_transform(X_incomplete)
	knn_mse3 = calculate_mse_ary(X_filled_knn3, missing_mask)

	X_filled_knn10 = fancyimpute.KNN(k=10).fit_transform(X_incomplete)
	knn_mse10 = calculate_mse_ary(X_filled_knn10, missing_mask)

	X_filled_knn15 = fancyimpute.KNN(k=15).fit_transform(X_incomplete)
	knn_mse15 = calculate_mse_ary(X_filled_knn15, missing_mask)

	X_incomplete_normalized = fancyimpute.BiScaler().fit_transform(X_incomplete)
	X_filled_softimpute = fancyimpute.SoftImpute().fit_transform(X_incomplete_normalized)
	softImpute_mse = calculate_mse_ary(X_filled_softimpute, missing_mask)

	X_filled_iter = fancyimpute.IterativeImputer().fit_transform(X_incomplete)
	iter_mse = calculate_mse_ary(X_filled_iter, missing_mask)

	X_filled_svd = fancyimpute.IterativeSVD().fit_transform(X_incomplete)
	svd_mse = calculate_mse_ary(X_filled_svd, missing_mask)

	X_filled_mf = fancyimpute.MatrixFactorization().fit_transform(X_incomplete)
	mf_mse = calculate_mse_ary(X_filled_mf, missing_mask)

	# It's too slow for large matrix, comment out
	#X_filled_nnm = fancyimpute.NuclearNormMinimization().fit_transform(X_incomplete)
	#nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()

	df_mse = pd.DataFrame([
		['SimpleFill', simple_mse],
		['KNN1', knn_mse1],
		['KNN3', knn_mse3],
		['KNN10', knn_mse10],
		['KNN15', knn_mse15],
		['SoftImpute', softImpute_mse],
		['IterativeImputer', iter_mse],
		['IterativeSVD', svd_mse],
		['MatrixFactorization', mf_mse]
	], columns=['method', 'mse'])

	return df_mse

#df_mse_80 = run_imputation(0.01)
#df_mse_80.to_csv('stat_user_ifp_brier.csv', index=False)
#pdb.set_trace()
#print('Before plot')

plot_X = []
plot_Y = []
for index, drop_probablity in enumerate(np.linspace(0.01, 0.85, 50)):
	print(index, drop_probablity)
	df_mse = run_imputation(drop_probablity)
	plot_X.append(drop_probablity)
	plot_Y.append(df_mse['mse'])

plot_Y = np.asarray(plot_Y)

with open('data/{}_{}.pickle'.format(save_prefix, run_name), 'wb') as fout:
	pickle.dump(plot_Y, fout, pickle.HIGHEST_PROTOCOL)

print('Done')
'''
plt.plot(plot_X, plot_Y[:, 0], label='SimpleFill')
plt.plot(plot_X, plot_Y[:, 1], label='KNN1')
plt.plot(plot_X, plot_Y[:, 2], label='KNN3')
plt.plot(plot_X, plot_Y[:, 3], label='KNN10')
plt.plot(plot_X, plot_Y[:, 4], label='KNN15')
plt.plot(plot_X, plot_Y[:, 5], label='SoftImpute')
plt.plot(plot_X, plot_Y[:, 6], label='IterativeImputer')
#plt.plot(plot_X, plot_Y[:, 7], label='IterativeSVD')
plt.plot(plot_X, plot_Y[:, 8], label='MatrixFactorization')
plt.xlabel('Drop Probablity')
plt.ylabel('MSE of Brier score')
plt.legend()
plt.show()
with open('user_ifp_brier.pickle', 'wb') as fout:
	pickle.dump([plot_X, plot_Y], fout, pickle.HIGHEST_PROTOCOL)

pdb.set_trace()
print('Pause before exit')
'''
