import pandas as pd
import pdb
import sklearn
import sklearn.preprocessing
import numpy as np
import fancyimpute
import math
import heapq
import matplotlib.pyplot as plt

# Fixed random seed
# Even though we fixed the random seed, there are still some randomness in the results due to SGD based algorithms
np.random.seed(2019)

df = pd.read_csv('source.csv')
df = df.dropna()

# convert date to unix timestep
df['date'] = df['date'].values.astype("datetime64[s]").astype(int)
df['resolved_date'] = df['resolved_date'].values.astype("datetime64[s]").astype(int)

# convert categorical to ordinal
categorical_columns = ['categories', 'condition', 'team_name', 'team_size']
categorical_encoders = []
for column in categorical_columns:
	enc = sklearn.preprocessing.OrdinalEncoder()
	df[column] = enc.fit_transform(df[column].values.reshape(-1, 1))
	categorical_encoders.append(enc)

# no need to drop columns
df = df.astype(float)
#df = df.drop(columns=['uid', 'ifpid', 'fcast_no', 'ifp_length', 'num_options', 'resolution', 'is_ordinal', 'has_historic_data', 'resolved_date', 'training', 'condition', 'categories', 'team_name', 'team_size'])
df.to_csv('truth.csv', index=False)

scaler = sklearn.preprocessing.StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df_scaled.to_csv('truth_scaled.csv', index=False)
X = df_scaled.values
n_columns = X.shape[1]

def calculate_mse_ary(X_filled, missing_mask):
	assert X_filled.shape == X.shape
	mse_ary = []
	for i in range(n_columns):
		mask = missing_mask[:, i]
		mse = ((X_filled[mask, i] - X[mask, i]) ** 2).mean()
		mse_ary.append(mse)
	return mse_ary

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

	X_incomplete = df_scaled.mask(missing_mask)
	#X_incomplete.to_csv('incomplete_{}.csv'.format(drop_probablity), index=False)
	X_incomplete = X_incomplete.values

	X_filled_simple = fancyimpute.SimpleFill().fit_transform(X_incomplete)
	simple_mse = calculate_mse_ary(X_filled_simple, missing_mask)

	X_filled_knn = fancyimpute.KNN(k=3).fit_transform(X_incomplete)
	knn_mse = calculate_mse_ary(X_filled_knn, missing_mask)

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
		['SimpleFill'] + simple_mse,
		['KNN'] + knn_mse,
		['SoftImpute'] + softImpute_mse,
		['IterativeImputer'] + iter_mse,
		['IterativeSVD'] + svd_mse,
		['MatrixFactorization'] + mf_mse
	], columns=['method'] + df.columns.tolist())

	df_mse.insert(1, 'SelectedAverage', np.mean(df_mse[['brier', 'correct_fcast', 'date', 'option_1', 'option_2', 'option_3', 'option_4', 'option_5', 'std_brier', 'week_no', 'high', 'high_category_1', 'high_category_2', 'high_category_3', 'high_category_4', 'high_category_5', 'no_max']], axis=1))
	return df_mse

df_mse_80 = run_imputation(0.8)
df_mse_80.to_csv('stat.csv', index=False)
pdb.set_trace()
print('Before plot')

plot_X = []
plot_Y = []
for index, drop_probablity in enumerate(np.linspace(0.01, 0.85, 10)):
	print(index, drop_probablity)
	df_mse = run_imputation(drop_probablity)
	plot_X.append(drop_probablity)
	plot_Y.append(df_mse['SelectedAverage'])

plot_Y = np.asarray(plot_Y)
plt.plot(plot_X, plot_Y[:, 0], label='SimpleFill')
plt.plot(plot_X, plot_Y[:, 1], label='KNN')
plt.plot(plot_X, plot_Y[:, 2], label='SoftImpute')
plt.plot(plot_X, plot_Y[:, 3], label='IterativeImputer')
plt.plot(plot_X, plot_Y[:, 4], label='IterativeSVD')
plt.plot(plot_X, plot_Y[:, 5], label='MatrixFactorization')
plt.xlabel('Drop Probablity')
plt.ylabel('Normalized Average MSE on Selected Columns')
plt.legend()
plt.show()
pdb.set_trace()
print('Pause before exit')
