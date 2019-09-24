import pandas as pd
import pdb
import sklearn
import sklearn.preprocessing
import numpy as np
import fancyimpute
import matplotlib.pyplot as plt
import pickle

# Fixed random seed
# Even though we fixed the random seed, there are still some randomness in the results due to SGD based algorithms
np.random.seed(2019)

df = pd.read_csv('source.csv')
df = df.drop(columns=['condition'])
df = df.dropna(subset=['brier'])
df.to_csv('source_clean.csv', index=False)

# convert date to unix timestep
df['date'] = df['date'].values.astype("datetime64[s]").astype(int)
df['resolved_date'] = df['resolved_date'].values.astype("datetime64[s]").astype(int)

# convert categorical to ordinal
categorical_columns = ['categories', 'team_name', 'team_size']
categorical_encoders = []
for column in categorical_columns:
	enc = sklearn.preprocessing.OrdinalEncoder()
	df[column] = enc.fit_transform(df[column].values.reshape(-1, 1))
	categorical_encoders.append(enc)

df = df.astype(float)
df.to_csv('truth.csv', index=False)

scaler = sklearn.preprocessing.StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df_scaled.to_csv('truth_scaled.csv', index=False)
X = df_scaled.values
n_columns = X.shape[1]

# True is drop, False is keep
mask_keep = np.expand_dims([False] * n_columns, axis=0)
mask_drop = np.expand_dims([False] * 2 + [True] * 18 + [False] * 10, axis=0)

def calculate_mse_ary(X_filled, missing_mask):
	assert X_filled.shape == X.shape
	mse_ary = []
	for i in range(n_columns):
		mask = missing_mask[:, i]
		mse = np.nanmean((X_filled[mask, i] - X[mask, i]) ** 2)
		mse_ary.append(mse)
	return mse_ary

def run_imputation(drop_probablity):
	# using shuffle instead of generating random numbers as mask to ensure exact percentages of rows are masked
	n_total = len(X)
	n_drop = int(round(n_total * drop_probablity))
	n_keep = n_total - n_drop
	print(n_total, n_drop, n_keep)

	missing_mask = np.concatenate([np.repeat(mask_drop, n_drop, axis=0), np.repeat(mask_keep, n_keep, axis=0)], axis=0)
	# only first dimension will be shuffled by np.random.shuffle
	np.random.shuffle(missing_mask)

	X_incomplete = df_scaled.mask(missing_mask)
	X_incomplete.to_csv('dump/incomplete_{}.csv'.format(n_drop), index=False)
	X_incomplete = X_incomplete.values

	X_filled_simple = fancyimpute.SimpleFill().fit_transform(X_incomplete)
	pd.DataFrame(X_filled_simple, columns=df_scaled.columns).to_csv('dump/filled_simple_{}.csv'.format(n_drop), index=False)
	simple_mse = calculate_mse_ary(X_filled_simple, missing_mask)

	X_filled_knn1 = fancyimpute.KNN(k=1).fit_transform(X_incomplete)
	pd.DataFrame(X_filled_knn1, columns=df_scaled.columns).to_csv('dump/filled_knn1_{}.csv'.format(n_drop), index=False)
	knn_mse1 = calculate_mse_ary(X_filled_knn1, missing_mask)

	X_filled_knn3 = fancyimpute.KNN(k=3).fit_transform(X_incomplete)
	pd.DataFrame(X_filled_knn3, columns=df_scaled.columns).to_csv('dump/filled_knn3_{}.csv'.format(n_drop), index=False)
	knn_mse3 = calculate_mse_ary(X_filled_knn3, missing_mask)

	X_filled_knn10 = fancyimpute.KNN(k=10).fit_transform(X_incomplete)
	pd.DataFrame(X_filled_knn10, columns=df_scaled.columns).to_csv('dump/filled_knn10_{}.csv'.format(n_drop), index=False)
	knn_mse10 = calculate_mse_ary(X_filled_knn10, missing_mask)

	X_filled_knn15 = fancyimpute.KNN(k=15).fit_transform(X_incomplete)
	pd.DataFrame(X_filled_knn15, columns=df_scaled.columns).to_csv('dump/filled_knn15_{}.csv'.format(n_drop), index=False)
	knn_mse15 = calculate_mse_ary(X_filled_knn15, missing_mask)

	X_incomplete_normalized = fancyimpute.BiScaler().fit_transform(X_incomplete)
	X_filled_softimpute = fancyimpute.SoftImpute().fit_transform(X_incomplete_normalized)
	pd.DataFrame(X_filled_softimpute, columns=df_scaled.columns).to_csv('dump/filled_softimpute_{}.csv'.format(n_drop), index=False)
	softImpute_mse = calculate_mse_ary(X_filled_softimpute, missing_mask)

	X_filled_iter = fancyimpute.IterativeImputer().fit_transform(X_incomplete)
	pd.DataFrame(X_filled_iter, columns=df_scaled.columns).to_csv('dump/filled_iter_{}.csv'.format(n_drop), index=False)
	iter_mse = calculate_mse_ary(X_filled_iter, missing_mask)

	X_filled_svd = fancyimpute.IterativeSVD().fit_transform(X_incomplete)
	pd.DataFrame(X_filled_svd, columns=df_scaled.columns).to_csv('dump/filled_svd_{}.csv'.format(n_drop), index=False)
	svd_mse = calculate_mse_ary(X_filled_svd, missing_mask)

	X_filled_mf = fancyimpute.MatrixFactorization().fit_transform(X_incomplete)
	pd.DataFrame(X_filled_mf, columns=df_scaled.columns).to_csv('dump/filled_mf_{}.csv'.format(n_drop), index=False)
	mf_mse = calculate_mse_ary(X_filled_mf, missing_mask)

	# It's too slow for large matrix, comment out
	#X_filled_nnm = fancyimpute.NuclearNormMinimization().fit_transform(X_incomplete)
	#nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()

	df_mse = pd.DataFrame([
		['SimpleFill'] + simple_mse,
		['KNN1'] + knn_mse1,
		['KNN3'] + knn_mse3,
		['KNN10'] + knn_mse10,
		['KNN15'] + knn_mse15,
		['SoftImpute'] + softImpute_mse,
		['IterativeImputer'] + iter_mse,
		['IterativeSVD'] + svd_mse,
		['MatrixFactorization'] + mf_mse
	], columns=['method'] + df.columns.tolist())

	df_mse.insert(1, 'SelectedAverage', np.mean(df_mse[['option_1', 'option_2', 'option_3', 'option_4', 'option_5']], axis=1))
	df_mse.to_csv('dump/stat_{}.csv'.format(n_drop), index=False)
	return df_mse

#df_mse_80 = run_imputation(0.00018)
#df_mse_80.to_csv('stat.csv', index=False)
#pdb.set_trace()
#print('Before plot')

plot_X = []
plot_Y = []
for index, drop_probablity in enumerate(np.linspace(1/len(X), 20/len(X), 20)):
	print(index, drop_probablity)
	df_mse = run_imputation(drop_probablity)
	plot_X.append(drop_probablity)
	plot_Y.append(df_mse['SelectedAverage'])

plot_Y = np.asarray(plot_Y)

#with open('by_row_data.pickle', 'rb') as fin:
#	plot_X, plot_Y = pickle.load(fin)
plot_X = list(range(20))
plt.plot(plot_X, plot_Y[:, 0], label='SimpleFill')
plt.plot(plot_X, plot_Y[:, 1], label='KNN1')
plt.plot(plot_X, plot_Y[:, 2], label='KNN3')
plt.plot(plot_X, plot_Y[:, 3], label='KNN10')
plt.plot(plot_X, plot_Y[:, 4], label='KNN15')
plt.plot(plot_X, plot_Y[:, 5], label='SoftImpute')
plt.plot(plot_X, plot_Y[:, 6], label='IterativeImputer')
#plt.plot(plot_X, plot_Y[:, 7], label='IterativeSVD')
plt.plot(plot_X, plot_Y[:, 8], label='MatrixFactorization')
plt.xlabel('Drop Count')
plt.ylabel('Normalized Average MSE on Selected Columns')
plt.legend()
plt.show()
pdb.set_trace()
with open('drop_little.pickle', 'wb') as fout:
	pickle.dump([plot_X, plot_Y], fout, pickle.HIGHEST_PROTOCOL)

print('Pause before exit')
