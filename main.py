import pandas as pd
import pdb
import sklearn
import sklearn.preprocessing
import numpy as np
import fancyimpute

# Fixed random seed
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

# drop some columns
df = df.astype(float)
#df = df.drop(columns=['uid', 'ifpid', 'fcast_no', 'ifp_length', 'num_options', 'resolution', 'is_ordinal', 'has_historic_data', 'resolved_date', 'training', 'condition', 'categories', 'team_name', 'team_size'])
df.to_csv('truth.csv', index=False)

scaler = sklearn.preprocessing.StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df.to_csv('truth_scaled.csv', index=False)

X = df_scaled.values
missing_mask = np.random.choice([True, False], size=df_scaled.shape, p=[.80,.20])
n_columns = X.shape[1]

# this part is added to ensure at least one unmasked feature per line, other wise it will report error
for row in missing_mask:
	if sum(row) == n_columns:
		add_index = np.random.randint(0, len(row))
		row[add_index] = False

X_incomplete = df_scaled.mask(missing_mask)
X_incomplete.to_csv('incomplete.csv', index=False)
X_incomplete = X_incomplete.values

def calculate_mse_ary(X_filled):
	assert X_filled.shape == X.shape
	mse_ary = []
	for i in range(n_columns):
		mask = missing_mask[:, i]
		mse = ((X_filled[mask, i] - X[mask, i]) ** 2).mean()
		mse_ary.append(mse)
	return mse_ary

X_filled_simple = fancyimpute.SimpleFill().fit_transform(X_incomplete)
simple_mse = calculate_mse_ary(X_filled_simple)

X_filled_knn = fancyimpute.KNN(k=3).fit_transform(X_incomplete)
knn_mse = calculate_mse_ary(X_filled_knn)

X_incomplete_normalized = fancyimpute.BiScaler().fit_transform(X_incomplete)
X_filled_softimpute = fancyimpute.SoftImpute().fit_transform(X_incomplete_normalized)
softImpute_mse = calculate_mse_ary(X_filled_softimpute)

X_filled_iter = fancyimpute.IterativeImputer().fit_transform(X_incomplete)
iter_mse = calculate_mse_ary(X_filled_iter)

X_filled_svd = fancyimpute.IterativeSVD().fit_transform(X_incomplete)
svd_mse = calculate_mse_ary(X_filled_svd)

X_filled_mf = fancyimpute.MatrixFactorization().fit_transform(X_incomplete)
mf_mse = calculate_mse_ary(X_filled_mf)

# It's too slow, comment out
#X_filled_nnm = fancyimpute.NuclearNormMinimization().fit_transform(X_incomplete)
#nnm_mse = ((X_filled_nnm[missing_mask] - X[missing_mask]) ** 2).mean()

with open('stat.csv', 'w') as fout:
	fout.write('method,' + ','.join(df.columns) + '\n')
	fout.write('SimpleFill,' + ','.join(map(str, simple_mse)) + '\n')
	fout.write('KNN,' + ','.join(map(str, knn_mse)) + '\n')
	fout.write('SoftImpute,' + ','.join(map(str, softImpute_mse)) + '\n')
	fout.write('IterativeImputer,' + ','.join(map(str, iter_mse)) + '\n')
	fout.write('IterativeSVD,' + ','.join(map(str, svd_mse)) + '\n')
	fout.write('MatrixFactorization,' + ','.join(map(str, mf_mse)) + '\n')

pdb.set_trace()
print('Pause before exit')
