from distances import euc_dist
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import datetime


def split_dataset(dataset):
	tr_df = pd.read_csv(dataset)
	for col in tr_df:
		if col != 'rate':
			tr_df[col] = ((tr_df[col] - tr_df[col].min()) / (tr_df[col].max() - tr_df[col].min()))
	test_set = tr_df.sample(n=7800)
	tr_df.drop(index=test_set.index, inplace=True)
	tr_set = tr_df.values.tolist()
	tr_y = [row[-1] for row in tr_set]
	tr_x = [row[:-1]for row in tr_set]
	y_test = [row[-1] for row in test_set.values.tolist()]
	test_set.drop(columns=['rate'], inplace=True)
	x_tst = test_set.values.tolist()
	return tr_set, tr_x, tr_y, x_tst, y_test


def get_neigb(tr_set, test_row, num):
	dist = [(tr_row, euc_dist(test_row, tr_row)) for tr_row in tr_set]
	dist.sort(key=lambda tup: tup[1])
	return [dist[i][0] for i in range(num)]


def classify(tr_set, test_row, num):
	neigbs = get_neigb(tr_set, test_row, num)
	out = [row[-1] for row in neigbs]
	return max(set(out), key=out.count)


train_set, x_train, y_train, x_test, y_target = split_dataset('train_data_1.csv')
knn_clsf = KNeighborsClassifier(n_neighbors=6, n_jobs=-1)
y_pred = []

star_time = datetime.datetime.now()

knn_clsf.fit(x_train, y_train)
for x_vec in x_test:
	y_pred.append(knn_clsf.predict([x_vec]))


print("Execution time: {}".format(datetime.datetime.now() - star_time))
print(accuracy_score(y_target, y_pred))
