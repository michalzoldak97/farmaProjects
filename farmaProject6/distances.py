from math import sqrt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cos_dist(r1, r2):
    return cosine_similarity([r1], [r2])[0][0]


def euc_dist(r1, r2):
    dist = 0.0
    for i, _ in enumerate(r1):
        dist += (r1[i] - r2[i]) ** 2
    return sqrt(dist)


def coverage_measure(y_pred, y_true):
    label = [float(i) for i in range(6)]
    shoots = [0 for _ in range(6)]
    for i, pred in enumerate(y_pred):
        diff = euc_dist([pred], [y_true[i]])
        for res in label:
            if diff == res:
                shoots[int(res)] += 1

    for i, val in enumerate(shoots):
        print("{}= {}  =>  {}%".format(label[i], val, (sum(shoots[:i+1]) / len(y_pred) * 100)))


def acc_measure(y_pred, y_true):
    all_acc = 0.0
    for i, pred in enumerate(y_pred):
        diff = euc_dist([pred], [y_true[i]])
        all_acc += (5.0 - diff) / 5.0
    return all_acc / len(y_pred)

# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
#
# row0 = dataset[0]
# for i, row in enumerate(dataset):
# 	distance = cos_dist(row0, row)
# 	print(i, " ", distance)
#
# a = [10, 5, 15, 7, 5]
# b = [5, 10, 17, 5, 3]
# print(a)
# print([a])
# cosine = cosine_similarity([a], [b])
# print(cosine[0][0])
# coverage_measure(a, b)