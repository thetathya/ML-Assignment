import numpy as np
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import math
from numpy import unravel_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
# ytrn = M[:, 0]
# Xtrn = M[:, 1:]
# y = M[:, 0]
# x = M[:, 1:]


# Load the test data
M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytst = M[:, 0]
Xtst = M[:, 1:]



def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    h = 0
    val, cnt = np.unique(y,  return_counts=True)
    for c in cnt:
        h = h + (c/len(y))*(math.log(c/len(y),2))
    return -h


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    eny = entropy(y)
    # print(eny)
    mi = 0
    x = 0
    val, cnt = np.unique(x,  return_counts=True)
    for v in val:
        newy1 = y[np.where(x==v)[0]]
        newy2 = y[np.where(x!=v)[0]]
        x = (eny - ((len(newy1)/len(y))*(entropy(newy1))+((len(newy2))/len(y))*entropy(newy2)))
        # print(v ,eny, x)
        mi = mi + x
    return mi

# def id3(x,y):
# 	info_gain = np.zeros(shape=(6,7))
# 	for i in range(x.shape[1]):
# 		val, cnt = np.unique(x[:,i],  return_counts=True)
# 		for j in val:
# 			info_gain[i][j] = mutual_information(x[np.where(x[:,i] == j)],y[np.where(x[:,i] == j)])
# 			# print(i)
# 	return info_gain


# def rec(x,y):
# 	de = {}
# 	# if max_depth<5:
# 	if np.all(y == 1):
# 		return 1
# 	if np.all(y == 0):
# 		return 0
# 	node = id3(x,y)
# 	node = (unravel_index(node.argmax(), node.shape))
# 	x = (x[np.where(x[:,node[0]]!=node[1])])
# 	y = (y[np.where(x[:,node[0]]!=node[1])])
# 	# print(node,x,y)
# 	de[node[0],node[1],True] = rec(x,y)
# 	x = (x[np.where(x[:,node[0]]==node[1])])
# 	y = (y[np.where(x[:,node[0]]==node[1])])
# 	de[node[0], node[1], False] = rec(x,y)

# 	return de



# de = (rec(x,y))
# print(de)

# # print(de[1,1,True])
# # print(info_gain[3][3])
# # fnode = id3(x,y)
# # fnode = (unravel_index(fnode.argmax(), fnode.shape))
# # # print(x)
# # x = (x[np.where(x[:,fnode[0]]!=fnode[1])])
# # y = (y[np.where(x[:,fnode[0]]!=fnode[1])])
# # snode = id3(x,y)
# # snode = (unravel_index(snode.argmax(), snode.shape))
# # x = (x[np.where(x[:,snode[0]]!=snode[1])])
# # y = (y[np.where(x[:,snode[0]]!=snode[1])])
# # snode = id3(x,y)
# # snode = (unravel_index(snode.argmax(), snode.shape))
# # x = (x[np.where(x[:,snode[0]]!=snode[1])])
# # y = (y[np.where(x[:,snode[0]]!=snode[1])])
# # print(snode)
# # snode = id3(x,y)
# # snode = (unravel_index(snode.argmax(), snode.shape))
# # x = (x[np.where(x[:,snode[0]]!=snode[1])])
# # y = (y[np.where(x[:,snode[0]]!=snode[1])])
# # print(snode)
# # snode = id3(x,y)
# # snode = (unravel_index(snode.argmax(), snode.shape))
# # x = (x[np.where(x[:,snode[0]]!=snode[1])])
# # y = (y[np.where(x[:,snode[0]]!=snode[1])])
# # print(snode)


# # print(x)

M = np.genfromtxt('./car.data', missing_values=0, skip_header=0, delimiter=',', dtype=str)
ytrn = M[:, 6]
Xtrn = M[:, :6]


# Xtrn = onehot_encoder.fit_transform(Xtrn)
Xtrn[:,0] = label_encoder.fit_transform(Xtrn[:,0])
Xtrn[:,1] = label_encoder.fit_transform(Xtrn[:,1])
Xtrn[:,2] = label_encoder.fit_transform(Xtrn[:,2])
Xtrn[:,3] = label_encoder.fit_transform(Xtrn[:,3])
Xtrn[:,4] = label_encoder.fit_transform(Xtrn[:,4])
Xtrn[:,5] = label_encoder.fit_transform(Xtrn[:,5])
ytrn = label_encoder.fit_transform(ytrn)

Xtrn, Xtst, ytrn, ytst = train_test_split(Xtrn, ytrn, test_size=0.33, random_state=42)



acc_test = {}
acc_train = {}
depth  = [1,3,5]
for i in depth:
	model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
	model = model.fit(Xtrn, ytrn)
	
	predi = model.predict(Xtrn)
	acc_train[i] = accuracy_score(ytrn, predi)

	predi = model.predict(Xtst)
	acc_test[i] = accuracy_score(ytst, predi)

	# print(acc_test,'\n' ,acc_train)

	# plot_confusion_matrix(model, Xtst, ytst)  
	# plt.show()  

	tree.plot_tree(model)
	plt.show()
	print(model.score(Xtst, ytst))
# Learn a decision tree of depth 3
# decision_tree = id3(Xtrn, ytrn, max_depth=3)
