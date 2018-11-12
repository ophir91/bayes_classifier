import numpy as np
import os

import scipy.io as sio

X_PATH = '../dataset/database_mfcc_matlab/TRAIN/input'
classes = 39
features = 351
xs = list()
for x in os.listdir(X_PATH)[:1000]:
    xs.append(sio.loadmat(os.path.join(X_PATH, x))['mfcc_39_9'])

C_PATH = '../dataset/database_mfcc_matlab/TRAIN/label'
cs = list()
for c in os.listdir(C_PATH)[:1000]:
    cs.append(sio.loadmat(os.path.join(C_PATH, c))['onehot_vec'])


# with sklearn.naive_bayes
import pickle
from sklearn.naive_bayes import GaussianNB
cs_not_onehot = np.argmax(np.asarray(cs).squeeze(), axis=1)
clf = GaussianNB()
clf.fit(np.asarray(xs).squeeze(), cs_not_onehot)
f = open('GaussianNB.pckl', 'wb')
pickle.dump(clf, f)
f.close()


# hard coded:
p_c = (sum(cs))/len(cs)
miu = np.zeros([features, classes])
var = np.zeros([features, classes])

for k in range(classes):
    x_k = [x for x, c in zip(xs, cs) if c[k] == 1]
    if len(x_k) > 0:
        miu[:, k] = np.mean(np.asarray(x_k), axis=0).flatten()
        var[:, k] = np.var(np.asarray(x_k), axis=0).flatten()

np.save('p_c.npy', p_c)
np.save('miu.npy', miu)
np.save('var.npy', var)

pass


