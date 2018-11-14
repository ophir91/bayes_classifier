import numpy as np
import os

import scipy.io as sio
from scipy import signal

X_PATH = '../dataset/database_mfcc_matlab/TEST/input'
classes=39
features=351
xs = list()
for x in os.listdir(X_PATH)[:1000]:
    xs.append(sio.loadmat(os.path.join(X_PATH, x))['mfcc_39_9'])

C_PATH = '../dataset/database_mfcc_matlab/TEST/label'
cs = list()
for c in os.listdir(C_PATH)[:1000]:
    cs.append(sio.loadmat(os.path.join(C_PATH, c))['onehot_vec'])

import pickle
from sklearn.naive_bayes import GaussianNB
f = open('GaussianNB.pckl', 'rb')
clf = pickle.load(f)
f.close()
predictions = clf.predict(np.asarray(xs).squeeze())


p_c = np.load('p_c.npy')
miu = np.load('miu.npy')
var = np.load('var.npy')

for x in xs:
    p_x_c = signal.gaussian(miu, std=var)
