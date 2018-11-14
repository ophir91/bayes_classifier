import numpy as np
import os
import copy
import scipy.io as sio
import scipy.stats
from random import shuffle
import matplotlib.pyplot as plt

classes = 39
features = 351
train_size = 1000000
test_size = 20000
eps = np.finfo(np.float32).eps
pc_x_old = eps
#load train
# load test

X_PATH = '../dataset/database_mfcc_matlab/TRAIN/input'
C_PATH = '../dataset/database_mfcc_matlab/TRAIN/label'

files_train_x = os.listdir(X_PATH)
files_train_c = os.listdir(C_PATH)

index_train = np.arange(len(files_train_x))
shuffle(index_train)
x_train = list()
c_train = list()

for i, j in enumerate(index_train):
    x_train.append(sio.loadmat(os.path.join(X_PATH,files_train_x[j]))['mfcc_39_9'])
    c_train.append(sio.loadmat(os.path.join(C_PATH,files_train_c[j]))['onehot_vec'])
    if i==train_size-1:
        break


# load test

X_PATH = '../dataset/database_mfcc_matlab/TEST/input'
C_PATH = '../dataset/database_mfcc_matlab/TEST/label'

files_test_x = os.listdir(X_PATH)
files_test_c = os.listdir(C_PATH)

index_test = np.arange(len(files_test_x))
shuffle(index_test)
x_test = list()
c_test = list()

for i, j in enumerate(index_test):
    x_test.append(sio.loadmat(os.path.join(X_PATH,files_test_x[j]))['mfcc_39_9'])
    c_test.append(sio.loadmat(os.path.join(C_PATH,files_test_c[j]))['onehot_vec'])
    if i==test_size-1:
        break




def create_norm(miu,std):
    norm_dict = {}
    for k in range(classes):
        for i in range(features):
            norm_dict['norm_c{}_f{}'.format(k,i)] = scipy.stats.norm(miu[i, k], std[i, k])
    return norm_dict


def p_x(p_c,likelihood):
    return np.dot(p_c.T,likelihood)+eps

def predict(x,miu, std, classes, p_c, norm_dict): #predicts class for x (vector of 391)
    likelihood=np.ones((classes, 1))
    for k in range(classes):
        for i in range(features):
            likelihood[k] *= norm_dict['norm_c{}_f{}'.format(k,i)].pdf(x[i])+eps
    px = p_x(p_c, likelihood)
    pc_x = (likelihood*p_c)/(px)

    return np.argmax(likelihood*p_c) , pc_x



# hard coded:
p_c = (sum(c_train))/len(c_train)
miu = np.zeros([features, classes])
var = np.zeros([features, classes])
std = np.zeros([features, classes])
m = np.zeros([features, classes])

mini_batch_size = 1000
I =[]
accuracy_list = []
for i in range(int(np.ceil(len(x_train)/mini_batch_size))):

    x = x_train[i*mini_batch_size:(i+1)*mini_batch_size]
    c = c_train[i*mini_batch_size:(i+1)*mini_batch_size]
    for k in range(classes):
        x_k = [x for x, c in zip(x, c) if c[k] == 1]
        n_k = len(x_k) # batch size
        samp_size = m[:, k] + n_k

        if n_k > 0:
            miu_m = copy.deepcopy(miu[:, k])
            miu_n = np.mean(np.asarray(x_k), axis=0).flatten()
            miu[:, k] = miu_m * ((m[:, k]) / (samp_size)) + miu_n* ((n_k/ samp_size))
            # miu[:, k] = np.mean(np.asarray(x_k), axis=0).flatten()
            var_n = np.var(np.asarray(x_k), axis=0).flatten()
            var[:, k] = ((m[:, k])*(var[:, k]+miu_m**2)+(n_k)*(var_n+miu_n**2))/samp_size-miu[:, k]**2
            std[:, k] = np.sqrt(var[:, k])
            # var[:, k] = np.var(np.asarray(x_k), axis=0).flatten()
        m[:, k] = samp_size

    np.save('p_c_iter.npy', p_c)
    np.save('miu_iter.npy', miu)
    np.save('var_iter.npy', var)
    np.save('i_iter.npy', i)


# validation
    norm_dict = create_norm(miu, std)
    tmp = [predict(x_te, miu, std, classes, p_c, norm_dict) for x_te in x_test]
    pc_x = [tmp[i][1] for i, tmp2 in enumerate(tmp)]
    c_predict = [tmp[i][0] for i, tmp2 in enumerate(tmp)]

    I.append(np.sum(pc_x*np.log(pc_x/pc_x_old)))
    np.save('I.npy', I)

    a = np.reshape(c_test, (-1, classes))
    accuracy_list.append(np.sum(np.argmax(a, axis=1) == c_predict))
    np.save('accuracy.npy', accuracy_list)


# p_data = np.arange(mini_batch_size,(i+1)*mini_batch_size+1,mini_batch_size)/((i)*mini_batch_size)*100

# len(x_train)

plt.plot(I)
plt.ylabel('Information gain')
plt.xlabel(' % data')
plt.show()









# x_k_test = [x for x, c in zip(x_train, c_train) if c[k] == 1]
# std_x_train = np.std(np.asarray(x_k_test), axis=0)
# std_x_train_batch = std[:, k]
#
# c= np.array(std_x_train_batch).flatten()
# b= np.array(std_x_train).flatten()
# a = print(np.sum(b-c< 1e-12))
#
# mean_x_train = np.nean(x_train[:, k])
# print()

pass
# np.save('p_c.npy', p_c)
# np.save('miu.npy', miu)
# np.save('var.npy', var)

pass


