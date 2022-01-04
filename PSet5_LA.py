#python
import  scipy.io as sio
import  numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import itertools
import sys
# load  the  data  matrix X
d_jest = sio.loadmat('jesterdata.mat')
X = d_jest['X']
# load  known  ratings y and  true  ratings truey
d_new = sio.loadmat('newuser.mat')
y = d_new['y']
true_y = d_new['truey']

# total  number  of joke  ratings  should  be m = 100, n = 7200
m, n = X.shape
# train  on  ratings  we know  for the new  user
train_indices = np.squeeze(y !=  -99)
num_train = np.count_nonzero(train_indices)

# test on  ratings  we don’t know
test_indices = np.logical_not(train_indices)
num_test = m - num_train

X_data = X[train_indices , 0:20] 
y_data = y[train_indices] 
y_test = true_y[test_indices]
# Part a)
# solve for weights
w = np.linalg.inv(X_data.T@X_data)@X_data.T@y_data

# compute predictions
y_hat_train = X_data @ w
y_hat_test = X[test_indices , 0:20]  @ w
# measure performance on training jokes
mistakes_train = np.sum((y_hat_train - y_data) ** 2) 
avgerr_train = mistakes_train/len(y_data)
mistakes_test = np.sum((y_hat_test - y_test) ** 2) 
avgerr_test = mistakes_test/len(y_test) 
print(avgerr_train,avgerr_test)

# display results
'''
ax1 = plt.subplot(121)
sorted_indices = np.argsort(np.squeeze(y_data)) 
ax1.plot(range(num_train), y_data[sorted_indices], 'b.',range(num_train), y_hat_train[sorted_indices], 'r.')
ax1.set_title("prediction of known ratings (trained with 20 users)")
ax1.set_xlabel("jokes (sorted by true rating)") 
ax1.set_ylabel("rating")
ax1.legend(["true rating", "predicted ratingv"], loc="upper left") 
ax1.axis([0, num_train , -15, 10])
print("Average l_2 error (train):", avgerr_train) 

# measure performance on unrated jokes
# display results
ax2 = plt.subplot(122)
sorted_indices = np.argsort(np.squeeze(y_test)) 
ax2.plot(range(num_test), y_test[sorted_indices], 'b.',range(num_test), y_hat_test[sorted_indices], 'r.')
ax2.set_title("prediction of unknown ratings (trained with 20 users)")
ax2.set_xlabel("jokes (sorted by true rating)") 
ax2.set_ylabel("rating")
ax2.legend(["true rating", "predicted rating"], loc="upper left") 
ax2.axis([0, num_test , -15, 10])
print("Average l_2 (test):", avgerr_test) 
#plt.show()
'''
# Part b)
X_data2 = X[train_indices] 
y_data2 = y[train_indices]

w2 = X_data2.T@np.linalg.inv(X_data2@X_data2.T)@y_data2
y_hat_train2 = X_data2 @ w2
y_hat_test2 = X[test_indices]  @ w2
# measure performance on training jokes
mistakes3 = np.sum((y_hat_train2 - y_data2) ** 2) 
avgerr_train2 = mistakes3 / len(y_hat_train2)
mistakes4 = np.sum((y_hat_test2 - y_test) ** 2)
avgerr_test2 = mistakes4 / len(y_hat_test2)
print(avgerr_train2,avgerr_test2)

#Part c) 
# Since the weights represent different users, by picking the user assigned the highest weight 
# we can find the best predictions for the new user. Similarly, the best two users to predict 
# the new user are the two users given the highest weights.
train_truey = true_y[train_indices]
y_from_weights = X_data2[:, np.argmax(w2)]
mistakes5 = np.sum((y_from_weights - y_test) ** 2) 
print(mistakes5/ len(y_test))

find_index = list(itertools.chain(*w2))
two_highest = [0,0]
second_index = 0
for i in range(len(find_index)):
	if find_index[i] > max(two_highest):
		two_highest[1] = find_index[i]
	elif find_index[i] > min(two_highest):
		two_highest[0] = find_index[i]
		second_index = i

y_from_weights2 = np.vstack([X[:, np.argmax(w2)],X[:, second_index]]).T
w_from_weights2 = np.matrix([w2[np.argmax(w2)],w2[second_index]])

testy_from_weights = y_from_weights2 @ w_from_weights2
compressed = list(itertools.chain(*testy_from_weights))
compressed_truey = list(itertools.chain(*true_y))
mistakes6 = np.sum((np.array(compressed) - np.array(compressed_truey)) ** 2) 
print(mistakes6/len(compressed_truey))

#Part d)
U, Sig, V = np.linalg.svd(X ,full_matrices = False)
#plt.plot(range(Sig.shape[0]),Sig)
print("rank:{}".format(np.linalg.matrix_rank(X)))
#plt.show()

#Part e) 
from mpl_toolkits.mplot3d import axes3d
threeD = U[:3,:]@X
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(threeD[0,:],threeD[1,:],threeD[2,:])
#plt.show()

#part f)
#The power meathod runs many times on a given matrix to produce 
#the greatest eigenvalue (and eigenvector) of that matrix and another non-zero matrix. 
def powerIter(A):
    rvector = np.random.rand(A.shape[1])
    for x in range(100):
        temp = A @ rvector
        t_norm = np.linalg.norm(temp)
        finalV = temp / t_norm
    return finalV

res = powerIter(X@X.T)
sigma_t = X.T @ res
v = sigma_t / np.linalg.norm(sigma_t)
sigma = np.linalg.norm(sigma_t)

U, sig2, Vt = np.linalg.svd(X)
u_tr = np.array([U[:,0]]).T
sig_tr = sig2[0]
v_tr = np.array([Vt[0,:]]).T
print("Pedicted vs. true U = {}".format(np.linalg.norm(res - u_tr) ** 2))
print("Pedicted vs. true V = {}".format(np.linalg.norm(v - v_tr) ** 2))
print("Pedicted vs. true Sigma = {}".format(abs(sigma - sig_tr)))

#part g)
#The zero vector is a starting vector for which the power method will 
#fail to find the first left and right singular vectors.

#2a)
from itertools import combinations 

def trunc_SVD(X,y):
	Wk = []
	for k in range(1,10):	
		U, sigmas, Vt = np.linalg.svd(X,full_matrices = False)
		U_k = U[:,:k]
		Sigma_plus = np.diag(1/sigmas[:k])
		V_k = Vt.T[:,:k]
		w_hat = V_k@Sigma_plus@U_k.T@y
		Wk.append(w_hat)
	return Wk

def find_error(Face_X,Face_y,func):
	ys1 = Face_y[0:16]
	ys2 = Face_y[16:32]
	ys3 = Face_y[32:48]
	ys4 = Face_y[48:64]
	ys5 = Face_y[64:80]
	ys6 = Face_y[80:96]
	ys7 = Face_y[96:112]
	ys8 = Face_y[112:128]
	subset1 = Face_X[0:16]
	subset2 = Face_X[16:32]
	subset3 = Face_X[32:48]
	subset4 = Face_X[48:64]
	subset5 = Face_X[64:80]
	subset6 = Face_X[80:96]
	subset7 = Face_X[96:112]
	subset8 = Face_X[112:128]

	yset1 = [ys1,ys2,ys3,ys4,ys5,ys6,ys7,ys8]
	set1 = [subset1,subset2,subset3,subset4,subset5,subset6,subset7,subset8]
	temp_set1 = [[0],[1],[2],[3],[4],[5],[6],[7]]
	temp = [x for x in combinations(set1,6) if len(x) == 6]
	tempy = [y for y in combinations(yset1,6) if len(y) == 6]
	temp_set = [x for x in combinations(temp_set1,6) if len(x) == 6]
	find_reg_param = []
	x_reg = []
	y_reg = []
	Wk = []
	error_final = []
	for s in range(len(temp_set)):
		x_reg.append([])
		y_reg.append([])
		for sub in range(len(set1)): 
			flatten = list(itertools.chain(*temp_set[s]))
			if sub not in flatten:
				x_reg[s].append(set1[sub])
				y_reg[s].append(yset1[sub])
	for i in range(len(temp)):
		subsets = np.vstack(temp[i])
		suby = np.vstack(tempy[i])
		Ws = func(subsets,suby)
		for w in Ws:
			y_hat_reg = x_reg[i][0] @ w
			errors = np.sum((y_hat_reg - y_reg[i][0]) ** 2)
			find_reg_param.append(errors)

		w_hat_best = Ws[np.argmin(find_reg_param)]
		y_hat_final = x_reg[i][1] @ w_hat_best
		y_hat_final_sign = np.sign(y_hat_final)
		error_final.append(np.sum(y_hat_final_sign != y_reg[i][1]) / 16)

	return sum(error_final)/len(error_final)

#b)
from numpy import eye
def ridger(X,y):
	lambda_vals = np.array([0, 0.5, 1, 2, 4, 8, 16])
	Wl = []
	for l in lambda_vals:
		U, sigmas, Vt = np.linalg.svd(X,full_matrices = False)
		Sigma = np.diag(sigmas)
		V = Vt.T
		Sigma_plus = Sigma.T@np.linalg.inv(Sigma.T@Sigma + l * eye(len(sigmas)))
		w_hat = V@Sigma_plus@U.T@y
		Wl.append(w_hat)
	return Wl

data = sio.loadmat('face_emotion_data.mat')
Face_X = data['X']
Face_y = data['y']

print(find_error(Face_X,Face_y,trunc_SVD))
print(find_error(Face_X,Face_y,ridger))
'''
n, p = np.shape(X)
# error rate for regularized least squares
error_RLS = np.zeros((8, 7))
# error rate for truncated SVD
error_SVD = np.zeros((8, 7))
# SVD parameters to test
k_vals = np.arange(9) + 1 
param_err_SVD = np.zeros(len(k_vals))
# RLS parameters to test
lambda_vals = np.array([0, 0.5, 1, 2, 4, 8, 16]) 
param_err_RLS = np.zeros(len(lambda_vals))
'''
#c)
#Face_X_aug = Face_X@np.random.randn(9,3)
#print(find_error(Face_X_aug,Face_y,trunc_SVD))
#print(find_error(Face_X_aug,Face_y,ridger))

#3a)
mnist_data = sio.loadmat('mnist.mat')
mnist_train_x = mnist_data['train_data']
mnist_train_y = mnist_data['train_target']
mnist_test_x = mnist_data['test_data']
mnist_test_y = mnist_data['test_target']

rs = [x for x in range(2,14,2)]
recon_error = []
Ur, Sigr, Vr = np.linalg.svd(mnist_train_x,full_matrices = False)
for r in rs:
	subspace_fit = Ur[:,:r]@np.diag(Sigr[:r])@(Vr[:r,:])
	diff = 0
	for i in range(len(subspace_fit)):
		diff = diff + np.linalg.norm(mnist_train_x[i] - subspace_fit[i])**2

	recon_error.append(diff/len(subspace_fit))

plt.plot(rs, recon_error, linewidth=2, color='black') 
plt.show()

#3b)
V = Vr.T
Z = mnist_train_x@V[:,:2]
c = ['red' if y == -1 else 'blue' for y in mnist_train_y[0]]
plt.scatter(Z[:,0],Z[:,1], c=c)
plt.show()

#3c)
w = np.linalg.inv(Z.T@Z)@Z.T@(mnist_train_y[0])
pred = Z@w
Z_test = mnist_test_x@V[:,:2]
test_pred = Z_test@w
test_error = (np.linalg.norm(mnist_test_y - test_pred)**2) / Z.shape[0]
train_error = (np.linalg.norm(mnist_train_y - pred)**2) / Z.shape[0]
print(test_error)
print(train_error)

#3d)
slope = (-w[0]/w[1])
plt.scatter(Z[:,0],Z[:,1], c=c)
xpoints = np.linspace(min(Z[:,0]),max(Z[:,0]),100)
y_comp = slope * xpoints
plt.plot(xpoints, y_comp, linewidth=2, color='black') 
plt.show()


