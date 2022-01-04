import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from random import randrange
from mpl_toolkits.mplot3d import Axes3D

#5)
def GS(matrix):
	transposed = np.hstack([np.array([c]).T for c in matrix])
	without_zeros = []
	for row in transposed:
		if not all([c == 0 for c in row]):
			without_zeros.append(row)

	norm = np.linalg.norm(without_zeros[0])
	u1 = without_zeros[0] / norm
	Us = [u1]
	Xjs = [u1]
	for c in range(2, len(without_zeros) + 1):	
		Xi = without_zeros[c-1:c][0]
		norm = np.linalg.norm(Xi)
		XiN = Xi / norm
		Us.append(XiN)

		FoundU = []
		for i in range(0,len(Us)-1):
			u2 = Us[i].dot(Us[i].T).dot(Xi)
			FoundU.append(u2)

		Xj_prime = without_zeros[c - 1] - sum(FoundU)

		if not all([x == 0 for x in Xj_prime]):
			Xjs.append(Xj_prime / np.linalg.norm(Xj_prime))

	return np.hstack(Xjs)
	
x2 = [[3,1],[0,3],[0,4]]
mx = np.matrix(x2)
print(GS(mx))

eps = 1e-12
def gram_schmidt(X):
	n, p = X.shape
	U = np.zeros((n, 0))
	for i in range(p):
		v = X[:, i]
		v = v - U@U.T@v
		v = np.reshape(v, (-1, 1))
		if np.linalg.norm(v) > eps:
			U = np.hstack((U, v / np.linalg.norm(v)))
	return U

t2 = [[3.0,1.0],[0.0,3.0],[0.0,4.0]]
tx = np.array(t2)
print(np.array(gram_schmidt(tx)))


#6) 
#a. Since the flower dataset contains full words as labels, we need to change the labels 
# into a form that can be used with least squares. I assigned a whole number between
# 1 and 3 to each of the three flower names. This produces three distinct catagories. 
# Thus, for a new array of feaures, the least squares weights will produce a result between ~0-3
# that can be  turned into a prediction by assigning it to the closes whole number. 
data = loadmat('fisheriris.mat')
flower_x = data['meas']
flower_y = data['species']

flower_labels = []
for f in flower_y:
	if f == "virginica": #100-150
		flower_labels.append([3])
	elif f == "versicolor": #50-100
		flower_labels.append([2])
	elif f == "setosa": #0-50
		flower_labels.append([1])
flower_w = np.linalg.inv(flower_x.T@flower_x)@flower_x.T@flower_labels

#b. 
def LS(set_size):
	y_testlabels = [[],[],[]]
	y_trainlabels = [[],[],[]]
	x_trainfeatures = [[],[],[]]
	x_testfeatures = [[],[],[]]
	used_nums = []
	new_num = []
	for i in range(int(set_size)):
		ran_num1 = randrange(50)
		while ran_num1 in used_nums:
			ran_num1 = randrange(50)
		ran_num2 = 50 + ran_num1
		ran_num3 = 100 + ran_num1
		'''
		y_trainlabels[0].append(flower_labels[ran_num1])
		y_trainlabels[1].append(flower_labels[ran_num2])
		y_trainlabels[2].append(flower_labels[ran_num3])
		x_trainfeatures[0].append(flower_x[ran_num1])
		x_trainfeatures[1].append(flower_x[ran_num2])
		x_trainfeatures[2].append(flower_x[ran_num3])
		'''
		y_trainlabels[0].append(flower_labels[ran_num1][0:3])
		y_trainlabels[1].append(flower_labels[ran_num2][0:3])
		y_trainlabels[2].append(flower_labels[ran_num3][0:3])
		x_trainfeatures[0].append(flower_x[ran_num1][0:3])
		x_trainfeatures[1].append(flower_x[ran_num2][0:3])
		x_trainfeatures[2].append(flower_x[ran_num3][0:3])
		used_nums.append(ran_num1)
		used_nums.append(ran_num2)
		used_nums.append(ran_num3)

	for i in range(0,150):
		if i not in used_nums:
			new_num.append(i)
			if i < 50:
				#y_testlabels[0].append(flower_labels[i])
				#x_testfeatures[0].append(flower_x[i])
				y_testlabels[0].append(flower_labels[i][0:3])
				x_testfeatures[0].append(flower_x[i][0:3])
			if 50 <= i  and i < 100:
				#y_testlabels[1].append(flower_labels[i])
				#x_testfeatures[1].append(flower_x[i])
				y_testlabels[1].append(flower_labels[i][0:3])
				x_testfeatures[1].append(flower_x[i][0:3])				
			if 100 <= i and i < 150:
				#y_testlabels[2].append(flower_labels[i])
				#x_testfeatures[2].append(flower_x[i])
				y_testlabels[2].append(flower_labels[i][0:3])
				x_testfeatures[2].append(flower_x[i][0:3])
	'''
	flower_w1 = np.linalg.inv(np.matrix(x_trainfeatures[0]).T@np.matrix(x_trainfeatures[0]))@np.matrix(x_trainfeatures[0]).T@np.matrix(y_trainlabels[0])
	flower_w2 = np.linalg.inv(np.matrix(x_trainfeatures[1]).T@np.matrix(x_trainfeatures[1]))@np.matrix(x_trainfeatures[1]).T@np.matrix(y_trainlabels[1])
	flower_w3 = np.linalg.inv(np.matrix(x_trainfeatures[2]).T@np.matrix(x_trainfeatures[2]))@np.matrix(x_trainfeatures[2]).T@np.matrix(y_trainlabels[2])

	y_hat1 = np.matrix(x_testfeatures[0]) @ flower_w1
	for i in y_hat1:
		if i[0] < 1.5:
			i[0] = 1
	mistakes1 = np.sum(y_hat1 != y_testlabels[0]) 

	y_hat2 = np.matrix(x_testfeatures[1]) @ flower_w2
	for i in y_hat2:
		if 1.5 <= i[0] and i[0] < 2.5:
			i[0] = 2
	mistakes2 = np.sum(y_hat2 != y_testlabels[1]) 
	
	y_hat3 = np.matrix(x_testfeatures[2]) @ flower_w3
	for i in y_hat3:
		if 2.5 <= i[0]:
			i[0] = 3
	mistakes3 = np.sum(y_hat3 != y_testlabels[2]) 

	error = (mistakes1 + mistakes2 + mistakes3) / (len(y_hat1)*3)
	'''
	#part a)
	fig = plt.figure(22)
	ax = fig.add_subplot(111, projection="3d")
	x1 = [item[0] for item in x_trainfeatures[0]] + [item[0] for item in x_trainfeatures[1]] + [item[0] for item in x_trainfeatures[2]]
	x2 = [item[1] for item in x_trainfeatures[0]] + [item[1] for item in x_trainfeatures[1]] + [item[1] for item in x_trainfeatures[2]]
	x3 = [item[2] for item in x_trainfeatures[0]] + [item[2] for item in x_trainfeatures[1]] + [item[2] for item in x_trainfeatures[2]]

	ys = [item[0] for elem in y_trainlabels for item in elem]
	pre_basis = [sum(x_trainfeatures[0])/len(x_trainfeatures[0]),sum(x_trainfeatures[1])/len(x_trainfeatures[1]),sum(x_trainfeatures[2])/len(x_trainfeatures[2])]

	p0, p1, p2 = pre_basis
	x0, y0, z0 = p0
	x1d, y1, z1 = p1
	x2d, y2, z2 = p2

	ux, uy, uz = u = [x1d-x0, y1-y0, z1-z0]
	vx, vy, vz = v = [x2d-x0, y2-y0, z2-z0]

	u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

	point  = np.array(p0)
	normal = np.array(u_cross_v)
	d = -point.dot(normal)

	xgrid, ygrid = np.meshgrid(range(4, 8), range(1, 6))
	z = (-normal[0] * xgrid - normal[1] * ygrid - d) * 1. / normal[2]

	ax.plot_surface(xgrid, ygrid, z, alpha=0.5)
	p = ax.scatter(x1, x2, x3, c=ys)
	ax.set_xlim(4, 8)
	ax.set_ylim(1.5, 5)
	ax.set_zlim(0.5, 7)
	ax.set_xlabel("feature 1")
	ax.set_ylabel("feature 2")
	ax.set_zlabel("feature 3")

	#part b)
	testys = [item[0] for elem in y_testlabels for item in elem]
	temp = np.matrix(np.vstack([u,v]))
	temp = [[x[0],x[1]] for x in pre_basis]
	ortho_basis = gram_schmidt(np.vstack(temp))
	
	training2 = np.vstack(x_trainfeatures)
	testing2 = np.vstack(x_testfeatures)
	z = training2 @ ortho_basis
	
	w2 = np.linalg.inv(z.T@z)@z.T@ys
	test_y_hat = (testing2 @ ortho_basis) @ w2
	test_y_hat_int = [round(x) for x in test_y_hat]
	mistakes = np.sum(test_y_hat_int != testys) 

	return (mistakes / len(test_y_hat_int))

print(LS(40))
plt.show()
#errors = 0
#for i in range(10):
#	errors = errors + LS(40)
#print(errors/10)

'''
#c.
error_per_ss = []
set_size = []
for i in range(5,38):
	avg_error = 0
	for n in range(100):
		avg_error = avg_error + LS(i) 
	final_error = avg_error /100
	set_size.append(i)
	error_per_ss.append(final_error)

plt.plot(set_size, error_per_ss, linewidth=2, color='black') 
plt.show()
'''