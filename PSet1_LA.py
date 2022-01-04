import numpy as np 

#a)
x = [[1,1,0,0,0],[1,1,1,1,0],[0,0,0,2,3],[0,0,1,1,1]]
#b)
y = [1,2,6,3,8]
#c)
z = [[6,2,2,0]]

mx = np.matrix(x)
my = np.matrix(y) 
mz = np.matrix(z)

#b)
xy = mx @ my.T
#c)
xz = mz @ mx
#d)
final = xz @ my.T

print(xy)
print(xz)
print(final)