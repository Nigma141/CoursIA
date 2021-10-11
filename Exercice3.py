


print('Exercice 3')
import numpy as np
from numpy import array, dot, random
import matplotlib.pyplot as plt
from random import choice

def sigma(x):
    if x<0:
        return 0.0
    else:
        return 1.0

# Learning step
p = 100
R = 1.5
t = np.linspace(0, 2*np.pi, p)

# First class +1
x1= [ 1+R*random.rand()*np.cos(t[n]) for n in range(p) ]
y1= [ 1+R*random.rand()*np.sin(t[n]) for n in range(p) ]

# Second class 0
x2= [ -1+R*random.rand()*np.cos(t[n]) for n in range(p) ]
y2= [ -1+R*random.rand()*np.sin(t[n]) for n in range(p) ]

plt.scatter(x1,y1,c='red', marker = 'o', s=4)
plt.scatter(x2,y2,c='green', marker = 'o', s=4)
plt.title('Perceptron algorithm')
#plt.savefig('datapoints.png')
#plt.show()

# Mise en place du data set
training_data=[[x1[i],y1[i],0] for i in range(p)]+[[x2[i],y2[i],1] for i in range(p)]
# Learning step


w = random.rand(3)
errors = []
eta = 0.001
n=5000
for i in range(n):
    x,y, expected = choice(training_data)
    X=array([x,y,1])
    result = dot(w,X)
    error = expected - sigma(result)
    errors.append(error)
    w += eta * error * X

    Y1=[-w[0]*(-R-1)/w[1]-w[2]/w[1],-w[0]*(R+1)/w[1]-w[2]/w[1]]
    plt.plot(Y1,color=((i/n,0,0,1)))
plt.xlim([-R-1,R+1])
plt.ylim([-R-1,R+1])
plt.show()