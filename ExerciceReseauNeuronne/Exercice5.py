print('Exercice 5')

import numpy as np
from numpy import array, dot, random
import matplotlib.pyplot as plt

# Learning step
p = 30
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
plt.show()

# Mise en place du data set
training_data=[[x1[i],y1[i]] for i in range(p)]+[[x2[i],y2[i]] for i in range(p)]


# initialisation
cdt=0
xb,yb=training_data[1]
xa,ya=training_data[0]

#bouclage
while cdt==0:
    distancea=[np.sqrt((xa-training_data[i][0])**2 + (ya-training_data[i][1])**2) for i in range(len(training_data))]
    distanceb=[np.sqrt((xb-training_data[i][0])**2 + (yb-training_data[i][1])**2) for i in range(len(training_data))]
    GroupeAX,GroupeAY=[],[]
    GroupeBX,GroupeBY=[],[]
    for i in range(len(training_data)):
        if distancea[i] == min(distancea[i],distanceb[i]):
            GroupeAX += [training_data[i][0]]
            GroupeAY += [training_data[i][1]]
        else:
            GroupeBX += [training_data[i][0]]
            GroupeBY += [training_data[i][1]]
    plt.scatter(GroupeAX,GroupeAY,c='red', marker = 'o', s=4)
    plt.scatter(GroupeBX,GroupeBY,c='green', marker = 'o', s=4)
    plt.scatter([xa,xb],[ya,yb], c='black', marker='x', s=8)
    plt.show()

    xan, yan=[np.sum(GroupeAX)/len(GroupeAX), np.sum(GroupeAY)/len(GroupeAY)]
    xbn, ybn = [np.sum(GroupeBX) / len(GroupeBX), np.sum(GroupeBY) / len(GroupeBY)]
    if [xb, yb] == [xbn, ybn] and [xa, ya] == [xan, yan]:
         cdt = 1
    else:
        [xb, yb] = [xbn, ybn]
        [xa, ya] = [xan, yan]