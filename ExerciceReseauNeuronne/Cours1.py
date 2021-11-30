## Exercice 1 K-nearest neighbors Algorithm
print("Exercie 1")

import numpy as np
import pylab as pb

# inputs
data=[[0,1],[1,1],[3,3],[2,4],[4,4],[4,5],[4,6],[3,5]]
target = [0,0,0,0,1,1,1,1]
P = [2.5,3.5]

#affichage
pb.plot(data[0:4],'x',color='blue',label='classe 0')
pb.plot(data[4:8],'x',color='red',label='classe 1')
pb.plot(P[0],P[1],'o',color='black')
pb.legend()
pb.axis('equal')
pb.show()

# define the distance function
def distance(P,Q):
    res= np.sqrt((Q[0]-P[0])**2+(Q[1]-P[1])**2)
    return res



# define the k-nearest neighboors function for any value of k
def knn(P,data,target,k):
    Dico={}
    for i in range(len(data)):
        Dico[distance(data[i],P)]=[data[i],target[i]]
    sort_Dico = sorted(Dico.items(), key=lambda x: x[0])
    listK=list(dict(sort_Dico).values())[0:k]
    label=0
    for i in range(k):
        label+=listK[i][1]/k
    return  round(label)


# Test the function for a given data point P
print("The class of point ",P," is : ",knn(P,data,target,3))

## Exercice 2 Perceptron
print("Exercie 2")

from random import choice
from numpy import array, dot, random

def sigma(x):
    if x<0:
        return 0.0
    else:
        return 1.0


Mode='XOR'
if Mode =='AND':
    training_data = [
            (array([0,0,1]), 0),
            (array([0,1,1]), 1),
            (array([1,0,1]), 1),
            (array([1,1,1]), 1),
            ]
elif Mode=='NOT':
    training_data = [
            (array([0,1,1]), 0),
            (array([1,0,1]), 1),
            ]
elif Mode =='NOR':
    training_data = [
            (array([0,0,1]), 1),
            (array([0,1,1]), 0),
            (array([1,0,1]), 0),
            (array([1,1,1]), 0),
            ]
elif Mode =='XOR':
    training_data = [
            (array([0,0,1]), 0),
            (array([0,1,1]), 1),
            (array([1,0,1]), 1),
            (array([1,1,1]), 0),
            ]


w = random.rand(3)
errors = []
eta = 0.5
n = 50
# Learning step
for i in range(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - sigma(result)
    errors.append(error)
    w += eta * error * x

# Prediction step
for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, sigma(result)))

# commentaire : le Xor focntionne pas

## Exercice 3
print('Exercice 3')
import numpy as np
from numpy import array, dot, random
import matplotlib.pyplot as plt

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
    X1=array([x,y,1])
    result = dot(w, X)
    error = expected - sigma(result)
    errors.append(error)
    w += eta * error * X

    Y1=[-w[0]*(-R-1)/w[1]-w[2]/w[1],-w[0]*(R+1)/w[1]-w[2]/w[1]]
    pb.plot(X1,Y1,color=((i/n,0,0,1)))
pb.xlim([-R-1,R+1])
pb.ylim([-R-1,R+1])
pb.show()

## Exercice 4
print('Exercice 4')

import numpy as np
import sklearn.neural_network as nn

# Function to approximate
def f(x):
    return [(x[0]**3)*(np.sin(2*x[1])**2),x[2]**2+x[0]]

# Synthetic dataset generation
inf_bound = 0
sup_bound = np.pi
N_train = 1000
N_test = 50
X = np.array([[np.random.uniform(inf_bound, sup_bound),np.random.uniform(inf_bound, sup_bound),np.random.uniform(inf_bound, sup_bound)] for i in range(N_train)])
Y = np.array([f(x) for x in X])

# Neural Network Learning process
model = nn.MLPRegressor(hidden_layer_sizes = (10, 10, 10, 10), activation = 'tanh', solver = 'lbfgs', max_iter = 100000) # le solver est basé sur une approximation de la dérivé seconde
model.fit(X, Y)

# Neural network data test set
X_test = np.array([[np.random.uniform(inf_bound, sup_bound),np.random.uniform(inf_bound, sup_bound),np.random.uniform(inf_bound, sup_bound)] for i in range(N_test)])
Y_test = np.array([f(x) for x in X_test])

print('fin création data')

# Neural network prediction
Y_pred = model.predict(X_test)
Y_pred = np.reshape(Y_pred,(N_test,2))

# Errors between the model and the True values
errors =np.sqrt((Y_test - Y_pred)[:,0]**2 + (Y_test - Y_pred)[:,1]**2)
pb.plot(errors)
pb.show()
print('Mean test error :', np.mean(errors[:]))
print('Max test error :', np.max(errors[:]))

## Exercice 5 Inverse commande robot
print('Exercice 5')

import numpy as np
import matplotlib.pyplot as pb
import sklearn.neural_network as nn

#creation des données d'entrainement
def Robot(q):
    l1=2.1
    l2=1.5
    return[l1*np.sin(q[0])+ l2*np.sin(q[0]+q[1]),l1*np.cos(q[0])+ l2*np.cos(q[0]+q[1])]

borne_sup=np.pi
borne_inf=0
nb_train=1000
nb_test=20

X_train=np.array([[np.random.uniform(borne_inf,borne_sup),np.random.uniform(borne_inf,borne_sup)] for i in range(nb_train)])
Y_train=np.array([Robot(X_train[i]) for i in range(nb_train)])


# entrainement du model
model = nn.MLPRegressor(hidden_layer_sizes = (10, 10, 10, 10), activation = 'tanh', solver = 'lbfgs', max_iter = 10000) # le solver est basé sur une approximation de la dérivé seconde
model.fit(Y_train,X_train)

# données de test
X_test=np.array([[np.random.uniform(borne_inf,borne_sup),np.random.uniform(borne_inf,borne_sup)] for i in range(nb_test)])

Y_test=np.array([Robot(X_test[i]) for i in range(nb_test)])

# test du modele
X_pred = model.predict(Y_test)
X_pred = np.reshape(X_pred,(nb_test,2))


Y_pred=np.array([Robot(X_pred[i]) for i in range(nb_test)])
Y_pred = np.reshape(Y_pred,(nb_test,2))
# Quantification des erreurs
errors =np.sqrt((Y_test - Y_pred)[:,0]**2 + (Y_test - Y_pred)[:,1]**2)

# Affichage
plt.subplot(121)
pb.plot(Y_test[:,0],Y_test[:,1],'x',label='Vrai')
pb.plot(Y_pred[:,0],Y_pred[:,1],'x',label='Prediction')
pb.legend()

plt.subplot(122)
plt.plot(errors)
pb.show()

##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)


l1=2.1
l2=1.5

t =[0,l1*np.sin(np.pi/2),Robot([np.pi/2,np.pi/2])[0]]
s =[0,l1*np.cos(np.pi/2),Robot([np.pi/2,np.pi/2])[1]]
l, = plt.plot(t, s)



X_pred = model.predict(np.reshape(Robot([np.pi/2,np.pi/2]),(1,2)))[0]
t_pred=[0,l1*np.sin(X_pred[0]),Robot(X_pred)[0]]
s_pred =[0,l1*np.cos(X_pred[1]),Robot(X_pred)[1]]
m, = plt.plot(t_pred, s_pred)
ax.margins(x=2,y=3)

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'angle q1', -np.pi, np.pi, valinit=np.pi/2)
samp = Slider(axamp, 'angle q2', 0,  np.pi, valinit=np.pi/2)


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata([0,l1*np.sin(amp),Robot([amp,freq])[0]])
    l.set_xdata([0,l1*np.cos(amp),Robot([amp,freq])[1]])

    X_pred = model.predict(np.reshape(Robot([amp,freq]),(1,2)))[0]
    t_pred=[0,l1*np.sin(X_pred[0]),Robot(X_pred)[0]]
    s_pred =[0,l1*np.cos(X_pred[0]),Robot(X_pred)[1]]
    m.set_ydata(t_pred)
    m.set_xdata(s_pred)


    fig.canvas.draw_idle()


sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)



plt.show()
## Exercice 6




