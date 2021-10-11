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