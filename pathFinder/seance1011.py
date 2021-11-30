import numpy as np

## dans le cadre d'un tableau 3x4
l=3
c=4

a0=4  #aller vers le haut
a1=-4 #aller vers le bas
a2=-1 #aller vers la gauche
a3=1 #aller vers la droite
A=[a0,a1,a2,a3]

R=np.full((c*l,4,c*l),0)
P=np.full((c*l,4,c*l),0)

# met un 1 pour la proba en parcourant toute la liste et en réalisant les mouvements possibles
for i in range(c*l):
    for a in A:
        if -1<i+a<c*l and not( i%c==0 and a==a2) and not(i%c==c-1 and a==a3) and not(i==5) and not(i+a==5) and not(i+a==11):
            P[i,A.index(a),i+a]=1;

R[3,0,7] = -1.0
R[6,3,7] = -1.0
R[11,1,7] = -1.0
R[7,0,11] = 1.0
R[10,3,11] = 1.0

##

m = 12
n = 4
P = np.full([m,n,m],0.0)
r = np.full([m,n,m],0.0)
for i in range(m):
    for a in range(n):
        for j in range(m):
            if (a==0 and j == i + n) or (a==1 and j == i - n) or (a==2 and j == i - 1) or (a==3 and j == i + 1):
                # Test to avoid to go to state S5 and S11 has no possible actions
                if (i!=5 and j!=5 and i!=(m-1)):
                    P[i,a,j] = 1.0
r[3,0,7] = -1.0
r[6,3,7] = -1.0
r[11,1,7] = -1.0
r[7,0,11] = 1.0
r[10,3,11] = 1.0




class MDP(object):

    def __init__(self, P, r, actions):
        self.P = P
        self.r = r
        self.actions = actions
        self.m = P.shape[0]

    def Bellman(self, gamma=0.95, niter=200):
        V0 = [0.0 for i in range(self.m)]
        V1 = V0
        for i in range(niter):
            for s in range(self.m):
                V1[s] = np.max(
                    [np.sum([self.P[s, a, sp] * (self.r[s, a, sp] + gamma * V0[sp]) for sp in range(self.m)]) for a in
                     self.actions[s]])
            V0 = V1
        return V0


actions = []
for i in range(m):
    temp = []
    for a in range(n):
        if (a == 0 and i + 4 < m) or (a == 1 and i - 4 >= 0) or (a == 2 and (i % 4 - 1) >= 0) or (
                a == 3 and (i % 4 + 1) <= 3):
            temp.append(a)
    actions.append(temp)

model = MDP(P, r, actions)
print(model.Bellman())