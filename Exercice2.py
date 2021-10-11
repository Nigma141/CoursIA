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