print('Exercice 4')

import numpy as np
import sklearn.neural_network as nn
import matplotlib.pyplot as pb

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