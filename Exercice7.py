import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


import tensorflow as tf
from keras import models
from keras import layers

# ***********************************************************
#                       Fonction
# ***********************************************************

def make_Dictionary(train_ndir):
    allwords=[]
    # create a list of all words
    for i in range(len(train_ndir)):
        for j in range (len(train_ndir[i])):
            allwords+=data[j]
    dictionary= Counter(allwords)

    # Remove not usefull words

    listtoremove=list(dictionary)
    for item in listtoremove:
        if item.isalpha()== False:
            del dictionary[item]
        elif len (item) == 1 :
            del dictionary[item]
    #Sorting most commun words
    dictionary=dictionary.most_common(3000)
    return dictionary

# ***********************************************************
#                   Creating Word Dictionary
# ***********************************************************

### Open the file in the directory
file = open("sms.txt" , "r")
line=file.readline()
target=[]
data =[]

while 1:
### Split each line in a list , the delimiters are removed
    line=file.readline().split()
### Stop reading when the len of the line is equal to zero
    if len(line)==0:
        break
    target.append(line[0])
    del(line[0])
    data.append(line)
file.close()

dictionary=make_Dictionary(data)
print(len(dictionary))
print(dictionary)
M=len(dictionary)
# ***********************************************************
#                   Feature Extraction Process
# ***********************************************************
List_Data = []
Vecteur_Data = []
for i in range(len(dictionary)):
    List_Data.append(dictionary[i][0])

### Data vector's creation
for k in range(len(data)):
    vecteur = np.zeros(M)  # M words in the dictionary after sorting it
    for j in range(len(data[k])):
        for i in range(M):
            if data[k][j] == List_Data[i]:
                vecteur[i] = vecteur[i] + 1
                break
    Vecteur_Data.append(vecteur)

### Label vector's creation  ham=0 , spam=1
Target_Vecteur = np.zeros(5573)
for i in range(len(target)):
    if target[i] == 'ham':
        Target_Vecteur[i] = 0.0
    elif target[i] == 'spam':
        Target_Vecteur[i] = 1.0

### Prediction vector's creation
Target_pred = np.zeros(4000)
Target_test = np.zeros(1573)
Target_pred[0:4000] = Target_Vecteur[0:4000]
Target_test[0:1573] = Target_Vecteur[4000:5573]

Data_pred = []
Data_test = []
Data_pred[0:4000] = Vecteur_Data[0:4000]
Data_test[0:1573] = Vecteur_Data[4000:5573]

Data_pred = np.array(Data_pred).reshape(4000, M, 1)
Data_test = np.array(Data_test).reshape(1573, M, 1)
Target_pred = np.array(Target_pred)
Target_test = np.array(Target_test)

Target_pred = tf.keras.utils.to_categorical(Target_pred, M)
Target_test = tf.keras.utils.to_categorical(Target_test, M)
Target_pred = Target_pred.reshape(4000, M, 1)
Target_test = Target_test.reshape(1573, M, 1)

print(Data_pred.shape)
print(Target_pred.shape)
print(Data_test.shape)
# print(Target_test)
# Création du modèle Keras et entraînement



model = models.Sequential()

model.add(layers.Dense(8, activation='relu',  input_shape=(M,1)))

model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid',name="sortie"))



sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

model.fit(Data_pred, Target_pred, epochs = 5, batch_size = 200, validation_split=0.5)

score = model.evaluate(Data_test,Target_test, verbose=0)

model.summary()

print('Test loss:', score[0])

print('Test accuracy:', score[1])