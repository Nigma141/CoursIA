import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import cv2
from shapely.geometry import Polygon
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout, BatchNormalization,Conv2D,MaxPooling2D,Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.applications import VGG16
from  tensorflow import device


X = []
folder_name = 'process_Cornell/x_split/'
for file_num in range(len(os.listdir(folder_name))):
    X.extend(np.load(folder_name + 'x_{}.npy'.format(file_num)))

X = np.array(X)
Y = np.load('process_Cornell/all_Y_test_format.npy', allow_pickle=True)
Y = np.array([np.array(y) for y in Y])

print('Input dimensions: ', X.shape)
print('Output dimensions: ', Y.shape)
print('First element of Y', Y[0])

## modification de l'image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.33/255, 0.33/255, 0.33/255])


## Draw the grasping rectangles on an  image
def draw_rectangle(mes_rectangles, image):
    '''
  mes_rectangles is a list of 4 points of a rectangle
  image is a numpy array where we draw the grasping rectangles
  '''
    for rectangle in mes_rectangles:
        point1, point2 = tuple([int(float(point)) for point in rectangle[0]]), tuple(
            [int(float(point)) for point in rectangle[1]])
        point3, point4 = tuple([int(float(point)) for point in rectangle[2]]), tuple(
            [int(float(point)) for point in rectangle[3]])
        cv2.line(image, point1, point2, color=(0, 0, 255), thickness=1)
        cv2.line(image, point3, point4, color=(0, 0, 255), thickness=1)
        cv2.line(image, point2, point3, color=(0, 255, 0), thickness=2)
        cv2.line(image, point4, point1, color=(0, 255, 0), thickness=2)
    return image


def vizualise(x, y):
    '''
  x : is a raw image
  y : is a list of lists with the grasping parameters

  even if you visualize only one grasping parameter,
  there must be a list of list
  '''
    tot_rect = []
    for box in y:
        if box[0] < 500:
            rect = grasp_to_bbox(box)
            rect = [float(item) for vertex in rect for item in vertex]
            grasp = bboxes_to_grasps(rect)
            new_rect = grasp_to_bbox(grasp)
            tot_rect.append(new_rect)
    image = draw_rectangle(tot_rect, x)
    plt.imshow(image)
    plt.title('grasping rectangles')


def bboxes_to_grasps(box):
    '''
  convert a rectangle into the grasping parameters
  '''
    x = (box[0] + (box[4] - box[0]) / 2)
    y = (box[1] + (box[5] - box[1]) / 2)
    if box[0] == box[2]:
        tan = 30
    else:
        tan = -(box[3] - box[1]) / (box[2] - box[0])
    tan = max(-11, min(tan, 11))
    w = np.sqrt(np.power((box[2] - box[0]), 2) + np.power((box[3] - box[1]), 2))
    h = np.sqrt(np.power((box[6] - box[0]), 2) +np.power((box[7] - box[1]), 2))
    angle = np.arctan(tan) * 180 / np.pi
    return x, y, angle, h, w


def grasp_to_bbox(grasp):
    '''
  convert the grasping parameters into a rectangle
  '''
    x, y, theta, h, w = tuple(grasp)
    theta = theta * np.pi / 180
    edge1 = [x - w / 2 * np.cos(theta) + h / 2 * np.sin(theta), y + w / 2 * np.sin(theta) + h / 2 * np.cos(theta)]
    edge2 = [x + w / 2 * np.cos(theta) + h / 2 * np.sin(theta), y - w / 2 * np.sin(theta) + h / 2 * np.cos(theta)]
    edge3 = [x + w / 2 * np.cos(theta) - h / 2 * np.sin(theta), y - w / 2 * np.sin(theta) - h / 2 * np.cos(theta)]
    edge4 = [x - w / 2 * np.cos(theta) - h / 2 * np.sin(theta), y + w / 2 * np.sin(theta) - h / 2 * np.cos(theta)]
    return [edge1, edge2, edge3, edge4]


def performance(Y_pred, Y_true):
    '''
  Y_pred and Y_true are the two grasping parameters
  '''
    grasp_pred = grasp_to_bbox(Y_pred)
    grasp_true = grasp_to_bbox(Y_true)

    p_pred = Polygon(grasp_pred)
    p_true = Polygon(grasp_true)

    iou = p_pred.intersection(p_true).area / (p_pred.area + p_true.area - p_pred.intersection(p_true).area)
    theta_pred, theta_true = Y_pred[2], Y_true[2]
    if iou > 0.25 and (np.abs(theta_pred - theta_true) < 30 or np.abs(theta_pred % 180 - theta_true % 180)):
        return True
    else:
        return False


n = np.random.randint(500)
vizualise(X[n], Y[n])


def model_1B():
    l2 = regularizers.l2(0.01)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(224, 224,1),
                     activation='relu'))  # Couche de convolution : traite les images en leur appliquant un filtre de convolution dont les valeurs sont les poids du modèle
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Rédui la taille des images en ne reprenant que les pixels les plus significatifs
    model.add(BatchNormalization())  
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))  # Couche de convolution : traite les images en leur appliquant un filtre de convolution dont les valeurs sont les poids du modèle
    model.add(MaxPooling2D(pool_size=(2, 2)))  # Rédui la taille des images en ne reprenant que les pixels les plus significatifs
    model.add(
        BatchNormalization())  # Normalise les données avant de rentrer dans une couche de convolution pour améliorer les performances
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2))
    model.add(Dropout(0.1))  # Couche de dropout permettant de lutter contre l'overfitting
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (5, 5), padding='same', activation='relu', kernel_regularizer=l2))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # Transforme une image en un vecteur

    model.add(Dense(units=512, activation='relu'))  # Permet de traiter le vecteur issu de la couche flatten
    model.add(Dense(5,
                    activation='linear'))  # On on indique que le nombre de neurones sur la couche vaut 5 car on a 5 paramètres de grasping à prédire
    # On utilise 'linear' comme fonction d'activation car on est en regression

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[
        'accuracy'])  # On utilise mean_square_error comme fonction de perte car elle est adaptée à la régression
    return model

def model_1():
    model = Sequential()
    model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))  
    model.add(Dense(1024,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(5, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='rmsprop',metrics=['accuracy'])
    print(model.summary())
    return model


## Data preparation
# To avoid having to reload data all the time, make a copy of the original data
X1, Y1 = np.copy(X), np.copy(Y)

#### Data separation into x_train, y_train
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.6, random_state=42)
x_train=np.array([rgb2gray(x_train[i]) for i in range(len(x_train))])
x_train=x_train.reshape(-1,224,224,1)
y_test_copie = np.copy(y_test)  # We keep a y_test in the initial format
x_test=np.array([rgb2gray(x_test[i]) for i in range(len(x_test))])
x_test=x_test.reshape(-1,224,224,1)

### Give Y an acceptable format ()
# on prend seulement le premier carré de la liste

y_train = np.array([[y_train[i][0][0]/224,y_train[i][0][1]/224,(y_train[i][0][2]+180)/360,y_train[i][0][3]/224,y_train[i][0][4]/224] for i in range(len(y_train))])

# test should return (something, 5)
#print('test', y_train.shape)

## Training

def show_prediction(model, x):
    y=model.predict(x)
    print(y[1:4])

    return ()
with device('/device:GPU:0'):
    model = model_1B()
    model.fit(x_train, y_train,batch_size=1,epochs=5,verbose=1,validation_data=(x_train,y_train))
    show_prediction(model, x_test)