import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os 
import cv2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout, BatchNormalization,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.applications import VGG16

from  tensorflow import device

X = []
folder_name = 'process_Cornell/x_split/'
for file_num in range(len(os.listdir(folder_name))):
  X.extend(np.load(folder_name + 'x_{}.npy'.format(file_num)))

X = np.array(X)
Y = np.load('process_Cornell/all_Y_test_format.npy', allow_pickle=True)
Y = np.array([np.array(y) for y in Y])

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
    if box[0]<500:
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
  x = (box[0] + (box[4] - box[0])/2)
  y = (box[1] + (box[5] - box[1])/2)
  if box[0] == box[2]:
      tan = 30
  else:
      tan = -(box[3] - box[1]) / (box[2] - box[0])
  tan = max(-11, min(tan, 11))
  w = np.sqrt(np.power((box[2] - box[0]), 2) + np.power((box[3] - box[1]), 2))
  h = np.sqrt(np.power((box[6] - box[0]), 2) + np.power((box[7] - box[1]), 2))
  angle = np.arctan(tan) * 180/np.pi
  return x, y, angle, h, w


def grasp_to_bbox(grasp):
  '''
  convert the grasping parameters into a rectangle
  '''
  x, y, theta, h, w = tuple(grasp)
  theta = theta * np.pi/180
  edge1 = [x - w/2*np.cos(theta) + h/2*np.sin(theta), y + w/2*np.sin(theta) + h/2*np.cos(theta)]
  edge2 = [x + w/2*np.cos(theta) + h/2*np.sin(theta), y - w/2*np.sin(theta) + h/2*np.cos(theta)]
  edge3 = [x + w/2*np.cos(theta) - h/2*np.sin(theta), y - w/2*np.sin(theta) - h/2*np.cos(theta)]
  edge4 = [x - w/2*np.cos(theta) - h/2*np.sin(theta), y + w/2*np.sin(theta) - h/2*np.cos(theta)]
  return [edge1, edge2, edge3, edge4]


from shapely.geometry import Polygon

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
  if iou > 0.25 and (np.abs(theta_pred-theta_true) < 30 or np.abs(theta_pred % 180-theta_true % 180)):
      return True
  else:
      return False
  
def model_1():
    model = Sequential()
    model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))    
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(5, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam',metrics=["accuracy"])
    return model

from sklearn.model_selection import train_test_split

## Data preparation
# To avoid having to reload data all the time, make a copy of the original data
X1, Y1 = np.copy(X), np.copy(Y)


#### Data separation into x_train, y_train 
x_train, x_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33)

### Give Y an acceptable format ()
y_train =np.array([np.array(y[0]) for y in y_train])
y_test =np.array([np.array(y) for y in y_test])

# test should return (something, 5)
print('y shape =', y_train.shape)
print('x shape =', x_train.shape)


def test_performance(y_pred, y_test):
  result=0
  for i in range(len(y_pred)):
    for j in range(len(y_test[i])):
      if performance(y_pred[i], y_test[i][j]):
        result+=1
        break
  return(result/len(y_pred))

def contour(img):
  image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
  return(binary)


## Training
with device('/device:GPU:0'):
    model = model_1()
    historique=model.fit(x_train, y_train,batch_size=10,epochs=5,verbose=1,validation_data=(x_train,y_train))
#    ### TestMagat
    y_pred = model.predict(x_test)
    print('Performance of my trained model : {:.1%}'.format(test_performance(y_pred, y_test)))
