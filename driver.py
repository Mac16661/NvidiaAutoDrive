import socketio
import eventlet
import numpy as np
from flask import Flask
import torch
import torch.nn as nn
import base64
from io import BytesIO
from PIL import Image
import cv2


class NvidiaDriver(nn.Module):

  def __init__(self):
    super(NvidiaDriver, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2))
    self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2))
    self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2))
    self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
    #self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))

    # Define fully connected layers
    self.fc1 = nn.Linear(64 * 3 * 20, 100) #need to calc it properly
    self.fc2 = nn.Linear(100, 50)
    self.fc3 = nn.Linear(50, 10)
    self.fc4 = nn.Linear(10, 1)

    # Define activation function
    self.elu = nn.ELU()

    # Dropout Layer
    self.drop = nn.Dropout(p=0.5, inplace=False)
 
  def forward(self, x):
        # Apply convolutional layers
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        #x = self.elu(self.conv5(x))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 3 * 20)
        #x = torch.flatten(x)
        #print(x.shape)

        # Applying dropout
        x = self.drop(x)

        # Apply fully connected layers with activation and dropout layer
        x = self.elu(self.fc1(x))

        x= self.drop(x)

        x = self.elu(self.fc2(x))

        x = self.drop(x)

        x = self.elu(self.fc3(x))

        x = self.drop(x)

        x = self.fc4(x)

        return x


sio = socketio.Server()


# '__main__'
app = Flask(__name__)  
speed_limit = 30


def img_preprocess(img):
    img = img[60: 135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def toTensor(preprocessed_image):
  img = torch.from_numpy(preprocessed_image.astype(np.float32))
  img = img.permute(2, 0, 1)
  #img.shape
  return img

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = toTensor(image)
    # image = np.array([image])
    steering_angle = float(model(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = NvidiaDriver()
    model = torch.load('C:/Users/macy1/OneDrive/Documents/PROJECTS/SelfDriving/Model/NvidiaAutoDrive')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)