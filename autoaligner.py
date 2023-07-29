from pylablib.devices import Thorlabs
import numpy as np
from PIL import Image 
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential




def isPinhole(img):
    img = np.clip(img, np.uint16(0), np.uint16(255))
    img[img < np.uint16(40)] = np.uint16(0)
    if edgesFound:
        img[img > np.uint16(40)] = np.uint16(255)
    img = img.astype(np.uint8)
    data = Image.fromarray(img)
    data.save('frame.png')
    path = 'frame.png'
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return [class_names[np.argmax(score)], 100 * np.max(score)]

class_names = ['notPinholes', 'pinholes']
new_model = tf.keras.models.load_model('pinholeFinder2.h5')
img_height = 180
img_width = 180
leftX = -1
rightX = -1
topY = -1
bottomY = -1
stagex = Thorlabs.KinesisMotor("27502629")
stagey = Thorlabs.KinesisMotor("27502723")
cam = Thorlabs.ThorlabsTLCamera(serial="15555")
edgesFound = False
cam.start_acquisition()

#while True:
#    cam.wait_for_frame()  # wait for the next available frame
#    frame = cam.read_oldest_image()  # get the oldest image which hasn't been read yet
#    pinholeCheck = isPinhole(frame)
#    if pinholeCheck[0] == "notPinholes" and pinholeCheck[1] > 85:
#        break
#    else:


cam.wait_for_frame()  # wait for the next available frame
frame = cam.snap()  # get the oldest image which hasn't been read yet
pinholeCheck = isPinhole(frame)

while True:  # acquisition loop
    if pinholeCheck[0] == "notPinholes":
        break
    else:
        x = (random.random() * 15 + 5) * 34554
        y = (random.random() * 15 + 5) * 34554
        stagex.move_to(x)
        stagey.move_to(y)
        stagex.wait_for_stop()
        stagey.wait_for_stop()
        cam.wait_for_frame()
        frame = cam.snap()  # get the oldest image which hasn't been read yet
        pinholeCheck = isPinhole(frame)
        print(pinholeCheck)
        print(x/34554, y/34554)
        if pinholeCheck[0] == "notPinholes":
            break

while True:  # acquisition loop
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  # get the oldest image which hasn't been read yet
    pinholeCheck = isPinhole(frame)
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes":
        leftX = stagex.get_position()
        print("found")
        break
    elif stagex.get_position() < 0:
        break
    stagex.move_by(-3500)
    #stagex.wait_for_stop()


print(stagex.get_position()/34554)
print(leftX)
cam.stop_acquisition()
stagex.move_by(33333)
stagex.wait_for_stop()
cam.start_acquisition()
print(stagex.get_position()/34554)

while True:  # acquisition loop
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  # get the oldest image which hasn't been read yet
    pinholeCheck = isPinhole(frame)
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes":
        rightX = stagex.get_position()
        print("found")
        break
    elif stagex.get_position() < 0:
        break
    stagex.move_by(3500)
    #stagex.wait_for_stop()

print(stagex.get_position()/34554)
stagex.move_to((rightX+leftX)/2)
stagex.wait_for_stop()
print(stagex.get_position()/34554)

while True:  # acquisition loop
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  # get the oldest image which hasn't been read yet
    pinholeCheck = isPinhole(frame)
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes":
        bottomY = stagey.get_position()
        print("found")
        break
    elif stagey.get_position() < 0:
        break
    stagey.move_by(-3500)
    #stagey.wait_for_stop()

print(stagey.get_position()/34554)
cam.stop_acquisition()
stagey.move_by(33333)
stagey.wait_for_stop()
cam.start_acquisition()

while True:  # acquisition loop
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  # get the oldest image which hasn't been read yet
    pinholeCheck = isPinhole(frame)
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes":
        topY = stagey.get_position()
        print("found")
        break
    elif stagey.get_position() < 0:
        break
    stagey.move_by(3500)
    #stagey.wait_for_stop()

print(stagey.get_position()/34554)
stagey.move_to((topY+bottomY)/2)
print(stagey.get_position()/34554)
stagey.wait_for_stop()

edgesFound = True
goingHorizontal = 1
spiral = 1

while edgesFound:
    for i in range(int(spiral)):
        cam.wait_for_frame()  # wait for the next available frame
        frame = cam.snap()  # get the oldest image which hasn't been read yet
        pinholeCheck = isPinhole(frame)
        if pinholeCheck[0] == "pinholes":
            print("found")
            edgesFound = False
            break

        elif goingHorizontal%2==1 and int(spiral)%2==1:
            stagex.move_by(350)
            print(str(stagex.get_position()/34554) + ", " + str(stagey.get_position()/34554))

        elif goingHorizontal%2==0 and int(spiral)%2==1:
            stagey.move_by(350)
            print(str(stagex.get_position()/34554) + ", " + str(stagey.get_position()/34554))

        elif goingHorizontal%2==1 and int(spiral)%2==0:
            stagex.move_by(-350)
            print(str(stagex.get_position()/34554) + ", " + str(stagey.get_position()/34554))

        elif goingHorizontal%2==0 and int(spiral)%2==0:
            stagey.move_by(-350)
            print(str(stagex.get_position()/34554) + ", " + str(stagey.get_position()/34554))
        
        stagey.wait_for_stop()
        stagex.wait_for_stop()

    goingHorizontal +=1
    spiral += 0.5