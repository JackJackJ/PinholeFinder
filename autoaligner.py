from pylablib.devices import Thorlabs
import numpy as np
from PIL import Image 
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential




def isPinhole(img): #function to determine if an image is a pinhole
    img = np.clip(img, np.uint16(0), np.uint16(255)) #standardize num values in image array to 0 to 255
    img[img < np.uint16(40)] = np.uint16(0) #convert all dark gray pixels to black
    if edgesFound: #if I have located the edges, convert all light gray to white
        img[img > np.uint16(40)] = np.uint16(255)
    img = img.astype(np.uint8) #re-encode image to bit 8 because the Image.fromarray function likes it
    data = Image.fromarray(img) #convert array to image
    data.save('frame.png') #save image as frame.png
    path = 'frame.png' #set path
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width) #load the image in keras
    )
    img_array = tf.keras.utils.img_to_array(img) 
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = new_model.predict(img_array) #predict what type of image it is (not pinhole or pinhole)
    score = tf.nn.softmax(predictions[0]) #get the prediction score

    return [class_names[np.argmax(score)], 100 * np.max(score)] #return computer's prediction

class_names = ['notPinholes', 'pinholes'] #create an array containing possible classes
new_model = tf.keras.models.load_model('pinholeFinder2.h5') #load in the model
img_height = 180 #standardize image height
img_width = 180 #standardize image width
leftX = -1 #initialize coordinates for edges
rightX = -1
topY = -1
bottomY = -1
stagex = Thorlabs.KinesisMotor("27502629") #connect to my stage (replace with your S/N)
stagey = Thorlabs.KinesisMotor("27502723") #connect to my stage (replace with your S/N)
cam = Thorlabs.ThorlabsTLCamera(serial="15555") #connect to my camera (replace with your S/N) 
edgesFound = False #initialize my edges found variable
cam.start_acquisition() #begin taking photos


cam.wait_for_frame()  # wait for the next available frame
frame = cam.snap()  # get the oldest image which hasn't been read yet
pinholeCheck = isPinhole(frame) #check if starting position is even inside the pinhole disc

while True:  # if not in the pinhole disc
    if pinholeCheck[0] == "notPinholes": #if in the pinhole disc, break loop
        break
    else:
        x = (random.random() * 15 + 5) * 34554 #get random coordinate from 5 mm to 15 mm
        y = (random.random() * 15 + 5) * 34554 #get random coordinate from 5 mm to 15 mm
        stagex.move_to(x) #move stage to random coordinate
        stagey.move_to(y)
        stagex.wait_for_stop()
        stagey.wait_for_stop()
        cam.wait_for_frame()
        frame = cam.snap()  # take a picture
        pinholeCheck = isPinhole(frame) #do I see a light?
        print(pinholeCheck)
        print(x/34554, y/34554)
        if pinholeCheck[0] == "notPinholes": #if I do, break. If not, start over.
            break

while True:  # loop to find the leftmost edge
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  # take a pic
    pinholeCheck = isPinhole(frame) #do I see a light?
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes": #yes, I see a light
        leftX = stagex.get_position() #set leftmost coordinate to my current position and break loop
        print("found")
        break
    elif stagex.get_position() < 0: #I am out of range, break loop. Task failed.
        break
    stagex.move_by(-3500) #I don't see a light, move left by 100 micron and start over
    #stagex.wait_for_stop()


print(stagex.get_position()/34554) 
print(leftX)
cam.stop_acquisition() #turn off camera
stagex.move_by(33333) #move a mm right
stagex.wait_for_stop()
cam.start_acquisition() #start up camera again
print(stagex.get_position()/34554)

while True:  # loop to find the rightmost edge
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  # take a pic
    pinholeCheck = isPinhole(frame)  # do I see a light?
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes": #yes, I see a light
        rightX = stagex.get_position() #set rightmost coordinate to my current position and break loop
        print("found")
        break
    elif stagex.get_position() > 25: #I am out of range, break loop. Task failed.
        break 
    stagex.move_by(3500) #I don't see a light, move right by 100 micron and start over
    #stagex.wait_for_stop()

print(stagex.get_position()/34554) 
stagex.move_to((rightX+leftX)/2) #go to the midpoint of my leftmost and rightmost positions. This should be the horizontal center of the disc.
stagex.wait_for_stop() 
print(stagex.get_position()/34554) 

while True:  # loop to find the bottom edge
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  #take a pic
    pinholeCheck = isPinhole(frame) #do I see a light?
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes": #yes, I see a light
        bottomY = stagey.get_position()  #set bottom coordinate to my current position and break loop
        print("found")
        break
    elif stagey.get_position() < 0: #I am out of range, break loop. Task failed.
        break
    stagey.move_by(-3500) #I don't see a light, move down by 100 micron and start over
    #stagey.wait_for_stop()

print(stagey.get_position()/34554)
cam.stop_acquisition() # stop camera
stagey.move_by(33333) #move stage up 1 mm
stagey.wait_for_stop()
cam.start_acquisition() #restart camera

while True:  # loop to find the top edge
    cam.wait_for_frame()  # wait for the next available frame
    frame = cam.snap()  # take a pic
    pinholeCheck = isPinhole(frame) #do I see a light?
    print(pinholeCheck)
    if pinholeCheck[0] == "pinholes": #yes, I see a light
        topY = stagey.get_position() #set topmost coordinate to my current position and break loop
        print("found")
        break
    elif stagey.get_position() > 25: #I am out of range, break loop. Task failed.
        break
    stagey.move_by(3500) #I don't see a light, move up by 100 micron and start over
    #stagey.wait_for_stop()

print(stagey.get_position()/34554) 
stagey.move_to((topY+bottomY)/2) #go to the midpoint of the topmost and bottom coordinate. This should be roughly the center
print(stagey.get_position()/34554) 
stagey.wait_for_stop() 

edgesFound = True #found the edges and approximated the center.
goingHorizontal = 1 #some values to start my spiral
spiral = 1

while edgesFound: #start the spiral
    for i in range(int(spiral)):
        cam.wait_for_frame()  # wait for the next available frame
        frame = cam.snap()  # take a pic
        pinholeCheck = isPinhole(frame) #do I see a light?
        if pinholeCheck[0] == "pinholes": #found it! break loop and stop program
            print("found")
            edgesFound = False
            break
        #spiral stuff
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
