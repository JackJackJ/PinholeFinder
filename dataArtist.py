from PIL import Image, ImageDraw
import random

#for i in range(1200):
    #if i < 700:
for i in range(1600):
    if i < 400:
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
    #x = random.randint(0, 600)
    #y = random.randint(0, 400)
        num = random.randint(3,8)
        for a in range(num):
            draw.ellipse((600/num*(a-1), 200, 600/num*(a+2), 600), fill="black")
        #draw.ellipse((600/i*(a-1), -200, 600/i*(a+2), 200), fill="black")
        #draw.ellipse((-200, 400/i*(a-1), 300, 400/i*(a+2)), fill="black")
        #draw.ellipse((300, 400/i*(a-1), 800, 400/i*(a+2)), fill="black")
        #image.save("C:/Users/Jack/Desktop/Research/training/notPinholes/" + str(i) + ".jpg")
    #else:
        #image = Image.new("RGB", (600, 400), "white")
        #draw = ImageDraw.Draw(image)
    #x = random.randint(0, 600)
    #y = random.randint(0, 400)
    #draw.ellipse((x, y, x+30, y+30), fill="white")
        image.save("C:/Users/Jack/Desktop/Research/training/edges/" + str(i) + ".jpg")
    elif i < 800:
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        num = random.randint(3,8)
        for a in range(num):
            draw.ellipse((600/num*(a-1), -200, 600/num*(a+2), 200), fill="black")
        image.save("C:/Users/Jack/Desktop/Research/training/edges/" + str(i) + ".jpg")
    elif i < 1200:
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        num = random.randint(3,8)
        for a in range(num):
            draw.ellipse((-200, 400/num*(a-1), 300, 400/num*(a+2)), fill="black")
        image.save("C:/Users/Jack/Desktop/Research/training/edges/" + str(i) + ".jpg")
    else:
        image = Image.new("RGB", (600, 400), "white")
        draw = ImageDraw.Draw(image)
        num = random.randint(3,8)
        for a in range(num):
            draw.ellipse((300, 400/num*(a-1), 800, 400/num*(a+2)), fill="black")
        image.save("C:/Users/Jack/Desktop/Research/training/edges/" + str(i) + ".jpg")



    