import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
servoPin = 12
GPIO.setup(servoPin, GPIO.OUT)

pwm=GPIO.PWM(servoPin, 50)
pwm.start(0)

def setAngle(angle, delay):
    angle = angle % 180 #to prevent it from getting above 180
    duty = angle / 18 + 2.5
    GPIO.output(servoPin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(delay)
    GPIO.output(servoPin, False)
    pwm.ChangeDutyCycle(0)

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return int(h), int(s*100), int(v*100)

def grabCut(img):
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (xpt1,ypt1,xpt2,ypt2)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

def plotHistogram(chart,figureNumber):
    plt.ion()
    plt.axis("off")
    plt.imshow(chart)
    plt.show()


def dominantColors(IMAGE, CLUSTERS):

    #convert to rgb from bgr
    img = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
            
    #reshaping to a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    
    #save image after operations
    IMAGE = img
    
    #using k-means to cluster pixels
    kmeans = KMeans(n_clusters = CLUSTERS)
    kmeans.fit(img)
    
    #the cluster centers are our dominant colors.
    COLORS = kmeans.cluster_centers_
    
    #save labels
    LABELS = kmeans.labels_
    
    #labels form 0 to no. of clusters
    numLabels = np.arange(0, CLUSTERS+1)
   
    #create frequency count tables    
    (hist, _) = np.histogram(LABELS, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    
    #appending frequencies to cluster centers
    colors = COLORS
    
    #descending order sorting as per frequency count
    colors = colors[(-hist).argsort()]
    hist = hist[(-hist).argsort()] 
    
    #creating empty chart
    chart = np.zeros((50, 500, 3), np.uint8)
    start = 0
    #creating array to store hsv attributes
    csvRow = [img_counter] #img_counter = row no.
    #creating color rectangles
    for i in range(1,4): #Storing 3 most dominant vals
        end = start + hist[i] * 500       
        #getting rgb values
        r = colors[i][0]
        g = colors[i][1]
        b = colors[i][2]
        h,s,v=rgb2hsv(r,g,b)
        csvRow.extend([h,s,v])
        #For first iteration i = 0
        print("HSV:(%d, %d, %d)" % (h,s,v))
        cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
        start = end
    
    #Ripeness Index
    csvRow.append(ripenessIndex)    
    #plt.clf()

    myFile = open('csvFile.csv', 'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerow(csvRow)

    plotHistogram(chart, img_counter)



def change_res(width, height):
    cap.set(4, width)
    cap.set(3, height)
    

def programHasStopped():
    # When everything done, release the video capture object
    cap.release()  
    # Closes all the frames
    cv2.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()
    print("All GPIO ports have been cleaned up.")

angle = 0
setAngle(angle,0)
try:
    ripenessIndex = 1
    cap = cv2.VideoCapture(0)
    frameWidth, frameHeight = 64, 64
    change_res(frameWidth, frameHeight)

    widthGap, heightGap = 0.25, 0.3

    frameWidth = cap.get(3)
    frameHeight = cap.get(4)
    #Foreground rect coordinates
    xpt1 = int(frameWidth - frameWidth*(1-widthGap))
    ypt1 = int(frameHeight - frameHeight*(1-heightGap))
    xpt2 = int(frameWidth - frameWidth*widthGap)
    ypt2 = int(frameHeight - frameHeight*heightGap)

    filename = 'csvFile.csv'
    img_counter =  sum(1 for line in open(filename))-1 #number of rows
    #print("Current row count: ",img_counter)
    #If there are no rows, then sum returns -1
    if img_counter == -1:
            img_counter = 0
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    time.sleep(5)
    #Creating a figure only once
    #plt.figure()  
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      cap.grab()
      ret, frame = cap.retrieve()

      if ret == True:

        #Segment background
        frame = grabCut(frame)
        #Display within which foreground should be present
        #cv2.rectangle(frame,(xpt1,ypt1),(xpt2,ypt2),(0,255,0),1)
        # Display the resulting frame
        cv2.imshow('Press ESC to exit',frame)
        img_counter += 1  
        startTime = time.time()
        print("Processing frame", img_counter)
        clusters = 5
        dominantColors(frame, clusters)
        endTime = time.time()
        print("FPS: %0.2f\n" % (1/(endTime - startTime)))
        # Press Q on keyboard to  exit
        if cv2.waitKey(1)%256 == 27: #ESC pressed
          break
        
        setAngle(angle,0)
        angle = angle + 30
        if angle >=180:
            print("180 degree values of the object has been recorded")
            break
     
      # Break the loop
      else: 
        break
     
    programHasStopped()
    
except KeyboardInterrupt:
    programHasStopped()
