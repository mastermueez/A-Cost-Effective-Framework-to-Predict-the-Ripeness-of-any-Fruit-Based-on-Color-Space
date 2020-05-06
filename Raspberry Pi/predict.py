import RPi.GPIO as GPIO
import time
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import sklearn
import pandas as pd
import os

GPIO.setmode(GPIO.BOARD)
#LEFT Servo
leftServoPin = 35
GPIO.setup(leftServoPin, GPIO.OUT)
pwmLeft=GPIO.PWM(leftServoPin, 50)
pwmLeft.start(0)

#RIGHT Servo
rightServoPin = 32
GPIO.setup(rightServoPin, GPIO.OUT)

pwmRight=GPIO.PWM(rightServoPin, 50)
pwmRight.start(0)

def rotateServo(angle, servoPin, pwm, delay):
    #angle = angle % 180 #to prevent it from getting above 180
    duty = angle / 18 + 2.5
    GPIO.output(servoPin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(delay)
    GPIO.output(servoPin, False)
    pwm.ChangeDutyCycle(0)

def resetServoPositions():
    #Reset both servo positions
    rotateServo(60,leftServoPin,pwmLeft,1)
    rotateServo(70,rightServoPin,pwmRight,1)


#SONAR
TRIG = 8
ECHO = 10
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

def getDistance(sonarReadingDelay,printDistance):
    
    GPIO.output(TRIG, False)
    time.sleep(sonarReadingDelay)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    while GPIO.input(ECHO)==0:
        pass
    pulse_start = time.time()

    while GPIO.input(ECHO)==1:
        pass
    
    pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    if printDistance:
        print ("Distance:",distance,"cm\n")
    return distance


import time
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

def plotHistogram(chart):
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
    csvRow = [] #img_counter = row no.
    #creating color rectangles
    for i in range(1,4): #Storing 3 most dominant vals
        end = start + hist[i] * 500       
        #getting rgb values
        r = colors[i][0]
        g = colors[i][1]
        b = colors[i][2]
        h,s,v=rgb2hsv(r,g,b)
        csvRow.append(h)
        #For first iteration i = 0
        #print("HSV:(%d, %d, %d)" % (h,s,v))
        cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
        start = end
    #plotHistogram(chart)
    return csvRow

def change_res(width, height):
    cap.set(4, width)
    cap.set(3, height)
    
def programHasStopped():
    print("Program Interrupted/Reseted. Performing GPIO cleanup...")
    GPIO.cleanup()
    print("All GPIO ports have been cleaned up")
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    pwmLeft.stop()
    pwmRight.stop()
    

df=pd.read_csv('csvFile.csv')
df.replace('?',-99999, inplace=True) #Making missing attributes outliers
df.drop(['id','sat1','val1','sat2','val2','sat3','val3'], axis=1, inplace=True) #Dropping unnecessary columns
x=np.array(df.drop(['ripenessIndex'],1))
y=np.array(df['ripenessIndex'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

#Train with given data
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

#Test with given data
accuracy = clf.score(x_test,y_test)
print("Accuracy: %0.1f%%" % (accuracy*100))

#Prediction time

try:
    resetServoPositions()
    img_counter=0
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
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

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    plt.figure()
    
    while True:
        for i in range (1, 6):
            cap.grab()
        if getDistance(1,True) < 16:
            print("Object presence detected\n")
            predictionArray = [-99999, -99999, -99999]
            count = 0
            predictionHasBeenFinalised = False
            while(cap.isOpened()):
              img_counter = img_counter+1
              cap.grab()
              ret, frame = cap.read()
              if ret == True:
                #Segment background
                frame = grabCut(frame)
                #Display within which foreground should be present
                #cv2.rectangle(frame,(xpt1,ypt1),(xpt2,ypt2),(0,255,0),1)
                # Display the resulting frame
                cv2.imshow('Press ESC to exit',frame)
                startTime = time.time()
                print("Processing frame", img_counter)
                clusters = 5
                csvRow = dominantColors(frame, clusters)

                example_measures = np.array(csvRow)
                example_measures = example_measures.reshape(1,-1)

                prediction = clf.predict(example_measures)
                count = (count+1)%3
                predictionArray[count] = prediction
                for i in range (len(predictionArray)):
                    if predictionArray[i] == predictionArray[(i+1)%3]:
                        predictionHasBeenFinalised = True
                    else:
                        predictionHasBeenFinalised = False
                        break

                if predictionHasBeenFinalised:
                    print("FINALISED prediction:", predictionArray[0])
                    if predictionArray[0]%2 ==0: #if class = even, rotate servo right
                        print("Placing object to the right")
                        rotateServo(150, rightServoPin, pwmRight, 1)
                    else:
                        print("Placing object to the left")
                        rotateServo(0, leftServoPin, pwmLeft, 1)
                    resetServoPositions()
                else:
                    print("Current Prediction:",prediction)

                
                endTime = time.time()
                print("FPS: %0.2f\n" % (1/(endTime - startTime)))
                # Press Q on keyboard to  exit
                if cv2.waitKey(1)%256 == 27: #ESC pressed
                  break
                if getDistance(1,False) > 16:
                    print("Object has been positioned")
                    break
              # Break the loop
              else: 
                break
            
    
except KeyboardInterrupt:
    programHasStopped()
