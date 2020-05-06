import os
import random
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


# COUNTING NO OF FILES
def get_file_count_from(dir):
    path, dirs, files = next(os.walk(dir))
    file_count = len(files)
    return file_count

# GENERATING RANDOM NUMBERS
def generate_random_num_list(start, end):
    randomlist = []
    for i in range(0,5):
        n = random.randint(start, end)
        randomlist.append(n)
    return randomlist

def get_img_file_names_from(dir):
    # Get names of all files
    files = os.listdir(dir)
    img_file_list = []
    for file in files:
        if file.endswith('.jpg'):
            img_file_list.append(file)
    return img_file_list

def rgb2hsv(r, g, b):
    #print("RGB:(%d, %d, %d)" % (r,g,b))
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
    plt.axis("off")
    plt.imshow(chart)
    plt.show()


def dominantColors(IMAGE, CLUSTERS, ripeness_index, file_name, csv_file_name):
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

    #creating color rectangles
    dominant_colors = []
    for i in range(1,4): #Storing 3 most dominant vals
        end = start + hist[i] * 500       
        #getting rgb values
        r = colors[i][0]
        g = colors[i][1]
        b = colors[i][2]
        h,s,v=rgb2hsv(r,g,b)
        dominant_colors.append(h)
        dominant_colors.append(s)
        dominant_colors.append(v)
        #For first iteration i = 0
        #print("HSV:(%d, %d, %d)" % (h,s,v))

        cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
        start = end
    print(dominant_colors)
    df = pd.DataFrame()
    df=df.append({"hue1":dominant_colors[0], "hue2":dominant_colors[3], "hue3":dominant_colors[6], "ripeness_index":ripeness_index, "file_name":file_name}, ignore_index=True)
    df_org = pd.read_csv(csv_file_name)
    df_org = df_org.append(df)
    df_org.to_csv(csv_file_name, index=False)
    plotHistogram(chart)

def createCSVfileFrom(fileName):
	column_names = ["file_name", "hue1",	"hue2",	"hue3",	"ripeness_index"]
	df = pd.DataFrame(columns = column_names)
	df.to_csv(fileName, index = False)

frameWidth, frameHeight = 85, 48
widthGap, heightGap = 0.005, 0.08

#Foreground rect coordinates
xpt1 = int(frameWidth - frameWidth*(1-widthGap))
ypt1 = int(frameHeight - frameHeight*(1-heightGap))
xpt2 = int(frameWidth - frameWidth*widthGap)
ypt2 = int(frameHeight - frameHeight*heightGap)

#  _________________________
# | User defined parameters |
#  -------------------------

# Available folders: Green, Yellowish_Green, Midripen, Overripen

# This should be done for both test and train, for all banana classes
csv_file_name = "Green_train.csv" # file name that the program will create where HSV values will be written
file_dir = "/Dataset (Images)/Resized (Train, Test)/Green/Train" # folder directory containing the images
ripeness_index = 1 # appropriate ripeness index

createCSVfileFrom(csv_file_name)
img_files = get_img_file_names_from(file_dir)

for file_name in img_files:
    frame = cv2.imread(file_dir+"/"+file_name)
    frame = grabCut(frame)
    frame_with_bounding_box = frame
    dominantColors(frame, 5, ripeness_index, file_name, csv_file_name)
    cv2.rectangle(frame_with_bounding_box,(xpt1,ypt1),(xpt2,ypt2),(0,255,0),1)
    cv2.imshow(file_name,frame_with_bounding_box)

cv2.waitKey(0)
cv2.destroyAllWindows()