import cv2 as cv
import numpy as np

image = cv.imread('Screenshot_4.png',0)

numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(image)
#print (numLabels)
"""label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img[label_hue==0] = 0
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)"""

for i in range(1, numLabels):
    if i ==0:
        text = "examining component {}/{} (background)".format(i + 1, numLabels)
    else:
        text= "examining component {}/{}".format( i , numLabels)
    print("[INFO] {}".format(text))
    x= stats[i, cv.CC_STAT_LEFT]
    y= stats[i, cv.CC_STAT_TOP]
    w= stats[i, cv.CC_STAT_WIDTH]
    h= stats[i, cv.CC_STAT_HEIGHT]
    area= stats[i, cv.CC_STAT_AREA]
    (cX, cY)= centroids[i]
    print(cX,cY)
    print(x,y)
    print(w+h)



    
mask = np.zeros(image.shape, dtype="uint8")

L=[]
R=[500,300]
for i in range(1, numLabels):
    x = stats[i, cv.CC_STAT_LEFT]
    y = stats[i, cv.CC_STAT_TOP]
    w = stats[i, cv.CC_STAT_WIDTH]
    h = stats[i, cv.CC_STAT_HEIGHT]
    (cX, cY)= centroids[i]
    area = stats[i, cv.CC_STAT_AREA]
    keepWidth = w > 40 
    keepHeight = h > 20
    keepArea = area > 50 and area < 1500

    if all((keepWidth, keepHeight, keepArea)):
		

        #print("[INFO] keeping connected component '{}'".format(i))
        componentMask = (labels == i).astype("uint8") * 255
        mask = cv.bitwise_or(mask, componentMask)
        #print(cX,cY)
        if cX <400:
            L.append(int(cX))
            L.append(int(cY))
        else:
            R.append(int(cX))
            R.append(int(cY))
        #print(R)
        parameters = np.polyfit((800, R[-2]), (600, R[-1]), 1)
        slope = parameters[0]
        intercept = parameters[1]
        #print(slope,intercept)
        y1=600
        y2=int(y1*(3/5))
        x1= 0
        x2= int((y2-intercept)/slope)
        print(R[-2],R[-1],slope,intercept)
#cv.rectangle(mask, (282, 252), (282 + 78, 252 + 10), (255, 255, 255), 3)
cv.line(mask, (282 ,252), (305,266), (255,255,255), 3)
cv.imshow('Computer Vision', image)
cv.imshow("Characters", mask)
cv.waitKey()
cv.destroyAllWindows()