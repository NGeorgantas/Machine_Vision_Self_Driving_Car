import win32gui, win32ui, win32con
from time import time
import numpy as np
import cv2 as cv
import time

class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name):
        # find the handle for the window we want to capture
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        
        img = np.ascontiguousarray(img)

        return img

from DirectInput import PressKey, ReleaseKey, W, A, S, D

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def little_left():
      
      PressKey(A)
      PressKey(W)

      ReleaseKey(D)
      time.sleep(0.0005)
      ReleaseKey(A)
      #ReleaseKey(W)
      
      

def full_left():
      #PressKey(S)
      PressKey(A)

      ReleaseKey(D)      
      ReleaseKey(W)
      #ReleaseKey(A)
      #ReleaseKey(S)

def little_right():
      
      PressKey(D)
      PressKey(W)

      ReleaseKey(A)
      time.sleep(0.0005)
      ReleaseKey(D)
      #ReleaseKey(W)
      
      
      
def full_right():
      #PressKey(S)
      PressKey(D)

      ReleaseKey(A)
      ReleaseKey(W)
      #ReleaseKey(D) 
      #ReleaseKey(S)
       
def slow():
      PressKey(S)
      ReleaseKey(W)
      ReleaseKey(A)
      ReleaseKey(D)
      ReleaseKey(S)

def roi(image, polygons):
      mask = np.zeros_like(image)
      
      cv.fillPoly(mask, polygons, 255)
      masked = cv.bitwise_and(image, mask)
      return masked

def process_image(image):
    
    # convert to gray
    processed_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # edge detection
    #processed_img = cv.GaussianBlur(processed_img,(5,5), 0)
    #processed_img =  cv.Canny(processed_img, threshold1 = 50, threshold2=150)
    #ret,processed_img = cv.threshold(processed_img,127,255,cv.THRESH_BINARY)
    #ret2,processed_img = cv.threshold(processed_img,127,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    """hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    
    mask1 = cv.inRange(hsv, (19,98,171), (29,131,255))    # vrikame to yellow apsoga
    mask2 = cv.inRange(hsv, (0,0,155), (255,50,255))    # vrikame to aspro apsoga

    mask = cv.bitwise_or(mask1, mask2)
    processed_img = cv.bitwise_and(hsv,hsv, mask=mask)
    
    processed_img=cv.cvtColor(processed_img, cv.COLOR_HSV2RGB)
    processed_img=cv.cvtColor(processed_img, cv.COLOR_BGR2GRAY)"""
    kernel = np.ones((3,3), np.uint8)
    #processed_img = cv.dilate(processed_img, kernel, iterations=5)
    #processed_img = cv.erode(processed_img, kernel, iterations=3)

    processed_img = cv.GaussianBlur(processed_img,(7,7), 0)
    processed_img =  cv.Canny(processed_img, threshold1 = 40, threshold2=120)
    #processed_img = cv.dilate(processed_img, kernel, iterations=3)
    polygons = np.array([[0,500],[0,390], [300,240], [500, 240],[800,390],[800,500]])
    processed_img = roi(processed_img, [polygons])
    
    


    return processed_img

import math
wincap = WindowCapture('Grand Theft Auto V')
loop_time = time.time()
while(True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    new_screen=process_image(screenshot)
    new_screen_copy=screenshot.copy()
    
    

    
    
    #combo_image=cv.addWeighted(new_screen, 0.8, line_image, 1, 1)
    numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(new_screen)
    #print (stats, '\n\n', centroids)
    
    mask = np.zeros(new_screen.shape, dtype="uint8")
    #L=[300,300,300,300,300,300,300,300]
    #R=[600,350,600,350,600,350,600,350]
    for i in range(1, numLabels):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        (cX, cY)= centroids[i]
        area = stats[i, cv.CC_STAT_AREA]
        keepWidth = w > 30 
        keepHeight = h > 20
        keepArea = area > 40 and area < 500

        if all((keepWidth, keepHeight, keepArea)):
            # construct a mask for the current connected component and
            # then take the bitwise OR with the mask
            #print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv.bitwise_or(mask, componentMask)
            
            lines = cv.HoughLinesP(mask, 1 ,np.pi/180, 100,np.array([]), minLineLength = 30, maxLineGap = 10)

            left_coordinate=[]
            right_coordinate=[]
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv.line(mask, (x1,y1), (x2,y2), (255,255,255), 2)
                    slope = (x2-x1)/(y2-y1)
                    if slope<0:
                        left_coordinate.append([x1,y1,x2,y2])
                    elif slope>0:
                        right_coordinate.append([x1,y1,x2,y2])
            l_avg = np.average(left_coordinate, axis =0)
            r_avg = np.average(right_coordinate, axis =0)

            try:
                l =l_avg.tolist()
                r = r_avg.tolist()
                c1,d1,c2,d2 = r
                a1,b1, a2,b2 = l
                l_slope = (b2-b1)/(a2-a1)
                r_slope = (d2-d1)/(c2-c1)
                l_intercept = b1 - (l_slope*a1)
                r_intercept = d1 - (r_slope*c1)
                
                y=360
                l_x = (y - l_intercept)/l_slope
                r_x = (y - r_intercept)/r_slope
                x_1 = int(l_x)
                x_2 = int(r_x)

                

                distance = math.sqrt((r_x - l_x)**2+(y-y)**2)
                #line_center repressent the center point on the line
                line_center = distance/2
                  
                center_pt =[(l_x+line_center)]
                center_pt_show= int((l_x+line_center))

                f_r = [(l_x+(line_center*0.20))]
                f_l = [(l_x+(line_center*1.80))]
                
                f_r_show = int((l_x+(line_center*0.20)))
                f_l_show = int((l_x+(line_center*1.80)))
                
            except:
                print("An exception occurred")
            
            
                  
            

            center_fixed =[415]
            
            
            
            """if center_pt==center_fixed:
                straight()
                print('forward')
                
            elif center_pt > center_fixed and center_fixed > f_r:
                little_right()
                print('right')
                
                
            elif center_pt < center_fixed and center_fixed < f_l:
                little_left()
                print('left')
                
            elif center_fixed < f_r:
                full_right()
                print('full_ right')
            elif center_fixed > f_l:
                full_left()
                print('full_left')
            else:
                slow()
                print('slow')"""
    
    
    
    
    try:
        left=cv.line(mask, (0,450), (x_1,360), (255,255,255), 3)
        right=cv.line(mask, (800,430), (x_2,360), (255,255,255), 3)
        line2=cv.line(mask, (x_1,360), (x_2,360), (255,255,255), 3)

        margin=cv.line(mask,(f_r_show,330),(f_l_show,330),(255,255,255),5)

        vert=cv.line(mask,(center_pt_show,330-25),(center_pt_show,330+25),(255,255,255),5)

        cyr=cv.circle(mask, (410,330),5,(255,255,255),10)
    except:
        print("error")
    cv.imshow('Computer Vision', mask)
    
    
    # debug the loop rate
    #print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time.time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key pressesaw
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()