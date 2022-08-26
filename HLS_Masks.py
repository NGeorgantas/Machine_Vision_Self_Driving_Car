import cv2 as cv
screenshot=cv.imread('Screenshot_2.png')

hsv = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)

## mask of green (36,25,25) ~ (86, 255,255)
# mask = cv.inRange(hsv, (36, 25, 25), (86, 255,255))
mask1 = cv.inRange(hsv, (22,50,220), (29,180,255))    # vrikame to yellow apsoga
mask2 = cv.inRange(hsv, (0,0,150), (255,8,255))    # vrikame to aspro apsoga

mask = cv.bitwise_or(mask1, mask2)
target = cv.bitwise_and(screenshot,screenshot, mask=mask)






cv.imshow('Computer Vision', target)
cv.waitKey()
cv.destroyAllWindows()