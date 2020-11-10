import cv2
import numpy as np

#this function will be called whenever the mouse is left-clicked twice
def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global clicks

        #store the coordinates of the right-click event
        clicks.append([x, y])

def getPoints(img, imW, imH):
    image = img
    global clicks

    #set mouse callback function for window
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.putText(image, "DOUBLE CLICK WITH LEFT MOUSE BUTTON TO DRAW 4 POINTS. THEN PRESS \"Q\"",(int(imW*0.1), int(imH*0.1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0),2 )
    cv2.setMouseCallback('image', mouse_callback)
    for i in range(len(clicks)):
        cv2.circle(image, tuple(clicks[i]), 10, (0, 0, 255), 10)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clicks

img = cv2.imread('./perspective/road_long.png')
rows,cols,ch = img.shape

clicks = []
clicks = getPoints(img, cols, rows)


cv2.circle(img, tuple(clicks[0]), 10, (0,0,255), 10)
cv2.circle(img, tuple(clicks[1]), 10, (0,0,255), 10)
cv2.circle(img, tuple(clicks[2]), 10, (0,0,255), 10)
cv2.circle(img, tuple(clicks[3]), 10, (0,0,255), 10)
cv2.circle(img, tuple(clicks[4]), 10, (0,255,0), 10)

pts1 = np.float32([clicks[0],clicks[1],clicks[2],clicks[3]])
pts2 = np.float32([[0,0],[500,0],[0,600],[500,600]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(600,600))

point_rand = clicks[4]

new_point = np.array(np.matmul(M, [point_rand[0], point_rand[1], 1]), np.int8)

print(new_point[:2])
cv2.circle(dst, tuple(new_point[:2]), 10, (0,255,0), 10)

cv2.circle(img, tuple(clicks[4]), 10, (0,255,0), 10)



cv2.imshow("original", img)
cv2.imshow("warped", dst)

key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()