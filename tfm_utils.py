import cv2

def getLine(videofile, imW, imH):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    global clicks

    if success:
        #set mouse callback function for window
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.putText(image, "DOUBLE CLICK WITH LEFT MOUSE BUTTON TO DRAW THE DETECTION LINE. THEN PRESS \"Q\"",(int(imW*0.1), int(imH*0.1)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2 )
        cv2.setMouseCallback('image', mouse_callback)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return clicks

#this function will be called whenever the mouse is left-clicked twice
def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global clicks

        #store the coordinates of the right-click event
        clicks.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        #print(right_clicks)