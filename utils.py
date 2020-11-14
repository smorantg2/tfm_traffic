import cv2
import numpy as np

def getLine(videofile, imW, imH):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    global clicks
    clicks = []
    if success:
        #set mouse callback function for window
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.putText(image, "DOUBLE CLICK WITH LEFT MOUSE BUTTON TO DRAW THE DETECTION LINE. THEN PRESS \"Q\"",(int(imW*0.1), int(imH*0.1)), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0),2 )
        cv2.setMouseCallback('image', mouse_callback)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return clicks

def getPPoints(videofile, imW, imH):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    global perspective_points

    if success:
        #set mouse callback function for window
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.putText(image, "DOUBLE CLICK WITH LEFT MOUSE BUTTON THE 4 POINTS NEEDED FOR PESPECTIVE. THEN PRESS \"Q\"",(int(imW*0.1), int(imH*0.1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0),2 )
        cv2.setMouseCallback('image', mouse_callback_points)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return perspective_points

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

#this function will be called whenever the mouse is left-clicked twice
def mouse_callback_points(event, x, y, flags, params):

    #right-click event value is 2
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global perspective_points

        #store the coordinates of the right-click event
        perspective_points.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        #print(right_clicks)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def calculate_distance_point_line(centroid, inicio_linea, final_linea):
    d = np.abs(np.cross(final_linea-inicio_linea,centroid-inicio_linea)/np.linalg.norm(final_linea-inicio_linea))
    return d