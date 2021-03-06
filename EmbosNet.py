from imutils.video import VideoStream
from imutils.video import FPS
from utils import *
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys
import time
import matplotlib.pyplot as plt
import importlib.util
import json
from classes import TrackableObject, CentroidTracker
import math

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model_name", required=True, help="Name of the .tflite file")
ap.add_argument("-p", "--model_path", required=True, help="Folder the .tflite file is located in")
ap.add_argument("-v", "--video", required = True, type=str, help="path to input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file, with /")
ap.add_argument("-t", "--threshold", required = True, type=float, default=0.5, help="minimum probability to filter weak detections")
#ap.add_argument("-s", "--skip_frames", type=int, default=5, help="# of skip frames between detections")
ap.add_argument("-u", "--use_tpu", help="Whether to use TPU or not")
ap.add_argument("-d", "--display", type = bool, help="Whether to display all the action or not")
args = vars(ap.parse_args())

# ---------------- Import TensorFlow libraries ---------
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if args["use_tpu"]:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if args["use_tpu"]:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if args["use_tpu"]:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    model_name = 'edgetpu.tflite'
else:
    model_name = "detect.tflite"
# ------------- OTHER VARIABLES NEEDED FOR OBJECT DETECTION --------------
# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = args["video"]

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, args["model_path"], model_name)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, args["model_path"], "labels.txt")

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if args["use_tpu"]:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# ----------------------- CREATE JSON ------------
json_vehicles = {}
json_vehicles["vehicles"] = []

# ------------------ Open Video File --------------- ###
print("[INFO] Opening video file...")
video = cv2.VideoCapture(VIDEO_PATH)
imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---------- GET LINE FOR COUNTING CARS (USER DEFINED) ------------

clicks = []
clicks = getLine(VIDEO_PATH, imW, imH)
pA = np.array(clicks[0])
pB = np.array(clicks[1])

# ------------------ Other initializations -------------------- ###
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject

maxDistance = (175/np.mean([1080,1920])) * np.mean([imH, imW])

ct = CentroidTracker(maxDisappeared=5, maxDistance=maxDistance)
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalcoming = 0
totalgoing = 0

# start the frames per second throughput estimator
fps = FPS().start()

if args["output"] is not None:
    # ------------------- SAVING VIDEO -----------
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    out = cv2.VideoWriter(str(args["output"])+'output.avi', fourcc, 20.0, (imW,imH))

#This is the maximum distance the vehicle must be from the line to be considered as candidate for counting.
#We want to count only vehicles near the line, because far from it the detections can be too unestable.
#And it could detect the same vehicle twice during its time on video
d_from_line = (150/1080) * imH

#Timestamp for every car that crosses the line -> for the json file
num_frame = 0

# ================= LOOP OVER FRAMES FROM THE VIDEO STREAM ===========
while video.isOpened():

    num_frame = num_frame + 1

    # grab the next frame
    ret, frame = video.read()
    meta_color = (0,255,0)

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if not ret:
        print("[INFO] Reached the end of the video!")
        break

    # First convert the frame from BGR to RGB
    # Then resize the frame to have a maximum width of 300 pixels (that's what the model can take)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []
    vehicle_types = []


    # set the status and initialize our new set of object trackers
    status = "Detecting"

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects

    # Loop over all detections if confidence is above minimum threshold
    for i in range(len(scores)):
        if (scores[i] > args["threshold"]) and (scores[i] <= 1.0):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions,
            # need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            if args["display"] == True or args["output"] is not None:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 1)

            # Draw label
            object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window

            if args["display"] == True or args["output"] is not None:
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text

            rect = (xmin, ymin, xmax, ymax)
            rects.append(rect)
            vehicle_types.append(object_name)


    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids

    objects, types, timestamps  = ct.update(rects, vehicle_types, [num_frame]*len(rects))


    # --- CHECK IF THEY'RE COMING OR GOING
    # loop over the tracked objects

    for (objectID, centroid), vehicle_type in zip(objects.items(), types.items()):

        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid, vehicle_type, num_frame)
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:

            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            to.vehicle_timestamp.append(num_frame)
            # check to see if the object has been counted or not
            if not to.counted:
                side = (to.centroids[-1][0] - pA[0] ) * (pB[1] - pA[1]) - (to.centroids[-1][1] - pA[1]) * (pB[0] - pA[0])
                dist_line = np.abs(calculate_distance_point_line(np.array(centroid), pA, pB))

                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < -12 and side > 0 and dist_line <= d_from_line:
                    totalgoing += 1
                    to.counted = True
                    json_vehicles["vehicles"].append({"vehicle": to.vehicle_type, "time": num_frame / 29.6, "direction":1})
                    meta_color = (0, 0, 255)
                    #print("Detected: ", objectID)
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 12 and side < 0 and dist_line <= d_from_line:
                    totalcoming += 1
                    to.counted = True
                    json_vehicles["vehicles"].append({"vehicle": to.vehicle_type, "time": num_frame / 29.6, "direction":0})
                    meta_color = (0, 0, 255)
                    print("Detected: ", objectID)
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # ----------- DISPLAY -----------
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        if to.counted == True:

            text_color = (0,0,255)
        else:
            text_color = (0,255,0)

        if args["display"] == True or args["output"] is not None:
            text = "ID {}, {}, {}".format(objectID, vehicle_type[1], to.counted)

            cv2.putText(frame, text, (centroid[0] - 1, centroid[1] - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            cv2.circle(frame, (centroid[0], centroid[1]), 3, text_color, -1)


    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalgoing),
        ("Down", totalcoming),
        ("Status", status),
    ]
    # loop over the info tuples and draw them on our frame
    if args["display"] == True or args["output"] is not None:
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, imH - ((i * 20) + 150)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, tuple(clicks[0]), tuple(clicks[1]), meta_color, 3)

    if args["output"] is not None:
        # Write video output
        out.write(frame)

    # DISPLAY ALL THE ACTION ON SCREEN
    if args["display"] == True:
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer, save the json and display FPS information
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[DATA] Total number of cars coming: ",totalcoming)
print("[DATA] Total number of cars going: ",totalgoing)

try:
    with open('vehicles.json', 'w') as json_file:
        json.dump(json_vehicles, json_file)

    print("[INFO] Json saved. All data is safe.")
except:
    print("[ERROR] Json could not be saved.")


# Clean up
video.release()
if args["output"] is not None:
    out.release()
cv2.destroyAllWindows()
