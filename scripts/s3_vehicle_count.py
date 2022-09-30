# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import sys
import csv
import collections
import numpy as np
from s3_tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker() #No presenta buenos resultados con la data de SMCV.

# Initialize the videocapture object
cap = cv2.VideoCapture('data/raw/smcv_30s.mp4')
input_size = 320 #of the Convolutional Neural Network

# Confirm video file can be opened.
if cap.isOpened():
    #Baja resolución para que no sea tan lento
    cap_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
else:
    print("Could not open video")
    sys.exit()

# Detection confidence threshold
confThreshold = 0.2
nmsThreshold = 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Gate
middle_line_position = 240
left_line_position = middle_line_position - 40
right_line_position = middle_line_position + 40

# Store Coco Names in a list
classesFile = "commons/coco.names"
classNames = open(classesFile).read().strip().split('\n')
#print(classNames)
#print(len(classNames))

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'commons/yolov3-320.cfg'
modelWeigheights = 'commons/yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

#Para usar el GPU en vez del CPU

## net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #nvidia cuda
## net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) #nvidia cuda

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

# List for store vehicle count information
temp_left_list = []
temp_right_list = []
left_list = [0, 0, 0, 0]
right_list = [0, 0, 0, 0]

# Function to count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    #print(id, ix)
    cv2.putText(img,str(id), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Find the current position of the vehicle
    if (ix > left_line_position) and (ix < middle_line_position):
        if id not in temp_left_list:
            temp_left_list.append(id)

    elif (ix < right_line_position) and (ix > middle_line_position):
        if id not in temp_right_list:
            temp_right_list.append(id)

    elif ix > right_line_position:
        if id in temp_left_list:
            temp_left_list.remove(id)
            print(cap.get(cv2.CAP_PROP_POS_MSEC))
            left_list[index] = left_list[index] + 1

    elif ix < left_line_position:
        if id in temp_right_list:
            temp_right_list.remove(id)
            print(cap.get(cv2.CAP_PROP_POS_MSEC))
            right_list[index] = right_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score
        cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)


def realTime():
    while True:
        success, img = cap.read()
        img = cv2.resize(img,(0,0),None,0.5,0.5) #resize a la mitad 480, 544
        ih, iw, channels = img.shape
        #print(ih, iw)
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()] #antes decía (layersNames[i[0] – 1])
        # Feed data to the network
        outputs = net.forward(outputNames)

        # Find the objects from the network output
        postProcess(outputs,img)

        # Draw the crossing lines

        cv2.line(img, (middle_line_position, 0), (middle_line_position, ih), (255, 0, 255), 2)
        cv2.line(img, (left_line_position, 0), (left_line_position, ih), (0, 0, 255), 2)
        cv2.line(img, (right_line_position, 0), (right_line_position, ih), (0, 0, 255), 2)

        # Draw counting texts in the frame
        cv2.putText(img, "De la izquierda", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "De la derecha", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Carro:        "+str(left_list[0])+"     "+ str(right_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motocicleta:  "+str(left_list[1])+"     "+ str(right_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:          "+str(left_list[2])+"     "+ str(right_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Camion:       "+str(left_list[3])+"     "+ str(right_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        left_list.insert(0, "Left")
        right_list.insert(0, "Right")
        cwriter.writerow(left_list)
        cwriter.writerow(right_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    realTime()