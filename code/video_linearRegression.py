import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import math
import socket
import time
import pickle

### Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

### Load Model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


### Variables for UDP Send
# Set IP address as local host, 12000 is destination port
serverAddressPort = ("127.0.0.1", 12000)
bufferSize = 1024
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
message = ""

# Mapping Function
def mapping(x,input_min,input_max,output_min,output_max):
    val = (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min
    if val < output_min:
        val = output_min
    elif val > output_max:
        val = output_max
    return val

### Draw EDGES
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

### Vector List
vectorList = [
    [0,1],
    [0,2],
    [1,3],
    [2,4],
    [3,5],
    [0,6],
    [1,7],
    [6,7],
    [6,8],
    [7,9],
    [8,10],
    [9,10]
]

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 3, (0,255,0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


###
# Variables for drawing plot in real-time


if __name__ == "__main__":
    ### Variables
    numberOfPeople = 4
    lamdaVal = 0.885

    minBPD = 10.0
    maxBPD = 0.0

    ### Load model
    with open('./model/lr_model.pickle', 'rb') as f:
        lr = pickle.load(f)
    # lr = joblib.load('./model/lr_model.pkl')

    ### Loading Video File
    cap = cv2.VideoCapture('./data/sampleVideo.mp4')
    while cap.isOpened():
        ret, frame = cap.read()

        ### Variables for each frame
        BPD = []
        
        # Resize image
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
        input_img = tf.cast(img, dtype=tf.int32)
        
        # Detection section
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        keypoints_with_scores = keypoints_with_scores[:numberOfPeople]
        keypoints_only = np.delete(keypoints_with_scores,2,2)
        keypoints_only_body = np.delete(keypoints_only, [0,1,2,3,4], 1)

        # Calculate each vector for each person
        vectors_only_body = []
        for person in keypoints_only_body:
            tempPerson = []
            for i in vectorList:
                tempVector = person[i[1]] - person[i[0]]
                tempPerson.append(tempVector)
                # print(person[i[1]] - person[i[0]])    # vector(second - first)
                # print("====")
            vectors_only_body.append(tempPerson)

        vectors_only_body = np.array(vectors_only_body)
        vectors_only_body.reshape(4,12,2)
        # print(vectors_only_body)
        # print("============")

        ### Calculate BPD(Body-part-level Pose Distance)
        for person in vectors_only_body:
            pass

        for i in range(12):
            tempBodyPart = []
            tempD = []
            
            # Calculate vector
            for person in vectors_only_body:
                tempBodyPart.append(person[i])
                # print(person[i])
                # print("====")
            tempBodyPart = np.array(tempBodyPart)
            tempAverageVector = tempBodyPart.mean(axis = 0)
            
            # Calculate d
            for vi in tempBodyPart:
                tempDVal = np.linalg.norm(vi - tempAverageVector)
                tempD = np.array(tempD)
                tempD = np.append(tempD, tempDVal)
            
            
            BPD = np.array(BPD)
            BPD = np.append(BPD, math.pow(tempD.mean(), lamdaVal))
        
        ### Check each Value....
        print(BPD)
        sumBPD = np.sum(BPD)
        print("sumBPD: ", sumBPD)
        mapBPD = int(mapping(sumBPD, 0.2, 0.5, 0.0, 255.0))     # 0.2 ~ 0.5
        print("mapBPD: ", mapBPD)
        lrResult = lr.predict([[sumBPD]])[0][0]
        print("lrResult: ", lrResult)
        maplrResult = int(mapping(lrResult, 0.0, 1.0, 0.0, 255.0))
        print("maplrResult: ", maplrResult)

        # if (minBPD > sumBPD):
        #     minBPD = sumBPD
        # if (maxBPD < sumBPD):
        #     maxBPD = sumBPD
        # print("minBPD: ", minBPD)   # BPD Maximum Value
        # print("maxBPD: ", maxBPD)   # BPD Minimum Value
        print("===============")

        # Sending mapBPD to Processing....
        UDPClientSocket.sendto(str.encode(str(mapBPD)), serverAddressPort)
        # UDPClientSocket.sendto(str.encode(str(mapBPD)), serverAddressPort)


        # Drawing Colored Rectangle with mapBPD
        start_point = (0, 0)
        end_point = (30, 30)
        color = (0, 255-maplrResult, maplrResult)
        thickness = -1
        # frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        # frame = cv2.putText(frame, str(maplrResult), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # frame = cv2.rectangle(frame, (120,0), (150,30), (0, 255-mapBPD, mapBPD), thickness)
        frame = cv2.putText(frame, "Sum of BPD: " + str(round(sumBPD,2)), (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
        # frame = cv2.rectangle(frame, (120,30), (150,60), (0, 255-maplrResult, maplrResult), thickness)
        frame = cv2.putText(frame, "Regression Model Prediction: " + str(round(lrResult,2)), (150,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)


        # Render keypoints 
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
        # loop_through_people(frame, [keypoints_with_scores[0]], EDGES, 0.1)    # Check for first person.....

        #time.sleep(0.1)
        
        cv2.imshow('Movenet Multipose', frame)
        
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # ### Raw Webcam Feed
    # cap = cv2.VideoCapture(0)

    # while cap.isOpened():
    #     ret, frame = cap.read()
        
    #     # Resize image
    #     img = frame.copy()
    #     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
    #     input_img = tf.cast(img, dtype=tf.int32)
        
    #     # Detection section
    #     results = movenet(input_img)
    #     keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        
    #     # Render keypoints
    #     loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    #     cv2.imshow('Raw Webcam Feed', frame)
        
    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         break
            
    # cap.release()
    # cv2.destroyAllWindows()