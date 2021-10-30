# import the required libraries
import os
import cv2
import numpy as np
import warnings

warnings.simplefilter("ignore")
# For getting the absolute path of the project directory or folder
PROJECT_BASE_PATH = os.path.dirname(os.path.abspath(__file__))

'''
This function initialize and resolve the absolute path of the required classes-labels, trained-weights and network configuration files.
'''


def yolo_cofiguration():
    # Concatenate the absolute path of project and path of coco classes labels file
    coco_labels_path = "{PROJECT_BASE_PATH}/coco.names.txt" \
        "".format(PROJECT_BASE_PATH=PROJECT_BASE_PATH)
    # Read all the class labels of the file
    labels = open(coco_labels_path).read().strip().split("\n")
    # Concatenate the absolute path of project and path of the coco trained weight file
    weights_path = "{PROJECT_BASE_PATH}/yolov3.weights" \
        "" .format(PROJECT_BASE_PATH=PROJECT_BASE_PATH)
    # Concatenate the absolute path of project and path of yolov3 network configuration file
    cfg_path = "{PROJECT_BASE_PATH}/yolov3.cfg" \
        "" .format(PROJECT_BASE_PATH=PROJECT_BASE_PATH)
    return labels, cfg_path, weights_path


'''
This function deals with the initialization of yolov3 network, its pretrained weights and get prediction
of the images from the algorithm.
'''


def object_detection(image):
    # Call the function to get the absolute path of the configuration,labels and weights file
    LABELS, cfg_path, weights_path = yolo_cofiguration()
    # Load the YOLO object detector, which is trained on COCO-Dataset
    network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # Determine only output layers name
    ln = network.getLayerNames()
    ln = [ln[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    # 4-dimensional blob from image is created using blobFromImage. It give option of resizing and cropping of
    # image from center, subtract mean values, scale values by scale factor, swap Blue and Red channels.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    network.setInput(blob)
    # After constructing a blob from the input image and then we perform a forward
    layerOutputs = network.forward(ln)
    return layerOutputs, LABELS


'''
This function get the output from the object_detection function on the input image and parse the output data. It resolve the 
classs label, confidence score and bounding box for every detection,Finally it visualize the result on to the window.
'''


def object_recog(frame):
    # Initialize the detected bounding bboxes, confidences score of detection, and class-IDs list
    detected_bboxes = []
    detection_confidences = []
    detected_classIDs = []
    # Get the output of the input image by calling the object_detection
    layeroutputs, LABELS = object_detection(frame)
    # Iterate over each of the layer outputs
    for output in layeroutputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # Parse the detections on the basis of confidence score and class-id
            if confidence > 0.4 and classID == 9:
                frame_height, frame_width = frame.shape[:2]
                box = detection[0:4] * \
                    np.array([frame_width, frame_height,
                              frame_width, frame_height])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                detected_bboxes.append([x, y, int(width), int(height)])
                detection_confidences.append(float(confidence))
                detected_classIDs.append(classID)

    # Parse the furthur results by applying the NMS threshold with 0.3 score and 0.5 is the detection confidence value
    idxs = cv2.dnn.NMSBoxes(detected_bboxes, detection_confidences, 0.5, 0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (detected_bboxes[i][0], detected_bboxes[i][1])
            (w, h) = (detected_bboxes[i][2], detected_bboxes[i][3])
            # visualize and draw the result on the window screen
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = LABELS[detected_classIDs[i]]
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
    return frame
