'''
The project is completed using OpenCv implementation of YOLO for object detection model.The description
 of every line of code is given in comment section
'''
# import the required libraries
import cv2
from yolo_utils import *

'''
This is main function where execution starts and it initialize the video catupre and read the frames of input video and pre-process
'''
if __name__ == "__main__":
    # Path of the input source
    source_path = "Video.mp4"
    # Here you are supposed to enter the path of video file, which you want to process.
    video_capture = cv2.VideoCapture(source_path)
    # Add a while loop to read the frame of the video or any other media
    while True:
        # Capture the frames of the videos
        ret, frame = video_capture.read()
        # checking whether frame is read or not
        if not ret:
            break
        # Resize of the input source
        frame = cv2.resize(frame, (608, 608))
        # Remove the noise from color images
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        # Call the object_recog function to get output on the frame
        frame = object_recog(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
