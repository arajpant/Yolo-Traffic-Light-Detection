
# import the required libraries
import cv2
from yolo_utils import *
'''
This is main function where execution starts and it read the input image and pre-process .
'''
if __name__ == "__main__":
    # Path of the input image
    image_path = "1.JPG"
    # Read of the input images
    image = cv2.imread(image_path)
    # Resize of the input images
    image = cv2.resize(image, (608, 608))
    # Remove the noise from color images
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
    # Call the object_recog function to get output on the input-image
    image = object_recog(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
