# Yolo-Traffic-Light-Detection
This project is based on detecting the Traffic light. 
- Pretained data is used.
- This application entertained both real time video or download video and also images.
- This is basic application and can be used for final year bachelor degree program.

#Steps or Guides to run the project.

- Create a Virtual environment
    $ python3 -m venv env

- Activate Virtual Environment

    $ source env/bin/activate

- Install required Packages

   $ pip install -r requirements.txt

- First need to download the pre-trained weights of the yolo-v3 from following given link, and place that file in the code folder

    https://pjreddie.com/media/files/yolov3.weights   

- Place the download weigts inside the project folder

- Testing For Image:
   
   For testing on the input image,Place the image inside the project folder, enter the path of the pic in the video and run the following command
   
    $ python3 test_images.py 

- Testing For Videos:
   
   For testing on the input video ,Place the video file inside the project folder and run the following command   
    
    $ python3 test_videos.py 


   