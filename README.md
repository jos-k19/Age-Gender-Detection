# PYCK PROJECT
# AGE-GENDER DETECTION (PYTHON-OpenCV-DNN)
## Introduction
Online search, shopping, payments, entertainment, social media, and many other technologies came into existence in the past 15-20 years. But it is observed that in the last 5 -6 years, companies are investing heavily in computer vision and machine learning technologies. But why? 
<br />
<br />
To know the needs and requirements of their user and customers more and more. And the more they fulfill their requirements, the more they build customer’s trust and loyalty. Computer vision can be used in securities, traffic control, self-drawing cars, etc. One can do hundreds of things using computer vision: object detection, face detection, age and gender detection, path detection, and many more. Among them, face detection is most valuable and uncomplicated. It is used in apps like Snapchat, identifying criminals in a crowd, security and video surveillance, and privacy of our smartphones. 
<br />
<br />
In this project, We have used Computer vision using Python-OpenCV and DNN. We implement CNN, Deep learning approaches, and OpenCV using Python to achieve robust age group and gender classification of faces. We used a dataset for face photos that include various real-world imaging conditions like noise, lighting, pose, and appearance. We classified people into male and female and different age groups. We have used various online materials for learning about this and implementing it in the most reliable way possible. 
<br />
<br />
## Code Implementation
### Following commands, files and functions were used in our project
* argparse: To take input from the user 
 ```pip install argparse ```
* --input: Provides the path to the input image for age detection
* --confidence: The minimum probability threshold to filter weak detections
*  imultis: Series of convenience functions to make basic image processing functions 
 ```pip install imutils ```
* blobFromImage: This function performs three significant tasks
    * Mean subtraction
    * Scaling
    * optionally channel swapping
* scale factor: After we perform mean subtraction, we can optionally scale our images by some factor using “scale factor.” 
* Size: It is used to supply the spatial size that the Convolutional Neural Network expects.
* Mean: These are our mean subtraction values. They can be a 3-tuple of the RGB means, or they can be a single value, in which case the supplied value is subtracted from 8 every channel of the image. 
* swapRB: OpenCV assumes images are in BGR channel order; however, the mean value assumes we are using RGB order. To remove this discrepancy, we can use swap
* detection: list of detected faces
* faceBlob: Blob generated after capturing region of interest(ROI)
* frame: A single frame from your webcam video stream
* faceNet: The initialized deep learning face detector
* ageNet: Our initialized deep learning age classifier
* genderNet: Our initialized deep learning gender classifier
* minConf: The confidence threshold to filter weak face detections
* Results: Final output that contains dictionary of age and gender predicted 
* imultis: Series of convenience functions to make basic image processing functions
* cv2.waitKey(): Keyboard binding function. 
* cv2.destroyAllWindows():  Destroy all windows that we created 
* readNet(): This method is used to load the networks
* .pb file: This is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model.
* .pb file: This is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model.
* .prototxt files: It describes the network configuration
* .caffemodel file: This defines the internal states of the parameters of the layers
* cv2.putText(): It is used to put text on the image detected
* os: To access desktop files 


### Command prompt commands to run the  python file
* For images: 
    * python {file_name.py} --input {image_address/image.extension}
* For video stream:
    * Python {file_name.py}
## Result:
 ```
    python AgeGender.py --input All_Images/image_02.jpg
    Gender : Male
    Age : (25-32)
 ```
 ![GitHub Logo](All_Results/result_01.png)
```
    python AgeGender.py --input All_Images/image_08.jpg
    Gender : Female
    Age : (25-32)
 ```
 ![GitHub Logo](All_Results/result_02.png)
 
 ```
    python AgeGender.py --input All_Images/image_10.jpg
    Gender : Female
    Age : (4-6)
 ```
 ![GitHub Logo](All_Results/result_05.png)
 ### Final result
 
 ![GitHub Logo](All_Results/final_result.jpg)
 
