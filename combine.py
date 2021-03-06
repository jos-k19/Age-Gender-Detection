import cv2 as cv
import argparse
import os


def Face_det(net, frame, conf_threshold=0.7):
    frame_copy = frame.copy()
    h = frame_copy.shape[0]
    w = frame_copy.shape[1]
    # resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
    # returns a blob  after mean subtraction, normalizing, and channel swapping on our image
    blob = cv.dnn.blobFromImage(frame_copy, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)

    # function of the Inflater class is used to set input data for uncompression.
    net.setInput(blob)
    # will give Numpy ndarray as output which you can use it to plot box on the given input image
    detections = net.forward()
    bboxes = []
    # loop through shape - shape[2]
    # k = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # create the bounding boxes
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(h / 150)), 1)
        # print(k)
        # k+= 1
    return frame_copy, bboxes

# code starts

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

# save our files

faceProto = os.path.sep.join(["face_detector", "opencv_face_detector.pbtxt"])
faceModel = os.path.sep.join(["face_detector","opencv_face_detector_uint8.pb"])

ageProto = os.path.sep.join(["age_detector", "age_deploy.prototxt"])
ageModel = os.path.sep.join(["age_detector", "age_net.caffemodel"])

genderProto = os.path.sep.join(["gender_detector", "gender_deploy.prototxt"])
genderModel = os.path.sep.join(["gender_detector", "gender_net.caffemodel"])

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load our model
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


# Open a video file or an image file or a camera stream
v = args.input if args.input else 0
# object to capture a video/image
# for video v = 0
cap = cv.VideoCapture(v)
print(f"v:{v}")
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    hasFrame, frame = cap.read()
    if not hasFrame:
        # wait until window is not closed
        cv.waitKey()
        break

    frameFace, bboxes = Face_det(faceNet, frame)
    print(f'bboxes:{bboxes}')
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)
        # extract face from bounding boxes using x1, x2, y1 and y2
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("Gender Output : {}".format(genderPreds))
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                   cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)
