from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound

frequency = 2500
duration = 1000

def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[5])
    eye = (A+B) / (2.0 * C)
    return eye

count = 0
eyeThresh = 0.3
eyeFrames = 48
shapePredictor = "shape_predictor_68_face_landmarks.dat"

cam = cv2.VideoCapture(0)
#cascade = 'haarcascade_frontalface_default.xml'
#detector = cv2.CascadeClassifier(cascade)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width = 450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)
    for rect in rects:
        shape = predictor(ggray,rect)
        shape = face_utils.shape_to_no(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEye = eyeAspectRatio(leftEye)
        rightEye = eyeAspectRatio(rightEye)

        eye = (leftEye + rightEye) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0,0,255))
        cv2.drawContours(frame, [rightEyeHull], -1, (0,0,255))

        if eye < eyeThresh:
            count += 1

        if count >= eyeFrames:
            cv2.putText(frame, "Drowsiness Detected",(10,30),
                        cv2.FONT_HESSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            winsound.Beep(frequency, duration)

        else:
            count = 0
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
