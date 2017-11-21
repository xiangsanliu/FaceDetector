import cv2
import gc
from train_data_set import Model

model = Model()
model.load_model(file_path="d:/face/me.face.model.h5")

color = (0, 255, 255)
cap = cv2.VideoCapture(0)

cascade_path = "D:/xiang/Downloads/Programs/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml"

while(cap.isOpened()):
    ok, frame = cap.read()

    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)

    faceRects = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))

    if len(faceRects)>0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            image = frame[y-10:y+h+10, x-10: x+w+10]
            cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)
            faceId = model.face_predict(image)
            print(faceId)
            if faceId == 0 :
                cv2.putText(frame, "xiang", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            elif faceId == 1:
                cv2.putText(frame, "wang", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else :
                cv2.putText(frame, "unknow", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            cv2.imshow("识别", frame)

            k = cv2.waitKey(10)
            if k& 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()