import cv2
import gc

from socket_test import Control
from train_data_set import Model



if __name__ == '__main__':
    control = Control()
    model = Model()
    model.load_model(file_path='C:/Users/xiang/Pictures/face/me.face.model.h5')

    color = (0, 255, 255)
    cap = cv2.VideoCapture(0)

    cascade_path = "haarcascade_frontalface_alt2.xml"

    while(cap.isOpened()):
        ok, frame = cap.read()

        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        cascade = cv2.CascadeClassifier(cascade_path)

        faceRects = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(16, 16))

        if len(faceRects)>0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                image = frame[y-10:y+h+10, x-10: x+w+10]
                cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, thickness=2)
                faceId = model.face_predict(image)
                print(faceId)
                if faceId == 0 :
                    cv2.putText(frame, "xiang", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    control.greenlight()
                    control.lockon()
                elif faceId == 1:
                    cv2.putText(frame, "nong", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    control.greenlight()
                    control.lockon()
                elif faceId == 2 :
                    cv2.putText(frame, "cheng", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    control.greenlight()
                    control.lockon()
                else :
                    cv2.putText(frame, "unknow", (x+30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    control.redlight()

        cv2.imshow("识别", frame)

        k = cv2.waitKey(10)
        if k& 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()