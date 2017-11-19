import cv2
import sys

def CatchVideo():
    cap = cv2.VideoCapture(0)
    classfier = cv2.CascadeClassifier("D:/xiang/Downloads/Programs/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml")
    num = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  #转换为灰度图，提高检测的效率
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(16, 16)) #检测出的人脸保存在faceRects列表中

        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                img_name = '%s/%d.jpg'%("D:/pic/xiangjianjian", num) #指定保存图片的路径和文件名
                image = frame[y-10:y+h+10, x-10: x+w+10]            #截取人脸
                cv2.imwrite(img_name, image)
                num += 1

                cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d'%(num), (x+30, y+30), font, 1, (255, 0, 255), 4)

        cv2.imshow("xiang", frame)

        c = cv2.waitKey(10)
        if num>500:
            break

    cap.release()
    cv2.destroyAllWindows()

CatchVideo()