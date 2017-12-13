import cv2
import os

def get_data(parent_dir, child_dir):
    cap = cv2.VideoCapture(0)       #打开视频流
    cascade_path = "haarcascade_frontalface_alt2.xml"
    num = 1
    items = os.listdir(parent_dir + child_dir)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  #读取视频的每一帧，并转换为灰度图，提高检测的效率
        cascade = cv2.CascadeClassifier(cascade_path)   #加载分类器
        faceRects = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(16, 16)) #识别出人脸,scaleFactor为图片缩放比例,minNeighbours为需要检测的有效点个数

        if len(faceRects) > 0:              #把检测出来的人脸框出来
            for faceRect in faceRects:
                x, y, w, h = faceRect
                img_name = '%s/%d.jpg'%(parent_dir+child_dir , num+items.__len__()) #指定保存图片的路径和文件名
                image = gray[y-10:y+h+10, x-10: x+w+10]            #截取人脸
                cv2.imwrite(img_name, image)    #保存图片的灰度图
                num += 1
                cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)        #框住人脸
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, '%d'%(num), (x, y-20), font, 1, (255, 255, 0), 2 )       #提示已拍摄人脸数量
        cv2.imshow("xiang", frame)

        if num>100:
            break

        k = cv2.waitKey(10)
        if k& 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_data("C:/Users/xiang/Pictures/face/", "unknow")
