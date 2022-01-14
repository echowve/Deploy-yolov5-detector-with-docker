import requests
import json
import cv2
import numpy as np
from utils.tools import Colors, names, plot_one_box
import time
from multiprocessing import Process

colors = Colors()  # create instance for 'from utils.plots import colors'
show_name_list = ["person"] # 只显示行人检索结果，所有支持的种类在names中定义

host_port="http://127.0.0.1:5001/detect"



def process(name):
    cap = cv2.VideoCapture(name)
    ret, frame = cap.read()
    while(ret):

        bytes = frame.tobytes()
        #向服务端发送请求并检测，返回检测结果
        files = {'files': ("", bytes, "", {})}
        result = requests.post(host_port, files=files, data={"shapex":frame.shape[0], "shapey":frame.shape[1]})
        detections = json.loads(result.text)["result"]
        detections= np.array(detections)
        time_pass = json.loads(result.text)["time"]

        for de in detections:
            xyxy=de[:4]
            conf, cls = de[4:]
            c = int(cls)
            if names[c] not in show_name_list: continue
            label = f'{names[c]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors(c, True), line_thickness=1)

        cv2.imshow(name, frame)
        print("time pass {:.2f} s".format(time_pass))
        cv2.waitKey(10)
        # time.sleep(1)

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows() 


if __name__=='__main__':
    p1=Process(target=process,args=('test.mp4',))
    # p2 = Process(target=process, args=('test2.mp4',))
    p1.start()
    # p2.start()
    p1.join()
    # p2.join()
    b=time.time()



