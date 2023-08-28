from pypylon import pylon
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=int(10), n_init = 10)
import os
import time

model = YOLO('./best.pt')
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = frame[y,x,0]
        colorsG = frame[y,x,1]
        colorsR = frame[y,x,2]
        colors = frame[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)

cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB',mouseRGB)

cv2.namedWindow('YOLOv8 Inference', cv2.WND_PROP_FULLSCREEN)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
count = len(os.listdir('./video'))
out = cv2.VideoWriter(f'./video/output_{count}.avi', fourcc, 20.0, (1920,1080))
line_position = 270
line_position_max = 300
count = 0
del_ids_list = []
list_ = []
import numpy as np
fps_list = []

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        
        # Access the image data
        frame = converter.Convert(grabResult)
     

        frame = frame.GetArray()



        y_shape, x_shape,_  = frame.shape
        cv2.line(frame, (0,line_position), (x_shape, line_position), (255,255,0),5)

        start_ = time.time()
        results = model(frame, verbose = not True,  conf = 0.9)
        boxes = results[0].boxes.cpu().numpy()
        annotated_frame = frame.copy()
        bbs = []

        for i, box in enumerate(boxes):
            r = box.xyxy[0].astype(int)
            # cv2.circle(annotated_frame, (r[0], r[1]))
            center_y = r[1] + (r[3] - r[1])/2
            bbox = np.array(box.xywh[0].astype(float))
            if center_y >line_position:
                cv2.rectangle(annotated_frame, (r[0], r[1]), (r[2], r[3]), (0,255,0),5)
                bbs.append(([bbox[0], bbox[1], bbox[2], bbox[3]], float(box.conf), int(box.cls)))
        detect_time = time.time() - start_
        start_track = time.time()
        tracks=tracker.update_tracks(bbs,frame=annotated_frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            if track_id not in list_:
                list_.append(track_id)        

        count = len(list_)
        fps = time.time() - start_

        # print(len(tracks) ,del_ids_list)
        cv2.putText(annotated_frame, f'Total Number: {count}', (60,160), cv2.FONT_HERSHEY_COMPLEX_SMALL,7, (0,0,255),7)
        # cv2.putText(annotated_frame, f'Total Fps: {1/fps}', (60,360), cv2.FONT_HERSHEY_COMPLEX_SMALL,7, (0,0,255),7)

        # cv2.putText(annotated_frame, f'Total detect: {1/(detect_time)}', (60,560), cv2.FONT_HERSHEY_COMPLEX_SMALL,7, (0,0,255),7)
        # cv2.putText(annotated_frame, f'Total tracking: {1/(time.time() - start_track)}', (60,760), cv2.FONT_HERSHEY_COMPLEX_SMALL,7, (0,0,255),7)
        annotated_frame_ = cv2.resize(annotated_frame, (1920,1080)).copy()
        # if result:
        #     annotated_frame = result[0].plot()
        # else:
        #     annotated_frame = frame.copy()
        out.write(annotated_frame_)
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()
# out.releqqase()
cv2.destroyAllWindows()
