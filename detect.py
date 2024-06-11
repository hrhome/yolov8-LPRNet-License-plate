from ultralytics import YOLO
import cv2

if __name__ =="__main__":
    # model = YOLO('yolov8n.pt')
    model = YOLO('best.pt')
    '''预测'''
    predict = model(r"C:\Users\CIH\Desktop\video\cut2.mp4",save_crop=True,save_frames=True,show=False,save=True,device="0   ")
    # test = model.val(split="test",save_dir="./output",save_crop=True)
    '''实时'''
    # cap = cv2.VideoCapture(0) # 0表示调用摄像头
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     results = model(frame,save_crop=True,save_frames=False,show=False,save=True)
    #     annotated_frame = results[0].plot()
    #     cv2.imshow('car predict', annotated_frame)
    #     if cv2.waitKey(1) == ord('q'):  # 按下q退出
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

