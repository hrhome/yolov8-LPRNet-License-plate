from ultralytics import YOLO

'''初始化模型'''
if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")
    model.train(data = r'/root/ultralytics-main/ultralytics/cfg/datasets/ccpd.yaml',epochs=1)
