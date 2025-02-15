from ultralytics import YOLO
det_model = YOLO(r"C:\Users\rameshpe\Downloads\Onyx\yolov11\best.pt")
det_model.to("cpu")
det_model.export(format="openvino", imgsz=640, dynamic=False, half=True)
