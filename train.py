from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data=r"C:\Users\rudra\OneDrive\Desktop\DESKTOP\STUDY\KJSCE\INTERNSHIP\INHOUSE\2_NINAD MEHENDALE SIR\DAY-6\BottleDetection\data.yaml",
    epochs=50,
    imgsz=640
)