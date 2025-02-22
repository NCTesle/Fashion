import cv2
import os
from ultralytics import YOLO

# Load the trained model
model = YOLO("E:/College/miniproject/Fashion/Training/yolov8m_fashion_finetuning/weights/best.pt")

# Path to the folder containing test images
test_images_folder = "E:/College/miniproject/Fashion/Dataset/fash/test/images/"

# Get all image file names in the folder
image_files = [f for f in os.listdir(test_images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
image_files.sort()  # Sort files alphabetically (optional)

# Initialize image index
index = 0

while index < len(image_files):
    image_path = os.path.join(test_images_folder, image_files[index])

    # Run YOLO prediction
    results = model.predict(source=image_path, conf=0.5)  # Confidence is still set, but all detections are drawn

    # Read the image
    image = cv2.imread(image_path)

    # Draw bounding boxes on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show image with bounding boxes
    cv2.imshow("YOLOv8 Detection", image)

    # Wait for key press: 'n' for next image, 'q' to quit
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):  # Quit if 'q' is pressed
        break
    elif key == ord("n"):  # Move to the next image if 'n' is pressed
        index += 1

cv2.destroyAllWindows()
