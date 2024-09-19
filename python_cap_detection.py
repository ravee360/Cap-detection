import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')  # Adjust path if needed

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

confidence_threshold = 0.5  # Adjust this value as needed

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform cap detection on the frame
    results = model(frame)
    
    # Flag to determine if cap is detected
    cap_detected = False

    # Process detection results
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            if confidence > confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extract bounding box coordinates
                class_id = int(box.cls[0].item())  # Get the class ID

                if class_id == 0:  # Assuming class ID 0 corresponds to 'cap'
                    # Set cap_detected to True
                    cap_detected = True

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Cap {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display text based on detection
    if cap_detected:
        cv2.putText(frame, 'Cap Detected', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Cap Not Detected', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with bounding boxes and text
    cv2.imshow('Cap Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
