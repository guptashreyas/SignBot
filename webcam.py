import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("best2.pt")  # Update with the correct path to your model

# Define a mapping from class indices to ASL letters (A-Z)
class_mapping = {
    0: 'A',  # Assuming class 0 corresponds to 'A'
    1: 'B',  # Assuming class 1 corresponds to 'B'
    2: 'C',  # Assuming class 2 corresponds to 'C'
    3: 'D',  # Assuming class 3 corresponds to 'D'
    4: 'E',  # Assuming class 4 corresponds to 'E'
    5: 'F',  # Assuming class 5 corresponds to 'F'
    6: 'G',  # Assuming class 6 corresponds to 'G'
    7: 'H',  # Assuming class 7 corresponds to 'H'
    8: 'I',  # Assuming class 8 corresponds to 'I'
    9: 'J',  # Assuming class 9 corresponds to 'J'
    10: 'K', # Assuming class 10 corresponds to 'K'
    11: 'L', # Assuming class 11 corresponds to 'L'
    12: 'M', # Assuming class 12 corresponds to 'M'
    13: 'N', # Assuming class 13 corresponds to 'N'
    14: 'O', # Assuming class 14 corresponds to 'O'
    15: 'P', # Assuming class 15 corresponds to 'P'
    16: 'Q', # Assuming class 16 corresponds to 'Q'
    17: 'R', # Assuming class 17 corresponds to 'R'
    18: 'S', # Assuming class 18 corresponds to 'S'
    19: 'T', # Assuming class 19 corresponds to 'T'
    20: 'U', # Assuming class 20 corresponds to 'U'
    21: 'V', # Assuming class 21 corresponds to 'V'
    22: 'W', # Assuming class 22 corresponds to 'W'
    23: 'X', # Assuming class 23 corresponds to 'X'
    24: 'Y', # Assuming class 24 corresponds to 'Y'
    25: 'Z'  # Assuming class 25 corresponds to 'Z'
}

# Start video capture from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Perform inference on the frame
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()  
            conf = box.conf[0].item()  
            cls = int(box.cls[0].item())  

            # Get the corresponding letter from the mapping
            letter = class_mapping.get(cls, "Unknown")

            # Set color based on confidence score (green for high confidence, red for low)
            color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)  
            
            # Draw rectangle and label on the frame
            label = f'Letter: {letter}, Conf: {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    # Display the frame with predictions
    cv2.imshow('Webcam Feed', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
