import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
# Open the default camera
cam = cv2.VideoCapture(1)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
model = YOLO('my_custom.pt')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

while True:
    ret, frame = cam.read()
    # lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # cl = clahe.apply(l)
    # limg = cv2.merge((cl, a, b))
    # final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # frame = final
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lwr = np.array([0, 0, 189])
    upp = np.array([255, 70, 255])
    img_mask = cv2.inRange(hsv, lwr, upp)
    
    frame = cv2.bitwise_and(frame, frame, mask=img_mask)    
    result = model.predict(frame)
    detections = sv.Detections.from_ultralytics(result[0])
    annotated_image = bounding_box_annotator.annotate(
    scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # Display the captured frame
    cv2.imshow('Camera', annotated_image)
    out.write(annotated_image)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()