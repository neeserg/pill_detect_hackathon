from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv

app = Flask(__name__)
WEBCAM_NUM =1 ## CHANGE THIS to swap webcam in mac.
_number_of_detected_objects = 0
cam = cv2.VideoCapture(WEBCAM_NUM)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

def get_annotated_image(picture,detections):
    annotated_image = bounding_box_annotator.annotate(
        scene=picture, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)
    return annotated_image

def update_number_of_detected_objects(detections):
    global _number_of_detected_objects
    _number_of_detected_objects = len(detections.class_id)


def get_prediction(frame,model):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lwr = np.array([0, 0, 130])
    upp = np.array([255, 100, 255])
    img_mask = cv2.inRange(hsv, lwr, upp)
    preprocessed_image = cv2.bitwise_and(frame, frame, mask=img_mask)
    result = model.predict(preprocessed_image)
    return result,preprocessed_image



def gen_frames():  # generate frame by frame from camera
    model = YOLO('my_custom.pt')
    while True:
        # Capture frame-by-frame
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        result,preprocessed_image = get_prediction(frame,model)
        detections = sv.Detections.from_ultralytics(result[0])
        update_number_of_detected_objects(detections)
        annotated_image = get_annotated_image(preprocessed_image,detections)
        ret, buffer = cv2.imencode('.jpg', annotated_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/count_feed')
def count_feed():
    def generate():
        yield str(_number_of_detected_objects)
    return Response(generate(), mimetype='text')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html',number_of_detected_objects=_number_of_detected_objects)


if __name__ == '__main__':
    app.run(debug=True)