from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320
import cv2

#Constants definition
PERSON_CLASS_ID = 0 
MOSAIC_SCALE = 0.1
CONFIDENCE_THRESHOLD = 0.5

#Person detection using modlib
def detection(frame, model):
    detections = frame.detections[(frame.detections.confidence > CONFIDENCE_THRESHOLD) & (frame.detections.class_id == PERSON_CLASS_ID)]
    labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

    return detections, labels

#Apply mosaic to detected person areas
def apply_privacymask(frame_data, detections):
    h, w, _ = frame_data.shape

    for box, score, _, _ in detections:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 *w), int(y1 *h), int(x2 *w), int(y2 *h)

        mosaic_w = x2 - x1 
        mosaic_h = y2 - y1
        
        roi = frame_data[y1:y2, x1:x2]

        #Apply mosaic by reducing and enlarging the image
        small_roi = cv2.resize(roi, (int(mosaic_w *MOSAIC_SCALE), int(mosaic_h *MOSAIC_SCALE)), interpolation=cv2.INTER_LINEAR)
        mosaic_roi = cv2.resize(small_roi, (mosaic_w,mosaic_h), interpolation = cv2.INTER_NEAREST)

        frame_data[y1:y2, x1:x2] = mosaic_roi

    return frame_data

#Execute person detection, mosaic processing, label drawing and display for each frame
def process_frame(frame, model, annotator):
    frame_data = frame.image

    #Person detection
    detections, labels  = detection(frame, model)

    #Apply mosaic
    frame_data = apply_privacymask(frame_data, detections)
    frame.image = frame_data        

    #Draw
    annotator.annotate_boxes(frame, detections, labels)
    frame.display()

def main():
    device = AiCamera() #Select device
    model = SSDMobileNetV2FPNLite320x320() #Select model
    device.deploy(model) #Deploy to IMX500

    #Innitialize annotator to display inference result
    annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

    #Start inference
    with device as stream:
        for frame in stream:
            process_frame(frame, model, annotator)

if __name__ == "__main__":
    main()
