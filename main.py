#Import required libraries:
import cv2
import numpy as np
import tensorflow as tf

#Load YOLOv3-tiny model:
model = tf.keras.models.load_model('path_to_yolov3_tiny_model')

#Load and preprocess the image:
image_path = 'path_to_input_image'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (416, 416))
image = image / 255.0
image = np.expand_dims(image, axis=0)

#Perform object detection using the YOLOv3-tiny model:
predictions = model.predict(image)

#Post-process the predictions:
def post_process(predictions, confidence_threshold=0.5):
    boxes = []
    for pred in predictions:
        for box in pred:
            if box[4] >= confidence_threshold:
                x, y, width, height = box[:4]
                x = int(x * image.shape[1])
                y = int(y * image.shape[0])
                width = int(width * image.shape[1])
                height = int(height * image.shape[0])
                boxes.append((x, y, width, height))
    return boxes

detected_boxes = post_process(predictions)

#Draw bounding boxes on the image:
for box in detected_boxes:
    x, y, width, height = box
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

#Display the result:
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
