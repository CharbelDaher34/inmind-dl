import onnxruntime as ort
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torchvision
import sys
import base64
import matplotlib.pyplot as plt
import os
import cv2

sys.path.append("./yoloInference")
from yoloInference.detect import detect_objects
# Load models
seg_session = ort.InferenceSession("./bestModels/fcn.onnx")
# Color map for segmentation
color_map_list = torch.tensor(
    [
        [0, 0, 0],
        [25, 82, 255],
        [255, 25, 197],
        [140, 255, 25],
        [0, 0, 0],
        [226, 255, 25],
        [255, 197, 25],
        [140, 25, 255],
        [54, 255, 25],
        [25, 255, 82],
        [255, 111, 25],
    ]
)


def map_one_hot_to_image(one_hot, color_map=color_map_list):
    batch_size, height, width, num_colors = one_hot.shape
    indices = torch.argmax(one_hot, dim=-1)
    output = color_map[indices]
    return output


def perform_inference(image, color_map=color_map_list):
    image = torch.tensor(np.expand_dims(np.array(image), axis=0), dtype=torch.float32)[
        :, :, :, :3
    ]
    resize = transforms.Resize((640, 640))
    image = image.permute(0, 3, 2, 1)
    image = resize(image)
    image = image.permute(0, 3, 2, 1) / 255

    input_name = seg_session.get_inputs()[0].name
    outputs = seg_session.run(None, {input_name: np.array(image)})
    outputs = torch.tensor(outputs[0][0]).permute(1, 2, 0)
    outputs = torch.tensor(np.expand_dims(np.array(outputs), axis=0))
    outputs = map_one_hot_to_image(outputs, color_map)
    return outputs, image


def draw_boxes(image, boxes, classes, scores):
    draw = ImageDraw.Draw(image)
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"Class: {cls}, Score: {score:.2f}", fill="red")
    return image


def displayImagesWithBoxesYolo(image_path, bounding_boxes, outputPath=None, show=False):
    """
    Plots bounding boxes on an image using Pillow.

    Args:
      image_path: Path to the image file.
      bounding_boxes: List of bounding boxes in the format (x_center, y_center, width, height).
    """
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Convert bounding boxes to xmin, ymin, xmax, ymax and draw rectangles
    for x_center, y_center, width, height in bounding_boxes:
        xmin = int((x_center - width / 2) * image.width)
        ymin = int((y_center - height / 2) * image.height)
        xmax = int((x_center + width / 2) * image.width)
        ymax = int((y_center + height / 2) * image.height)

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    if show:
        image.show()
    if outputPath:
        image.save(outputPath)  # Replace with your desired filename and format

    return image


def process_frame(frame):
    # Resize frame to 640x640 and ensure it's RGB
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Remove alpha channel if present
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)

    # Perform YOLO detection
    temp_path = "./temp_frame.jpg"
    pil_image.save(temp_path)  # Save using Pillow in RGB format
    yolo_output = detect_objects(os.path.abspath(temp_path))[0]
    image_with_boxes = pil_image.copy()
    if yolo_output:
        boxes = [x[1:5] for x in yolo_output]
        image_with_boxes = displayImagesWithBoxesYolo(temp_path, boxes)
    os.remove(temp_path)

    # Convert image_with_boxes to numpy array (already in RGB)
    image_with_boxes_np = np.array(image_with_boxes)

    # Perform segmentation
    segmented_image, _ = perform_inference(pil_image)

    # Convert segmented image to numpy array (already in RGB)
    segmented_np = cv2.resize(
        segmented_image.squeeze().numpy().astype(np.uint8), (640, 640)
    )

    # Blend original frame with segmentation
    alpha = 0.3
    blended = cv2.addWeighted(frame, 1 - alpha, segmented_np, alpha, 0)

    return blended, image_with_boxes_np
# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    blended, side_by_side = process_frame(np.array(frame))

    # Display the results
    cv2.imshow("Blended", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    cv2.imshow("boxes", cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
