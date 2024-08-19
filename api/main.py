from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import onnxruntime as ort
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from io import BytesIO
import torchvision
import sys
import base64
import matplotlib.pyplot as plt
import os

sys.path.append("./yoloInference")
sys.path.append("./yoloInference/models")
sys.path.append("./yoloInference/utils")
from yoloInference.detect import detect_objects


app = FastAPI()

# Load models
fcn_model_path = "./bestModels/fcn.onnx"
seg_session = ort.InferenceSession(fcn_model_path)


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


def perform_inference(image, model_path, color_map=color_map_list):
    session = ort.InferenceSession(model_path)
    image = torch.tensor(np.expand_dims(np.array(image), axis=0), dtype=torch.float32)[
        :, :, :, :3
    ]
    resize = transforms.Resize((640, 640))
    image = image.permute(0, 3, 2, 1)
    image = resize(image)
    image = image.permute(0, 3, 2, 1) / 255

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.array(image)})
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


@app.get("/models")
async def list_models():
    return JSONResponse(
        content={
            "models": [
                {"name": "Segmentation Model", "type": "ONNX", "path": fcn_model_path},
                {
                    "name": "Bounding Box Model",
                    "type": "Pytorch",
                    "path": "./bestModels/yolo.pt",
                },
            ]
        }
    )


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    segmented, _ = perform_inference(image, fcn_model_path)

    segmented_image = Image.fromarray(segmented[0].numpy().astype("uint8"))

    buffered = BytesIO()
    segmented_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse(content={"image": img_str})


@app.post("/detect")
async def detect_bbox(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image_path = os.path.abspath("./image.png")
    image.save(image_path)
    output = detect_objects(image_path)
    if output == []:
        return JSONResponse(content={"boxes": [], "classes": [], "confidences": []})
    ### Extract and convert the data to JSON serializable types
    boxes = [list(map(float, x[1:5])) for x in output]  # Convert to list of floats
    classes = [int(x[0]) for x in output]  # Convert to list of integers
    scores = [float(x[-1]) for x in output]  # Convert to list of floats

    ## Output
    ### Return JSON serializable content
    return JSONResponse(
        content={
            "boxes": boxes,
            "classes": classes,
            "confidences": scores,
        }
    )


@app.post("/detect_image")
async def detect_bbox_image(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image_path = os.path.abspath("./image.png")
    image.save(image_path)
    output = detect_objects(image_path)
    if output == []:
        return JSONResponse(content={})
    ### Extract and convert the data to JSON serializable types
    boxes = [list(map(float, x[1:5])) for x in output]  # Convert to list of floats
    displayImagesWithBoxesYolo(
        image_path=image_path, bounding_boxes=boxes, outputPath="./image.png"
    )
    image = Image.open("./image.png")
    image = Image.fromarray(np.array(image).astype("uint8"))

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return JSONResponse(content={"image": img_str})


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=5000)
