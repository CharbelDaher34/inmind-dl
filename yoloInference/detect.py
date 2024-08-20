import torch
import time
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device

try:
    model = attempt_load("./bestModels/yolo.pt", map_location="cpu")
except Exception as e:
    model = attempt_load("../bestModels/yolo.pt")


def detect_objects(image_path, conf_thres=0.25, iou_thres=0.45, img_size=640):
    # Initialize
    device = select_device("")
    half = device.type != "cpu"
    output = []

    # Load model
    # model = attempt_load(weights_path, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)

    if half:
        model.half()

    # Load image
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride)

    # Get names
    names = model.module.names if hasattr(model, "module") else model.names

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )

    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            start_time = time.time()
            pred = model(img, augment=False)[0]
            end_time = time.time()

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=None, agnostic=False
        )

        # Process detections
        for i, det in enumerate(pred):
            p, s, im0, _ = path, "", im0s, getattr(dataset, "frame", 0)
            # p = Path(p)
            # txt_path = str(Path(output_path) / p.stem) + ".txt"
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                # with open(txt_path, "w") as f:
                for *xyxy, conf, cls in reversed(det):
                    if conf < conf_thres:
                        continue
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )
                    output.append([cls, *xywh, conf])
                    # line = (cls, *xywh, conf)
                    # f.write(("%g " * len(line)).rstrip() % line + "\n")

    return output, end_time - start_time


# Example usage:
# detect_objects('path/to/weights.pt', 'path/to/image.jpg', 'path/to/output/directory')
