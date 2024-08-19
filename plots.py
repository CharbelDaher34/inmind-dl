import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def displayImagesWithBoxesXminXmaxYminYmax(image_path, bounding_boxes):
    """
    Displays an image with bounding boxes.

    Args:
        image_path: Path to the image file.
        bounding_boxes: List of bounding boxes in the format [id, xmin, ymin, xmax, ymax, occlusion].
    """
    bounding_boxes = list(bounding_boxes)
    # Load the image
    image = Image.open(image_path)
    plt.imshow(image)

    # Plot each bounding box
    for box in bounding_boxes:
        id, xmin, ymin, xmax, ymax, occlusion = box

        plt.gca().add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
        )
        plt.text(
            xmin,
            ymin,
            str(id),
            color="white",
            fontsize=2,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    plt.axis("off")  # Hide axes
    plt.show()


# def displayImagesWithBoxesYolo(image_path, bounding_boxes,outputPath=None):
#     """
#     Plots bounding boxes on an image.

#     Args:
#       image_path: Path to the image file.
#       bounding_boxes: List of bounding boxes in the format (x_center, y_center, width, height).
#     """

#     # Load the image
#     image = np.array(Image.open(image_path))[:, :, :3]
#     image = np.array(torch.tensor(image).permute(0, 1, 2))
#     plt.imshow(image)

#     # Convert bounding boxes to xmin, ymin, xmax, ymax
#     for x_center, y_center, width, height in bounding_boxes:

#         xmin = float(x_center - width / 2) * image.shape[0]
#         ymin = float(y_center - height / 2) * image.shape[1]
#         xmax = float(x_center + width / 2) * image.shape[0]
#         ymax = float(y_center + height / 2) * image.shape[1]

#         # Plot the bounding box
#         rect = plt.Rectangle(
#             (xmin, ymin),
#             xmax - xmin,
#             ymax - ymin,
#             fill=False,
#             edgecolor="red",
#             linewidth=2,
#         )
#         plt.gca().add_patch(rect)

#     plt.axis("off")
#     plt.show()
#     if outputPath:
#         plt.savefig(
#             outputPath
#         )  # Replace with your desired filename and format  # Replace with your desired filename and format


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
