import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def displayImagesWithBoxesYolo(image_path, bounding_boxes):
    """
    Plots bounding boxes on an image.

    Args:
      image_path: Path to the image file.
      bounding_boxes: List of bounding boxes in the format (x_center, y_center, width, height).
    """
    print("ete")

    # Load the image
    image = np.array(Image.open(image_path))
    image=image[:,:,:,3]
    plt.imshow(image)
    print(image.shape)

    # Convert bounding boxes to xmin, ymin, xmax, ymax
    for x_center, y_center, width, height in bounding_boxes:
        xmin = int(x_center - width / 2)
        ymin = int(y_center - height / 2)
        xmax = int(x_center + width / 2)
        ymax = int(y_center + height / 2)

        # Plot the bounding box
        rect = plt.Rectangle(
            (xmin, ymin),
            width,
            height,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)

    plt.axis("off")
    plt.show()
