import torch
import torchvision

def one_hot_encode_images(masks, color_map):
    batch_size, height, width, _ = masks.shape
    num_colors = color_map.shape[0]

    # Reshape masks and color_map for efficient comparison
    masks_reshaped = masks.reshape(batch_size, height * width, 1, 3)
    color_map_reshaped = color_map.view(1, 1, num_colors, 3)

    # Compare masks with color_map
    matches = torch.all(masks_reshaped == color_map_reshaped, dim=-1)

    # Create one-hot encoding
    one_hot = torch.zeros(batch_size, height, width, num_colors, dtype=torch.float32)
    one_hot.view(batch_size, height * width, num_colors)[matches] = 1

    return one_hot.permute(0, 3, 2, 1)


# Log images and predicted masks
def log_segmentation_results(writer, images, true_masks, predicted_masks, epoch):
    images_grid = torchvision.utils.make_grid(images)
    true_masks_grid = torchvision.utils.make_grid(true_masks)
    predicted_masks_grid = torchvision.utils.make_grid(predicted_masks)

    writer.add_image("Images", images_grid, epoch)
    writer.add_image("True Masks", true_masks_grid, epoch)
    writer.add_image("Predicted Masks", predicted_masks_grid, epoch)


# Example function to calculate metrics
def calculate_metrics(predicted_masks, true_masks):
    smooth = 1e-6
    predicted_masks = predicted_masks.int()
    true_masks = true_masks.int()
    intersection = (predicted_masks & true_masks).float().sum((1, 2))
    union = (predicted_masks | true_masks).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)

    dice = (
        2
        * intersection
        / (
            predicted_masks.float().sum((1, 2))
            + true_masks.float().sum((1, 2))
            + smooth
        )
    )

    tp = (predicted_masks & true_masks).float().sum((1, 2))
    fp = (predicted_masks & ~true_masks).float().sum((1, 2))
    fn = (~predicted_masks & true_masks).float().sum((1, 2))

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    return (
        iou.mean().item(),
        dice.mean().item(),
        precision.mean().item(),
        recall.mean().item(),
    )


# def map_one_hot_to_image(one_hot, color_map):
#     batch_size, height, width, num_colors = one_hot.shape

#     # Use argmax to find the index of the 1 in each one-hot vector
#     indices = torch.argmax(one_hot, dim=-1)

#     # Use the indices to select colors from the color map
#     output = color_map[indices]

#     return output


def map_one_hot_to_image(one_hot, color_map):
    batch_size, height, width, num_classes = one_hot.shape

    # Use argmax to find the index of the class with the highest score in each one-hot vector
    indices = torch.argmax(one_hot, dim=-1)  # shape: (batch_size, height, width)

    # Expand indices to have an extra dimension to match color_map shape
    indices = indices.unsqueeze(-1)  # shape: (batch_size, height, width, 1)

    # Convert color_map to a tensor if it isn't already
    color_map_tensor = torch.tensor(color_map, dtype=torch.uint8, device=one_hot.device)

    # Use indices to select colors from the color map
    output = color_map_tensor[
        indices.squeeze(-1)
    ]  # shape: (batch_size, height, width, 3)

    return output
