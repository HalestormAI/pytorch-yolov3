import cv2
import math
import torch


def load_image(img_path, yolo_dims):
    img = cv2.imread(img_path)

    img_h, img_w = img.shape[:2]
    h, w = yolo_dims
    resize_ratio = min(w / img_w, h / img_h)

    img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)

    tb_pad = (h - img.shape[0]) / 2
    rl_pad = (w - img.shape[1]) / 2

    img = cv2.copyMakeBorder(img,
                             math.floor(tb_pad), math.ceil(tb_pad),
                             math.floor(rl_pad), math.ceil(rl_pad), cv2.BORDER_CONSTANT, value=(128, 128, 128))

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = rgb_img.copy().transpose((2, 0, 1))
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img, rgb_img


def load_classnames(file_path):
    with open(file_path, 'r') as fh:
        return [l.strip() for l in fh.readlines()]


def handle_result(predictions, conf_threshold):
    confidence_mask = (predictions[:, :, 4] < conf_threshold).flatten()
    keep_predictions = predictions[:, torch.logical_not(confidence_mask), :].detach().clone()
    predictions[:, confidence_mask, :] = 0

    def box_hw_to_corners(x):
        """
        Remap bounding box from [cx, cy, h, w] -> [tlx, tly, brx, bry]
        :param x: Predictions with first 4 elements in axis 2 representing the bbox coords/shape
        :return: Predictions with bbox remapped as above
        """

        y_offset = x[:, :, 2] / 2
        x_offset = x[:, :, 3] / 2

        x_corners = x.detach().clone()
        x_corners[:, :, 0] = x_corners[:, :, 0] - x_offset
        x_corners[:, :, 1] = x_corners[:, :, 1] - y_offset
        x_corners[:, :, 2] = x_corners[:, :, 0] + x_offset
        x_corners[:, :, 3] = x_corners[:, :, 1] + y_offset
        return x_corners

    # predictions = box_hw_to_corners(predictions)
    # Could just use torchvision's NMS, but this is for learning after all :)

    return box_hw_to_corners(keep_predictions)
