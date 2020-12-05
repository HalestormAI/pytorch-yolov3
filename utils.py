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


def iou(src, tgts):
    # Intersection
    i_x0 = torch.max(src[0], tgts[:, 0])
    i_y0 = torch.max(src[1], tgts[:, 1])
    i_x1 = torch.min(src[2], tgts[:, 2])
    i_y1 = torch.min(src[3], tgts[:, 3])

    i_w = torch.clamp_min(i_x1 - i_x0, 0)
    i_h = torch.clamp_min(i_y1 - i_y0, 0)

    intersection_area = i_w * i_h

    # Union
    src_wh = src[2:4] - src[:2] + 1
    torch.clamp_min_(src_wh, 0)
    src_area = torch.prod(src_wh, 0)

    tgts_wh = tgts[:, 2:4] - tgts[:, :2] + 1
    torch.clamp_min_(tgts_wh, 0)
    tgts_areas = torch.prod(tgts_wh, 1)

    union_area = src_area + tgts_areas

    return intersection_area / (union_area - intersection_area)


def batchwise_nms(batch_sample, nms_threshold):
    # Replace the per-label confidence masks [80] for each bounding box with the confidence
    # score and best label for that bbox:
    #   [x0, y0, x1, y1, c1, c2, ..., c80] -> [x0, y0, x1, y1, l, lc]
    label_confidence, label = batch_sample[:, 5:].max(1, keepdim=True)
    prediction = torch.cat((batch_sample[:, :4], label, label_confidence), 1)

    # List of all classes detected in the batch
    classes = torch.unique(prediction[:, 4])

    keep = torch.empty((0, 6), dtype=prediction.dtype).to(prediction.device)

    # Now go through each of the unique classes, merge bounding boxes using NMS
    for cls in classes:
        # Filter predictions to those with class `cls`
        cls_prediction = prediction[prediction[:, 4] == cls, :]

        if cls_prediction.size(0) == 1:
            keep = torch.cat((keep, cls_prediction[0].unsqueeze(0)), 0)
            continue

        # Sort the detections so the most confident is at the end
        _, sort_idx = torch.sort(cls_prediction[:, 5])

        # Run through all the items remaining in the list to keep, get the
        # IOU between the most confident remaining for this class and the others
        while len(sort_idx) > 0:
            last = len(sort_idx) - 1
            i = sort_idx[last]

            keep = torch.cat((keep, cls_prediction[i, :].unsqueeze(0)), 0)

            iou_i = iou(cls_prediction[i].flatten(), cls_prediction[i + 1:, :])
            iou_mask = torch.nonzero(iou_i <= nms_threshold).flatten()

            # Only keep the indices for the bounding boxes that don't merge with
            # this one
            sort_idx = sort_idx[iou_mask]
            print(iou_i)
    return keep


def handle_result(predictions, conf_threshold=0.5, nms_threshold=0.6):
    def box_hw_to_corners(x):
        """
        Remap bounding box from [cx, cy, w, h] -> [tlx, tly, brx, bry]
        :param x: Predictions with first 4 elements in axis 2 representing the bbox coords/shape
        :return: Predictions with bbox remapped as above
        """

        x_offset = x[:, :, 2] / 2
        y_offset = x[:, :, 3] / 2

        # Clone the bbox proportions as we need to use x0/y0 more than once.
        z = predictions[:, :, :4].clone()
        z[:, :, 0] = x[:, :, 0] - x_offset
        z[:, :, 1] = x[:, :, 1] - y_offset
        z[:, :, 2] = x[:, :, 0] + x_offset
        z[:, :, 3] = x[:, :, 1] + y_offset
        x[:, :, :4] = z

        return x

    # Could just use torchvision's NMS, but this is for learning after all :)
    predictions = box_hw_to_corners(predictions)

    batch_size = predictions.shape[0]
    batched_output = [None] * batch_size
    for b in range(batch_size):
        # For all predictions in image b (for b in Batch), create a logical mask of those
        # above the confidence threshold
        confidence_mask = (predictions[b, :, 4] > conf_threshold).flatten()
        if torch.count_nonzero(confidence_mask) == 0:
            print(f"No objects found with sufficient confidence for batch {b}")
            continue

        # batch_sample is the subset of bounding box predictions for image b in the batch,
        # for which we have sufficient confidence
        batch_sample = predictions[b, confidence_mask, :]
        keep = batchwise_nms(batch_sample, nms_threshold)
        batched_output[b] = keep
    return batched_output
