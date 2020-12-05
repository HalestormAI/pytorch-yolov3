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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classnames(file_path):
    with open(file_path, 'r') as fh:
        return [l.strip() for l in fh.readlines()]
