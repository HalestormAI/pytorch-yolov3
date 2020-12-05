import torch

import model
from layers import TinyYoloV3
from utils import (
    load_classnames,
    load_image
)

if __name__ == "__main__":
    cfg_file = "cfg/yolov3-tiny.cfg"
    builder = model.TinyYoloBuilder(cfg_file)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # device = torch.device("cpu")
    model = TinyYoloV3(builder)
    model.to(device, )

    model.load_weights(r"yolov3-tiny.weights")

    input_height = int(builder.hyper_params['height'])
    input_width = int(builder.hyper_params['width'])

    coco_names = load_classnames('datasets/coco.names')

    img = load_image(r"coco-cat-test.jpg", (input_height, input_width))

    img = img.to(device, )

    output = model(img)

    print(builder)
