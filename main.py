import cv2
import matplotlib.pyplot as plt
import torch

import model
from layers import TinyYoloV3
from tut_code import write_results, nasty_output_code
from utils import (
    load_classnames,
    load_image)


def plot_predictions(predictions, classes, img):
    predictions = predictions.detach().cpu().numpy()

    for i in range(predictions.shape[1]):
        tlx, tly, brx, bry, conf = predictions[0, i, :5]

        cv2.rectangle(img, (tlx, tly), (brx, bry), (0, 255, 0), 1)

    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    cfg_file = "cfg/yolov3-tiny.cfg"
    builder = model.TinyYoloBuilder(cfg_file)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = TinyYoloV3(builder)
    model.to(device)

    model.load_weights(r"yolov3-tiny.weights")

    input_height = int(builder.hyper_params['height'])
    input_width = int(builder.hyper_params['width'])

    coco_names = load_classnames('datasets/coco.names')

    img = load_image(r"coco-cat-test.jpg", (input_height, input_width))

    img = img.to(device)
    with torch.no_grad():
        # # This is a hack for the fact that the ops inside the nested sequentials
        # # seem to be missing the no_grad context
        # model.disable_grad()
        predictions = model(img)

    output = model(img)

    print(builder)
