import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import torch

import model
from layers import TinyYoloV3
from utils import (
    load_classnames,
    load_image, handle_result)


def plot_predictions(cv_images, classes, batch_predictions, palette=None):
    pad = 3
    for i, img in enumerate(cv_images):
        pred = batch_predictions[i]

        for j in range(pred.size(0)):
            class_id = int(pred[j, 4])
            class_nm = classes[class_id]

            colour = palette[class_id] if palette is not None else (0, 255, 0)

            tlx, tly, brx, bry, conf = pred[j, :5]

            (l_width, l_height), l_baseline = cv2.getTextSize(class_nm, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            l_width += 2 * pad + 1
            l_height += 2 * pad + 1

            # Draw the text background and write the text on top
            cv2.rectangle(img, (tlx, tly), (tlx + l_width, tly + l_height), (0, 0, 0), -1)
            cv2.putText(img, class_nm, (tlx + pad, tly + l_height - pad), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

            cv2.rectangle(img, (tlx, tly), (brx, bry), colour, 1)

        plt.imshow(img)
        plt.show()


def print_predictions(filenames, classes, batch_predictions):
    for i, fname in enumerate(filenames):
        pred = batch_predictions[i]

        print(f"File: {fname}:")
        for j in range(pred.size(0)):
            class_id = int(pred[j, 4])
            class_nm = classes[class_id]
            print(f"  {class_nm} [{pred[j, 5]:.4f}]")


if __name__ == "__main__":
    cfg_file = "cfg/yolov3-tiny.cfg"
    builder = model.TinyYoloBuilder(cfg_file)

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    model = TinyYoloV3(builder)
    model.to(device)

    model.load_weights(r"yolov3-tiny.weights")
    model.eval()

    input_height = int(builder.hyper_params['height'])
    input_width = int(builder.hyper_params['width'])

    coco_names = load_classnames('datasets/coco.names')

    num_channels = 3

    image_filenames = [
        r"coco-cat-test.jpg"
    ]

    raw_images = [None] * len(image_filenames)
    batch = torch.empty((len(image_filenames), num_channels, input_width, input_height), dtype=torch.float32)

    for b, image_fn in enumerate(image_filenames):
        batch[b, :], raw_images[b] = load_image(image_fn, (input_height, input_width))

    batch = batch.to(device)
    with torch.no_grad():
        # # This is a hack for the fact that the ops inside the nested sequentials
        # # seem to be missing the no_grad context
        # model.disable_grad()
        predictions = model(batch)

    palette = pkl.load(open("palette", "rb"))

    batch_predictions = handle_result(predictions, 0.5, 0.4)
    print_predictions(image_filenames, coco_names, batch_predictions)
    plot_predictions(raw_images, coco_names, batch_predictions, palette)

    print(builder)
