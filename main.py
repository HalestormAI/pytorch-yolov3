import model


if __name__ == "__main__":
    cfg_file = "cfg/yolov3-tiny.cfg"
    builder = model.TinyYoloBuilder(cfg_file)
    builder.build()

    print(builder)