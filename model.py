import torch.nn as nn

from layers import (
    DetectionLayer,
    RouteLayer,
    MaxPoolStride1
)


# Shamelessly stolen from:
# https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/master/darknet.py
#
# Changes:
#  - Switched l/r strips to full strips to make parsing easier
def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].strip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.strip()
    blocks.append(block)

    return blocks


class TinyYoloBuilder(object):

    def __init__(self, config_file, input_depth=3):
        self.ops = nn.ModuleList()
        self._config_file = config_file
        self.hyper_params = None
        self._blocks = None

        self._input_depth = input_depth

        self._filter_history = []
        self.is_built = False

    def load_config(self):
        blocks = parse_cfg(self._config_file)
        self.hyper_params = blocks[0]
        self._blocks = blocks[1:]

    def build(self):
        self.load_config()
        for idx, b in enumerate(self._blocks):
            parser = self._get_parser(b)
            block, n_filters = parser(idx, b)
            self.update_filter_history(n_filters)
            self.ops.append(block)
        self.is_built = True

    @property
    def blocks(self):
        return self._blocks

    @property
    def current_in_filter(self):
        """
        Take the tail of the list of filters to get the input filter dimension
        for the next layer.

        If the history is empty, we return the input depth (nominally 3 for RGB
        images.

        :return: The current input filter dimension, or input_depth if this is
        the first layer.
        """
        if self._filter_history:
            return self._filter_history[-1]

        # The first filter input is 3, since RGB images have 3 channels
        return 3

    def update_filter_history(self, n):
        """
        Update the layerwise filter history with the most recent layer's result.

        If a layer does not change the number of filters, the parser will return
        None. In this case we simply duplicate the previous layer's filter count

        :param n The number of filters output by the most recent layer (or None
        if this hasn't changed
        """
        if n is None:
            self._filter_history.append(self.current_in_filter)
        else:
            self._filter_history.append(n)

    def _get_parser(self, block):
        parsers = {
            'yolo': self._parser_yolo,
            'route': self._parser_route,
            'convolutional': self._parser_conv,
            'maxpool': self._parser_maxpool,
            'upsample': self._parser_upsample,
        }

        if block['type'] in parsers:
            return parsers[block['type']]

        raise NotImplementedError(f"Cannot find parser for block type {block['type']}")

    def _parser_conv(self, idx, config):
        output = nn.Sequential()
        n_filters = int(config['filters'])
        stride = int(config['stride'])
        k_size = int(config['size'])
        padding = k_size // 2 if bool(int(config['pad'])) else 0

        use_batchnorm = bool(int(config.get('batch_normalize', 0)))
        use_bias = not use_batchnorm

        conv = nn.Conv2d(self.current_in_filter,
                         n_filters,
                         k_size,
                         stride,
                         padding,
                         bias=use_bias)

        # Match the layer names with the reference implementation so we can
        # load pretrained weights
        output.add_module(f"conv_{idx}", conv)

        if use_batchnorm:
            output.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(n_filters))

        if config['activation'] == 'leaky':
            # TODO: Ref has this hard-coded - should it be a param?
            act = nn.LeakyReLU(0.1, inplace=True)
            output.add_module(f"leaky_{idx}", act)

        # TODO: Other act funcs (see https://github.com/ultralytics/yolov3/issues/441)

        return output, n_filters

    def _parser_upsample(self, idx, _):
        up = nn.Upsample(scale_factor=2, mode="nearest")
        return up, None

    def _parser_maxpool(self, idx, config):
        stride = int(config['stride'])
        k_size = int(config['size'])
        if stride > 1:
            mp = nn.MaxPool2d(k_size, stride)
        else:
            mp = MaxPoolStride1(k_size)
        return mp, None

    def _parser_route(self, idx, config):
        layer_idx = [int(c.strip()) for c in config['layers'].split(',')]

        start = layer_idx[0]
        end = None if len(layer_idx) == 1 else layer_idx[1]

        # The RouteLayer will calculate the absolute indices based on the current
        # index during construction - these will be rt.start_idx, rt.end_idx
        rt = RouteLayer(idx, start, end)

        n_filters = self._filter_history[rt.start_idx]
        if end is not None:
            n_filters += self._filter_history[rt.end_idx]

        return rt, n_filters

    def _parser_yolo(self, idx, config):
        mask = [int(i.strip()) for i in config['mask'].split(',')]

        def parse_masked_anchors(raw_anchors, mask):
            indices = [int(a.strip()) for a in raw_anchors.split(",")]
            a = [indices[i:i + 2] for i in range(0, len(indices), 2)]
            return [a[i] for i in mask]

        anchors = parse_masked_anchors(config['anchors'], mask)
        det = DetectionLayer(anchors, int(config['classes']), int(self.hyper_params['height']))
        # TODO: Not sure None is the right output for this...
        return det, None
