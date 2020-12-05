import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteLayer(nn.Module):
    def __init__(self, idx, start, end=None):
        super().__init__()
        self.start_idx = None
        self.end_idx = None
        self._set_start_end_indices(idx, start, end)

    def forward(self, output_map):
        start_tensor = output_map[self.start_idx]

        if self.end_idx is None:
            return start_tensor

        end_tensor = output_map[self.end_idx]
        return torch.cat((start_tensor, end_tensor), 1)

    def _set_start_end_indices(self, idx, start, end):
        # Positive indices are absolute pointers within the model.
        # Negative indices are relative to the current op.
        self.start_idx = start
        if self.start_idx < 0:
            self.start_idx = idx + self.start_idx

        if end is None:
            self.end_idx = None
            return

        self.end_idx = end
        if self.end_idx < 0:
            self.end_idx = idx + self.end_idx


class DetectionLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_dim = img_dim

    def forward(self, x):
        return self._decode(x.detach())

    # Based heavily on PyTorch Yolo tutorial code here:
    # https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/8264dfba39a866998b8936a24133f41f12bfbdb7/util.py#L47
    def _decode(self, x):
        batch_size = x.size(0)
        stride = self.img_dim // x.size(2)
        grid_size = self.img_dim // stride

        # Num params: tx, ty, th, tw, confidence, class_prob_0, ... class_prob_n
        num_params = 5 + self.num_classes

        # len(anchors) gives the number of detection heads we're going to use
        # Because torch is BCHW, we need to dimshuffle, force a copy and reshape to get the dimensions in the right order
        # This isn't necessary in TF as it's BHWC by default
        x = x.view(batch_size, num_params * len(self.anchors), grid_size * grid_size)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batch_size, grid_size * grid_size * len(self.anchors), num_params)

        # Anchors defined in terms of the original input image, need to scale down by stride
        anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]

        # Create the grid offsets
        grid = torch.arange(grid_size).type(torch.FloatTensor)
        a, b = torch.meshgrid(grid, grid)

        # PyTorch meshgrid works in ij indexing, numpy in xy
        a = a.transpose(0, 1)
        b = b.transpose(0, 1)
        xy_offset = torch.cat((a.reshape(-1, 1), b.reshape(-1, 1)), 1).repeat(1, len(anchors)).view(-1, 2).unsqueeze(0)
        xy_offset = xy_offset.to(x.device)

        # Calculate the bbox offsets
        x[:, :, :2] = torch.sigmoid(x[:, :, :2]) + xy_offset

        # Now calculate the heights based on the anchor priors
        anchors = torch.FloatTensor(anchors).to(x.device)

        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * anchors

        # Sigmoid on the classes
        x[:, :, 5:] = torch.sigmoid(x[:, :, 5:])

        # Sigmoid & scale the confidences
        x[:, :, 4] = torch.sigmoid(x[:, :, 4])
        x[:, :, :4] *= stride
        return x


class MaxPoolStride1(nn.Module):
    """
        Verbatim from https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/master/darknet.py
    """

    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class TinyYoloV3(nn.Module):
    def __init__(self, builder):
        super().__init__()
        self.builder = builder
        if not self.builder.is_built:
            self.builder.build()
        self.ops = self.builder.ops

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.builder.ops.to(*args, **kwargs)

    def eval(self):
        x = super().eval()
        self.builder.ops = self.builder.ops.eval()
        return x

    def disable_grad(self):
        for op in self.builder.ops:
            op.requires_grad_(False)

    def forward(self, x):
        # Need to cache outputs for route
        output_map = {}
        ops_done = {}
        detections = None
        for i, m in enumerate(self.builder.blocks):
            m_type = m['type']

            if m_type in ('convolutional', 'upsample', 'maxpool'):
                x = self.builder.ops[i](x)
            elif m_type == 'route':
                x = self.builder.ops[i](output_map)
            elif m_type == 'yolo':
                x = self.builder.ops[i](x.detach())
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)

            output_map[i] = x
            ops_done[i] = m_type
        return detections

    def load_weights(self, file_path):
        """
        Mostly verbatim from https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/master/darknet.py

        A few changes at the top to fit my implementation of the model.
        :param file_path:
        """
        # Open the weights file
        fp = open(file_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.builder.blocks)):
            config = self.builder.blocks[i]
            m_type = config["type"]

            # Only conv layers have weights to be loaded
            if m_type != "convolutional":
                continue

            sequential = self.builder.ops[i]

            use_batchnorm = bool(int(config.get('batch_normalize', 0)))

            conv = sequential[0]

            if use_batchnorm:
                bn = sequential[1]

                # Get the number of weights of Batch Norm Layer
                num_bn_biases = bn.bias.numel()

                # Load the weights
                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                # Cast the loaded weights into dims of model weights.
                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)

                # Copy the data to model
                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)

            else:
                # Number of biases
                num_biases = conv.bias.numel()

                # Load the weights
                conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                ptr = ptr + num_biases

                # reshape the loaded weights according to the dims of the model weights
                conv_biases = conv_biases.view_as(conv.bias.data)

                # Finally copy the data
                conv.bias.data.copy_(conv_biases)

            # Let us load the weights for the Convolutional layers
            num_weights = conv.weight.numel()

            # Do the same as above for weights
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights

            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)
