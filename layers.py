import torch
import torch.nn as nn


class RouteLayer(nn.Module):
    def __init__(self, idx, start, end=None):
        super(RouteLayer).__init__()
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
    def __init__(self, anchors):
        super(DetectionLayer).__init__()
        self.anchors = anchors

    def forward(self):
        raise NotImplementedError("Still TODO...")
