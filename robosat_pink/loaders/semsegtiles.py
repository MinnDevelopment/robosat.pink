"""PyTorch-compatible datasets. Cf: https://pytorch.org/docs/stable/data.html """

import os
import numpy as np
import torch.utils.data

from robosat_pink.tiles import tiles_from_dir, tile_image_from_file, tile_label_from_file
from robosat_pink.da.core import to_normalized_tensor


class SemSegTiles(torch.utils.data.Dataset):
    def __init__(self, config, ts, root, mode):
        super().__init__()

        self.root = os.path.expanduser(root)
        self.config = config
        self.mode = mode

        assert mode == "train" or mode == "predict"

        ts = (ts, ts)

        num_channels = 0
        self.tiles = {}
        for channel in config["channels"]:
            path = os.path.join(self.root, channel["name"])
            self.tiles[channel["name"]] = [(tile, path) for tile, path in tiles_from_dir(path, xyz_path=True)]
            self.tiles[channel["name"]].sort(key=lambda tile: tile[0])
            num_channels += len(channel["bands"])

        self.shape_in = (num_channels,) + ts  # C,W,H
        self.shape_out = (len(config["classes"]),) + ts  # C,W,H

        if self.mode == "train":
            path = os.path.join(self.root, "labels")
            self.tiles["labels"] = [(tile, path) for tile, path in tiles_from_dir(path, xyz_path=True)]
            self.tiles["labels"].sort(key=lambda tile: tile[0])

    def __len__(self):
        return len(self.tiles[self.config["channels"][0]["name"]])

    def __getitem__(self, i):

        tile = None
        mask = None
        image = None

        for channel in self.config["channels"]:

            image_channel = None
            bands = None if not channel["bands"] else channel["bands"]

            if tile is None:
                tile, path = self.tiles[channel["name"]][i]
            else:
                assert tile == self.tiles[channel["name"]][i][0], "Dataset channel inconsistency"
                tile, path = self.tiles[channel["name"]][i]

            image_channel = tile_image_from_file(path, bands)

            assert image_channel is not None, "Dataset channel {} not retrieved: {}".format(channel["name"], path)
            image = np.concatenate((image, image_channel), axis=2) if image is not None else image_channel

        if self.mode == "train":
            assert tile == self.tiles["labels"][i][0], "Dataset mask inconsistency"
            mask = tile_label_from_file(self.tiles["labels"][i][1])
            assert mask is not None, "Dataset mask not retrieved"

            image, mask = to_normalized_tensor(self.config, self.shape_in[1:3], self.mode, image, mask)
            return image, mask, tile

        if self.mode == "predict":
            image = to_normalized_tensor(self.config, self.shape_in[1:3], self.mode, image)
            return image, torch.IntTensor([tile.x, tile.y, tile.z])
