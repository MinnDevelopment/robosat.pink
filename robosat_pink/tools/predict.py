import os
from tqdm import tqdm

import numpy as np
import mercantile

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

from robosat_pink.core import load_config, load_module, check_classes, check_channels, make_palette, web_ui, Logs
from robosat_pink.tiles import tiles_from_dir, tile_label_to_file


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "predict", help="Predict masks, from given inputs and an already trained model", formatter_class=formatter_class
    )

    inp = parser.add_argument_group("Inputs")
    inp.add_argument("dataset", type=str, help="predict dataset directory path [required]")
    inp.add_argument("--checkpoint", type=str, required=True, help="path to the trained model to use [required]")
    inp.add_argument("--config", type=str, help="path to config file [required]")

    out = parser.add_argument_group("Outputs")
    out.add_argument("out", type=str, help="output directory path [required]")

    perf = parser.add_argument_group("Data Loaders")
    perf.add_argument("--workers", type=int, help="number of workers to load images [default: GPU x 2]")
    perf.add_argument("--bs", type=int, default=4, help="batch size value for data loader [default: 4]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def main(args):
    config = load_config(args.config)
    check_channels(config)
    check_classes(config)
    palette = make_palette(config["classes"][0]["color"], config["classes"][1]["color"])
    args.workers = torch.cuda.device_count() * 2 if torch.device("cuda") and not args.workers else args.workers

    log = Logs(os.path.join(args.out, "log"))

    if torch.cuda.is_available():
        log.log("RoboSat.pink - predict on {} GPUs, with {} workers".format(torch.cuda.device_count(), args.workers))
        log.log("(Torch:{} Cuda:{} CudNN:{})".format(torch.__version__, torch.version.cuda, torch.backends.cudnn.version()))
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        log.log("RoboSat.pink - predict on CPU, with {} workers".format(args.workers))
        device = torch.device("cpu")

    chkpt = torch.load(args.checkpoint, map_location=device)
    model_module = load_module("robosat_pink.models.{}".format(chkpt["nn"].lower()))
    nn = getattr(model_module, chkpt["nn"])(chkpt["shape_in"], chkpt["shape_out"]).to(device)
    nn = torch.nn.DataParallel(nn)
    nn.load_state_dict(chkpt["state_dict"])
    nn.eval()

    log.log("Model {} - UUID: {}".format(chkpt["nn"], chkpt["uuid"]))

    loader_module = load_module("robosat_pink.loaders.{}".format(chkpt["loader"].lower()))
    loader_predict = getattr(loader_module, chkpt["loader"])(config, chkpt["shape_in"][1:3], args.dataset, mode="predict")

    loader = DataLoader(loader_predict, batch_size=args.bs, num_workers=args.workers)
    assert len(loader), "Empty predict dataset directory. Check your path."

    with torch.no_grad():  # don't track tensors with autograd during prediction

        for images, tiles in tqdm(loader, desc="Eval", unit="batch", ascii=True):

            images = images.to(device)

            outputs = nn(images)
            probs = torch.nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                x, y, z = list(map(int, tile))
                mask = np.around(prob[1:, :, :]).astype(np.uint8).squeeze()
                tile_label_to_file(args.out, mercantile.Tile(x, y, z), palette, mask)

    if not args.no_web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "./"
        tiles = [tile for tile in tiles_from_dir(args.out)]
        web_ui(args.out, base_url, tiles, tiles, "png", template)
