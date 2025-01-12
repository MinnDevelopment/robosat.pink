import os
import shutil
import concurrent.futures as futures

import mercantile
from tqdm import tqdm

from robosat_pink.tiles import tiles_from_csv, tile_from_xyz
from robosat_pink.core import web_ui


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "subset", help="Filter images in a slippy map dir using a csv tiles cover", formatter_class=formatter_class
    )
    inp = parser.add_argument_group("Inputs")
    inp.add_argument("--dir", type=str, required=True, help="to XYZ tiles input dir path [required]")
    inp.add_argument("--cover", type=str, required=True, help="path to csv cover file to filter dir by [required]")
    inp.add_argument("--workers", type=int, help="number of workers [default: 4]")

    mode = parser.add_argument_group("Alternate modes, as default is to copy.")
    mode.add_argument("--move", action="store_true", help="move tiles from input to output")
    mode.add_argument("--delete", action="store_true", help="delete tiles listed in cover")

    out = parser.add_argument_group("Output")
    out.add_argument("out", type=str, nargs="?", default=os.getcwd(), help="output dir path [required for copy or move]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def main(args):
    assert args.out or args.delete, "out parameter is required"
    args.out = os.path.expanduser(args.out)
    if not args.workers:
        args.workers = 4 if os.cpu_count() >= 4 else os.cpu_count()

    print("RoboSat.pink - subset {} with cover {}, on CPU, with {} workers".format(args.dir, args.cover, args.workers))

    tiles = set(tiles_from_csv(os.path.expanduser(args.cover)))
    progress = tqdm(total=len(tiles), ascii=True, unit="tiles")
    with futures.ThreadPoolExecutor(args.workers) as executor:

        def worker(tile):

            if isinstance(tile, mercantile.Tile):
                _, src = tile_from_xyz(args.dir, tile.x, tile.y, tile.z)
                dst_dir = os.path.join(args.out, str(tile.z), str(tile.x))
            else:
                src = tile
                dst_dir = os.path.join(args.out, os.path.dirname(tile))

            assert os.path.isfile(src)
            dst = os.path.join(dst_dir, os.path.basename(src))

            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)

            if args.move:
                shutil.move(src, dst)

            elif args.delete:
                os.remove(src)

            else:
                shutil.copyfile(src, dst)

            return os.path.splitext(src)[1][1:]  # ext

        for extension in executor.map(worker, tiles):
            progress.update()

    if not args.no_web_ui and not args.delete:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "./"
        web_ui(args.out, base_url, tiles, tiles, extension, template)
