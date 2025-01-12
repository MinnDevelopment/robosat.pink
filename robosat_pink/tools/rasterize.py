import os
import sys
import json
import collections

import numpy as np
from tqdm import tqdm

import mercantile
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.warp import transform
from supermercado import burntiles

import psycopg2

from robosat_pink.core import load_config, check_classes, make_palette, web_ui, Logs
from robosat_pink.tiles import tiles_from_csv, tile_label_to_file, tile_bbox


def add_parser(subparser, formatter_class):
    parser = subparser.add_parser(
        "rasterize", help="Rasterize GeoJSON or PostGIS features to tiles", formatter_class=formatter_class
    )

    inp = parser.add_argument_group("Inputs [either --postgis or --geojson is required]")
    inp.add_argument("--cover", type=str, help="path to csv tiles cover file [required]")
    inp.add_argument("--config", type=str, help="path to config file [required]")
    inp.add_argument("--type", type=str, required=True, help="type of feature to rasterize (e.g Building, Road) [required]")
    inp.add_argument("--pg", type=str, help="PostgreSQL dsn using psycopg2 syntax (e.g 'dbname=db user=postgres')")
    inp.add_argument("--sql", type=str, help="SQL to retrieve geometry features [e.g SELECT geom FROM your_table]")
    inp.add_argument("--geojson", type=str, nargs="+", help="path to GeoJSON features files")

    out = parser.add_argument_group("Outputs")
    out.add_argument("out", type=str, help="output directory path [required]")
    out.add_argument("--ts", type=int, default=512, help="output tile size [default: 512]")

    ui = parser.add_argument_group("Web UI")
    ui.add_argument("--web_ui_base_url", type=str, help="alternate Web UI base URL")
    ui.add_argument("--web_ui_template", type=str, help="alternate Web UI template path")
    ui.add_argument("--no_web_ui", action="store_true", help="desactivate Web UI output")

    parser.set_defaults(func=main)


def geojson_reproject(feature, srid_in, srid_out):
    """Reproject GeoJSON Polygon feature coords
       Inspired by: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe
    """

    if feature["geometry"]["type"] == "Polygon":
        xys = (zip(*ring) for ring in feature["geometry"]["coordinates"])
        xys = (list(zip(*transform(CRS.from_epsg(srid_in), CRS.from_epsg(srid_out), *xy))) for xy in xys)

        yield {"coordinates": list(xys), "type": "Polygon"}


def geojson_tile_burn(tile, features, srid, ts, burn_value=1):
    """Burn tile with GeoJSON features."""

    shapes = ((geometry, burn_value) for feature in features for geometry in geojson_reproject(feature, srid, 3857))

    bounds = tile_bbox(tile, mercator=True)
    transform = from_bounds(*bounds, ts, ts)

    try:
        return rasterize(shapes, out_shape=(ts, ts), transform=transform)
    except:
        return None


def main(args):

    if args.pg:
        if not args.sql:
            sys.exit("ERROR: With PostgreSQL db, --sql must be provided")

    if (args.sql and args.geojson) or (args.sql and not args.pg):
        sys.exit("ERROR: You can use either --pg or --geojson inputs, but only one at once.")

    config = load_config(args.config)
    check_classes(config)
    palette = make_palette(*[classe["color"] for classe in config["classes"]], complementary=True)
    burn_value = next(config["classes"].index(classe) for classe in config["classes"] if classe["title"] == args.type)
    if "burn_value" not in locals():
        sys.exit("ERROR: asked type to rasterize is not contains in your config file classes.")

    args.out = os.path.expanduser(args.out)
    os.makedirs(args.out, exist_ok=True)
    log = Logs(os.path.join(args.out, "log"), out=sys.stderr)

    def geojson_parse_polygon(zoom, srid, feature_map, polygon, i):

        try:
            if srid != 4326:
                polygon = [xy for xy in geojson_reproject({"type": "feature", "geometry": polygon}, srid, 4326)][0]

            for i, ring in enumerate(polygon["coordinates"]):  # GeoJSON coordinates could be N dimensionals
                polygon["coordinates"][i] = [[x, y] for point in ring for x, y in zip([point[0]], [point[1]])]

            if polygon["coordinates"]:
                for tile in burntiles.burn([{"type": "feature", "geometry": polygon}], zoom=zoom):
                    feature_map[mercantile.Tile(*tile)].append({"type": "feature", "geometry": polygon})

        except ValueError:
            log.log("Warning: invalid feature {}, skipping".format(i))

        return feature_map

    def geojson_parse_geometry(zoom, srid, feature_map, geometry, i):

        if geometry["type"] == "Polygon":
            feature_map = geojson_parse_polygon(zoom, srid, feature_map, geometry, i)

        elif geometry["type"] == "MultiPolygon":
            for polygon in geometry["coordinates"]:
                feature_map = geojson_parse_polygon(zoom, srid, feature_map, {"type": "Polygon", "coordinates": polygon}, i)
        else:
            log.log("Notice: {} is a non surfacic geometry type, skipping feature {}".format(geometry["type"], i))

        return feature_map

    if args.geojson:

        tiles = [tile for tile in tiles_from_csv(os.path.expanduser(args.cover))]
        assert tiles, "Empty cover"

        zoom = tiles[0].z
        assert not [tile for tile in tiles if tile.z != zoom], "Unsupported zoom mixed cover. Use PostGIS instead"

        feature_map = collections.defaultdict(list)

        log.log("RoboSat.pink - rasterize - Compute spatial index")
        for geojson_file in args.geojson:

            with open(os.path.expanduser(geojson_file)) as geojson:
                feature_collection = json.load(geojson)

                try:
                    crs_mapping = {"CRS84": "4326", "900913": "3857"}
                    srid = feature_collection["crs"]["properties"]["name"].split(":")[-1]
                    srid = int(srid) if srid not in crs_mapping else int(crs_mapping[srid])
                except:
                    srid = int(4326)

                for i, feature in enumerate(tqdm(feature_collection["features"], ascii=True, unit="feature")):

                    if feature["geometry"]["type"] == "GeometryCollection":
                        for geometry in feature["geometry"]["geometries"]:
                            feature_map = geojson_parse_geometry(zoom, srid, feature_map, geometry, i)
                    else:
                        feature_map = geojson_parse_geometry(zoom, srid, feature_map, feature["geometry"], i)
        features = args.geojson

    if args.pg:

        conn = psycopg2.connect(args.pg)
        db = conn.cursor()

        assert "limit" not in args.sql.lower(), "LIMIT is not supported"
        db.execute("SELECT ST_Srid(geom) AS srid FROM ({} LIMIT 1) AS sub".format(args.sql))
        srid = db.fetchone()[0]
        assert srid, "Unable to retrieve geometry SRID."

        if "where" not in args.sql.lower():  # TODO: Find a more reliable way to handle feature filtering
            args.sql += " WHERE ST_Intersects(tile.geom, geom)"
        else:
            args.sql += " AND ST_Intersects(tile.geom, geom)"
        features = args.sql

    log.log("RoboSat.pink - rasterize - rasterizing {} from {} on cover {}".format(args.type, features, args.cover))
    with open(os.path.join(os.path.expanduser(args.out), "instances.cover"), mode="w") as cover:

        for tile in tqdm(list(tiles_from_csv(os.path.expanduser(args.cover))), ascii=True, unit="tile"):

            geojson = None

            if args.pg:

                w, s, e, n = tile_bbox(tile)

                query = """
                WITH
                  tile AS (SELECT ST_Transform(ST_MakeEnvelope({},{},{},{}, 4326), {}) AS geom),
                  geom AS (SELECT ST_Intersection(tile.geom, sql.geom) AS geom FROM tile CROSS JOIN LATERAL ({}) sql),
                  json AS (SELECT '{{"type": "Feature", "geometry": '
                         || ST_AsGeoJSON((ST_Dump(ST_Transform(ST_Force2D(geom.geom), 4326))).geom, 6)
                         || '}}' AS features
                        FROM geom)
                SELECT '{{"type": "FeatureCollection", "features": [' || Array_To_String(array_agg(features), ',') || ']}}'
                FROM json
                """.format(
                    w, s, e, n, srid, args.sql
                )

                db.execute(query)
                row = db.fetchone()
                try:
                    geojson = json.loads(row[0])["features"] if row and row[0] else None
                except Exception:
                    log.log("Warning: Invalid geometries, skipping {}".format(tile))
                    conn = psycopg2.connect(args.pg)
                    db = conn.cursor()

            if args.geojson:
                geojson = feature_map[tile] if tile in feature_map else None

            if geojson:
                num = len(geojson)
                out = geojson_tile_burn(tile, geojson, 4326, args.ts, burn_value)

            if not geojson or out is None:
                num = 0
                out = np.zeros(shape=(args.ts, args.ts), dtype=np.uint8)

            tile_label_to_file(args.out, tile, palette, out)
            cover.write("{},{},{}  {}{}".format(tile.x, tile.y, tile.z, num, os.linesep))

    if not args.no_web_ui:
        template = "leaflet.html" if not args.web_ui_template else args.web_ui_template
        base_url = args.web_ui_base_url if args.web_ui_base_url else "./"
        tiles = [tile for tile in tiles_from_csv(args.cover)]
        web_ui(args.out, base_url, tiles, tiles, "png", template)
