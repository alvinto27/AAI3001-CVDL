"""Freeze RAISE/DIV2K identities from existing records; never replace files.

Run on the team VM with --repo-root pointing at its read-only data checkout
and --output-root pointing at a new scratch directory. Outputs use UTF-8,
LF newlines, fixed column order, and stable identity order. No images are
written, RAW files are not decoded, and source records are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

PHOTO_FIELDS = (
    "source", "source_type", "photo_id", "photo_index", "master_path",
    "camera_make", "camera_model", "source_height", "source_width",
    "noise", "grid_policy", "overlap512_rows", "overlap512_cols",
    "overlap256_rows", "overlap256_cols", "overlap512_y", "overlap512_x",
    "overlap256_y", "overlap256_x", "legacy512_policy", "legacy512_rows",
    "legacy512_cols", "legacy512_height", "legacy512_width", "split",
)
TILE_FIELDS = (
    "source", "cover_id", "cover_index", "photo_id", "tile_row", "tile_col",
    "pixel_y", "pixel_x", "clean_path", "stego_path", "split",
)
SPLIT_FIELDS = ("photo_id", "master_path", "split")
OUTPUTS = (
    "data/raise/photo_manifest.csv", "data/raise/legacy_tile_manifest.csv",
    "data/div2k/photo_manifest.csv", "data/div2k/split_seed0.csv",
)
EXPECTED_RAISE = {
    ("NIKON D7000", "landscape"): (570, 3280, 4948, 54),
    ("NIKON D7000", "portrait"): (144, 4948, 3280, 54),
    ("NIKON D90", "landscape"): (194, 2868, 4310, 40),
    ("NIKON D90", "portrait"): (84, 4310, 2868, 40),
    ("NIKON D40", "landscape"): (6, 2014, 3039, 15),
    ("NIKON D40", "portrait"): (2, 3039, 2014, 15),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path, required: set[str], *, comments: bool = False) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        lines = (line for line in stream if not (comments and line.startswith("#")))
        reader = csv.DictReader(lines)
        require(required <= set(reader.fieldnames or ()), f"missing columns in {path}")
        rows = list(reader)
    require(bool(rows), f"empty CSV: {path}")
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            f"malformed CSV rows: {path}")
    return rows


def unique(rows: list[dict[str, str]], key: str, name: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        require(row[key] not in result, f"duplicate {key} in {name}: {row[key]}")
        result[row[key]] = row
    return result


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    require(bool(value) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in value, f"unsafe relative path: {value!r}")
    return value


def starts(length: int, tile: int) -> list[int]:
    require(tile > 0 and length >= tile, f"side {length} cannot support tile {tile}")
    count = (length + tile - 1) // tile
    return [i * tile for i in range(count - 1)] + [length - tile]


def photo(source: str, ident: str, index: int, master: str, h: int, w: int,
          split: str, make: str = "", model: str = "") -> dict:
    require(split in {"train", "val", "test"}, f"invalid split: {split}")
    row = dict(source=source, source_type="nef" if source == "raise" else "png8",
               photo_id=ident, photo_index=index, master_path=safe_relative(master),
               camera_make=make, camera_model=model, source_height=h, source_width=w,
               noise="disabled" if source == "raise" else "not_applicable",
               grid_policy="overlap_last", split=split)
    for tile in (512, 256):
        ys, xs = starts(h, tile), starts(w, tile)
        row[f"overlap{tile}_rows"], row[f"overlap{tile}_cols"] = len(ys), len(xs)
        row[f"overlap{tile}_y"] = ";".join(map(str, ys))
        row[f"overlap{tile}_x"] = ";".join(map(str, xs))
    row.update(legacy512_policy="", legacy512_rows="", legacy512_cols="",
               legacy512_height="", legacy512_width="")
    if source == "raise":
        row.update(legacy512_policy="topleft_crop", legacy512_rows=h // 512,
                   legacy512_cols=w // 512, legacy512_height=h // 512 * 512,
                   legacy512_width=w // 512 * 512)
    return row


def build_raise(root: Path, tiles_manifest: Path) -> tuple[list[dict], list[dict]]:
    selection = unique(read_csv(root / "data/raise/selection.csv", {"File"}, comments=True),
                       "File", "selection")
    require(len(selection) == 1000, "RAISE selection must contain 1,000 photos")
    dataset = root / "Datasets/raise512-run1"
    manifest = unique(read_csv(tiles_manifest, {
        "tile_file", "source_file", "source_format", "source_height", "source_width",
        "camera_make", "camera_model", "tile_row", "tile_col", "pixel_x", "pixel_y",
    }), "tile_file", "preparation manifest")
    require(len(manifest) == 49796, "preparation manifest must contain 49,796 tiles")
    metadata = unique(read_csv(dataset / "metadata.csv", {
        "file", "label", "cover_id", "cover_index", "variant", "source_file",
        "height", "width", "channels",
    }), "file", "metadata")
    labels = unique(read_csv(dataset / "labels.csv", {
        "file", "label", "cover_id", "cover_index", "variant", "source_file",
        "source_photo", "camera_make", "camera_model", "tile_row", "tile_col", "x", "y",
    }), "file", "labels")
    splits = unique(read_csv(root / "data/raise/split_seed0.csv", {"file", "split"}),
                    "file", "RAISE splits")
    require(len(metadata) == 99592 and metadata.keys() == labels.keys() == splits.keys(),
            "metadata, labels, and splits must cover exactly the same 99,592 images")
    pairs: dict[int, dict[str, dict]] = defaultdict(dict)
    photo_info: dict[str, tuple] = {}
    photo_splits: dict[str, set[str]] = defaultdict(set)
    tile_covers: dict[str, set[int]] = defaultdict(set)
    for file, meta in metadata.items():
        safe_relative(file)
        role = meta["label"]
        require(role in {"clean", "stego"}, f"invalid metadata label: {role}")
        label = labels[file]
        tile_name = meta["source_file"]
        require(tile_name in manifest, f"no preparation row for {tile_name}")
        tile = manifest[tile_name]
        source = tile["source_file"]
        ident = Path(source).stem
        require(ident in selection and source == f"{ident}.NEF", f"unselected source: {source}")
        require(tile["source_format"] == "raw", f"non-RAW RAISE source: {source}")
        for field in ("cover_id", "cover_index", "variant"):
            require(label[field] == meta[field], f"label/metadata {field} mismatch: {file}")
        require(label["label"] == ("0" if role == "clean" else "1"), f"label mismatch: {file}")
        for label_field, expected in {
            "source_file": source, "source_photo": ident,
            "camera_make": tile["camera_make"], "camera_model": tile["camera_model"],
            "tile_row": tile["tile_row"], "tile_col": tile["tile_col"],
            "x": tile["pixel_x"], "y": tile["pixel_y"],
        }.items():
            require(label[label_field] == expected, f"label/preparation mismatch: {file}/{label_field}")
        require((meta["height"], meta["width"], meta["channels"]) == ("512", "512", "3"),
                f"invalid legacy tile shape: {file}")
        require(meta["variant"] == ("" if role == "clean" else "1"), f"invalid variant: {file}")
        index = int(meta["cover_index"])
        require(role not in pairs[index], f"duplicate role for cover {index}")
        require(meta["cover_id"] and int(meta["cover_id"]) == index, f"invalid cover_id: {file}")
        pairs[index][role] = meta
        tile_covers[tile_name].add(index)
        split = splits[file]["split"]
        require(split in {"train", "val", "test"}, f"invalid RAISE split: {file}")
        photo_splits[ident].add(split)
        info = (int(tile["source_height"]), int(tile["source_width"]),
                tile["camera_make"], tile["camera_model"])
        require(ident not in photo_info or photo_info[ident] == info, f"inconsistent source: {ident}")
        photo_info[ident] = info
    require(photo_info.keys() == selection.keys(), "missing RAISE photo records")
    require(tile_covers.keys() == manifest.keys() and all(len(v) == 1 for v in tile_covers.values()),
            "each preparation tile must map to exactly one original cover")
    require(len(pairs) == 49796, "wrong legacy pair count")
    photos, tiles = [], []
    for index, ident in enumerate(sorted(selection)):
        require(len(photo_splits[ident]) == 1, f"photo split leakage: {ident}")
        h, w, make, model = photo_info[ident]
        photos.append(photo("raise", ident, index, f"ImageSets/RAISE/NEF/{ident}.NEF",
                            h, w, next(iter(photo_splits[ident])), make, model))
    by_photo = {p["photo_id"]: p for p in photos}
    positions: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for index, pair in sorted(pairs.items()):
        require(set(pair) == {"clean", "stego"}, f"incomplete pair {index}")
        clean, stego = pair["clean"], pair["stego"]
        require(clean["source_file"] == stego["source_file"] and clean["cover_id"] == stego["cover_id"],
                f"mismatched twins: {index}")
        tile = manifest[clean["source_file"]]
        ident = Path(tile["source_file"]).stem
        r, c = int(tile["tile_row"]), int(tile["tile_col"])
        y, x = int(tile["pixel_y"]), int(tile["pixel_x"])
        require((y, x) == (r * 512, c * 512), f"invalid legacy start: {index}")
        require((r, c) not in positions[ident], f"duplicate position: {ident}/{r}/{c}")
        positions[ident].add((r, c))
        tiles.append(dict(source="raise", cover_id=clean["cover_id"], cover_index=index,
                          photo_id=ident, tile_row=r, tile_col=c, pixel_y=y, pixel_x=x,
                          clean_path=clean["file"], stego_path=stego["file"],
                          split=by_photo[ident]["split"]))
    audit = Counter()
    for p in photos:
        expected = {(r, c) for r in range(p["legacy512_rows"]) for c in range(p["legacy512_cols"])}
        require(positions[p["photo_id"]] == expected, f"incomplete grid: {p['photo_id']}")
        orientation = "landscape" if p["source_width"] > p["source_height"] else "portrait"
        key = p["camera_model"], orientation
        require(key in EXPECTED_RAISE, f"unexpected camera/orientation: {key}")
        _, h, w, count = EXPECTED_RAISE[key]
        require((p["source_height"], p["source_width"], len(expected)) == (h, w, count),
                f"source dimensions/grid disagree with audit: {p['photo_id']}")
        audit[key] += 1
    require(dict(audit) == {k: v[0] for k, v in EXPECTED_RAISE.items()}, "camera audit count mismatch")
    return photos, tiles


def png_dimensions(path: Path) -> tuple[int, int]:
    """Read/validate RGB8 IHDR only; never decode or rewrite source pixels."""
    with path.open("rb") as stream:
        header = stream.read(33)
    require(len(header) == 33 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[8:16] == b"\x00\x00\x00\rIHDR", f"invalid PNG header: {path}")
    w, h, depth, colour, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
    require(zlib.crc32(header[12:29]) == int.from_bytes(header[29:33], "big"), f"bad IHDR CRC: {path}")
    require(w > 0 and h > 0 and (depth, colour, compression, filtering) == (8, 2, 0, 0)
            and interlace in {0, 1}, f"expected RGB8 PNG: {path}")
    return h, w


def div2k_train_ids() -> set[int]:
    """Seed 0, PCG64, permute sorted IDs 0001..0800; first 680 train.

    Explicit PCG64 matches default_rng(0) in NumPy 2.5.3. Freeze the CSV;
    Generator transformations need not stay identical across future versions.
    """
    return {int(i) + 1 for i in np.random.Generator(np.random.PCG64(0)).permutation(800)[:680]}


def build_div2k(root: Path) -> list[dict]:
    train_ids = div2k_train_ids()
    photos = []
    for folder, low, high in (("DIV2K_train_HR", 1, 800), ("DIV2K_valid_HR", 801, 900)):
        directory = root / "ImageSets" / folder
        files = sorted(directory.glob("*.png"))
        require([p.name for p in files] == [f"{i:04d}.png" for i in range(low, high + 1)],
                f"unexpected PNG inventory: {directory}")
        for path in files:
            ident = int(path.stem)
            h, w = png_dimensions(path)
            split = "test" if ident >= 801 else ("train" if ident in train_ids else "val")
            photos.append(photo("div2k", f"div2k_{ident:04d}", 999 + ident,
                                path.relative_to(root).as_posix(), h, w, split))
    return photos


def totals(photos: list[dict], tile: int) -> int:
    return sum(p[f"overlap{tile}_rows"] * p[f"overlap{tile}_cols"] for p in photos)


def build(root: Path, tiles_manifest: Path) -> tuple[dict[str, tuple[tuple, list[dict]]], dict]:
    raise_photos, legacy = build_raise(root, tiles_manifest)
    div_photos = build_div2k(root)
    all_photos = raise_photos + div_photos
    require(len(all_photos) == 1900 and len({p["photo_id"] for p in all_photos}) == 1900
            and len({p["photo_index"] for p in all_photos}) == 1900, "photo identity collision")
    require(all((root / p["master_path"]).is_file() for p in all_photos), "missing master path")
    require((totals(raise_photos, 512), totals(raise_photos, 256)) == (65184, 243120), "RAISE overlap totals")
    require((totals(div_photos, 512), totals(div_photos, 256)) == (10960, 42744), "DIV2K overlap totals")
    split_report = {}
    for split, expected in {"train": (680, 8276, 32216), "val": (120, 1456, 5720), "test": (100, 1228, 4808)}.items():
        rows = [p for p in div_photos if p["split"] == split]
        observed = len(rows), totals(rows, 512), totals(rows, 256)
        require(observed == expected, f"DIV2K split totals: {split}/{observed}")
        split_report[split] = dict(photos=observed[0], tiles512=observed[1], tiles256=observed[2])
    split_rows = [{k: p[k] for k in SPLIT_FIELDS} for p in div_photos]
    outputs = dict(zip(OUTPUTS, ((PHOTO_FIELDS, raise_photos), (TILE_FIELDS, legacy),
                                (PHOTO_FIELDS, div_photos), (SPLIT_FIELDS, split_rows))))
    report = dict(tiles_manifest=str(tiles_manifest),
                  tiles_manifest_sha256=hashlib.sha256(tiles_manifest.read_bytes()).hexdigest(),
                  checks="PASS", photos=1900, unique_photo_ids=1900, unique_photo_indices=1900,
                  master_paths_exist=1900, legacy_tiles=len(legacy), original_indices="preserved",
                  legacy_grids="complete; six camera/orientation audit groups matched",
                  raise_split="joined labels/metadata/preparation to committed split; one per photo",
                  raise_photo_splits=dict(Counter(p["split"] for p in raise_photos)),
                  raise_camera_splits={model: dict(Counter(p["split"] for p in raise_photos
                                                          if p["camera_model"] == model))
                                       for model in sorted({p["camera_model"] for p in raise_photos})},
                  raise_overlap={512: totals(raise_photos, 512), 256: totals(raise_photos, 256)},
                  div2k_overlap={512: totals(div_photos, 512), 256: totals(div_photos, 256)},
                  div2k_splits=split_report, numpy_version=np.__version__,
                  div2k_draw="Generator(PCG64(0)).permutation(800); first 680 train, rest val; 0801..0900 test")
    return outputs, report


def write_outputs(output_root: Path, outputs: dict) -> list[dict]:
    """Refuse the whole batch if any destination exists; use exclusive writes."""
    require(all(not (output_root / path).exists() for path in outputs), "output exists; use a new output root")
    created = []
    report = []
    try:
        for relative, (fields, rows) in outputs.items():
            path = output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as stream:
                created.append(path)
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            report.append(dict(file=relative, rows=len(rows), bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="read-only source repository/data root")
    parser.add_argument("--output-root", type=Path, required=True, help="scratch root for NEW CSV files only")
    parser.add_argument("--tiles-manifest", type=Path,
                        default=Path("ImageSets/RAISE/tiles512-records/manifest.csv"),
                        help="preparation manifest; relative to repo root unless absolute")
    args = parser.parse_args(argv)
    try:
        root, out = args.repo_root.resolve(), args.output_root.resolve()
        require(not out.is_relative_to(root), "output root must be outside the source checkout")
        require(all(not (out / path).resolve().is_relative_to(root) for path in OUTPUTS),
                "output symlink leads into source checkout")
        manifest = (root / args.tiles_manifest).resolve()
        outputs, report = build(root, manifest)
        report["files"] = write_outputs(out, outputs)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
