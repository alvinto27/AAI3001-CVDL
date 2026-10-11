"""Build image labels by joining generator metadata to tile metadata."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METADATA_FIELDS = (
    "height",
    "width",
    "channels",
    "generator_version",
    "seed",
    "embed_seed_seed",
    "embed_seed_cover_index",
    "embed_seed_variant",
    "embed_seed_stream",
    "bit_mode",
    "k",
    "method",
    "placement",
    "start",
    "start_random",
    "stride",
    "payload_kind",
    "payload_bytes",
    "embedded_bits",
    "units_used",
    "units_changed",
    "bpp_actual",
    "size_kind",
    "size_value",
    "mask_file",
)
OUTPUT_FIELDS = (
    "file",
    "label",
    "cover_id",
    "cover_index",
    "variant",
    "source_photo",
    "source_file",
    "camera_make",
    "camera_model",
    "tile_row",
    "tile_col",
    "x",
    "y",
    *METADATA_FIELDS,
)


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Read the tile manifest, indexed by tile file name."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"tile_file", "source_file", "camera_make", "camera_model", "tile_row", "tile_col", "pixel_x", "pixel_y"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
    manifest: dict[str, dict[str, str]] = {}
    for row in rows:
        tile_file = row["tile_file"]
        if tile_file in manifest:
            raise ValueError(f"manifest has duplicate tile_file: {tile_file}")
        manifest[tile_file] = row
    return manifest


def _label_row(
    metadata: dict[str, str],
    manifest: dict[str, str],
    dataset: Path,
) -> dict[str, str]:
    """Create one clean or stego label row."""
    file_path = metadata.get("file", "")
    label_name = metadata.get("label", "")
    if label_name not in {"clean", "stego"}:
        raise ValueError(f"unsupported metadata label: {label_name}")
    source_file = manifest["source_file"]
    row = {
        "file": file_path,
        "label": "0" if label_name == "clean" else "1",
        "cover_id": metadata.get("cover_id", ""),
        "cover_index": metadata.get("cover_index", ""),
        "variant": metadata.get("variant", ""),
        "source_photo": Path(source_file).stem,
        "source_file": Path(source_file).name,
        "camera_make": manifest.get("camera_make", ""),
        "camera_model": manifest.get("camera_model", ""),
        "tile_row": manifest["tile_row"],
        "tile_col": manifest["tile_col"],
        "x": manifest["pixel_x"],
        "y": manifest["pixel_y"],
    }
    row.update({field: metadata.get(field, "") for field in METADATA_FIELDS})
    if not (dataset / file_path).is_file():
        raise ValueError(f"metadata file does not exist: {file_path}")
    return row


def build_labels(dataset: Path, tiles_manifest: Path) -> int:
    """Write labels.csv and return its data-row count."""
    dataset = Path(dataset)
    metadata_path = dataset / "metadata.csv"
    if not metadata_path.is_file():
        raise ValueError(f"metadata file does not exist: {metadata_path}")
    manifest = _read_manifest(Path(tiles_manifest))
    with metadata_path.open(newline="", encoding="utf-8") as stream:
        metadata_rows = list(csv.DictReader(stream))
    output_rows: list[dict[str, str]] = []
    for number, metadata in enumerate(metadata_rows, 2):
        tile_file = metadata.get("source_file", "")
        if tile_file not in manifest:
            raise ValueError(
                f"metadata row {number} has no manifest match for tile file: {tile_file}"
            )
        output_rows.append(_label_row(metadata, manifest[tile_file], dataset))
    output_path = dataset / "labels.csv"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(output_rows)


def main(argv: list[str] | None = None) -> int:
    """Run the label builder CLI."""
    parser = argparse.ArgumentParser(description="Build image labels from v6 metadata")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tiles-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        count = build_labels(args.dataset, args.tiles_manifest)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Wrote {count} rows to {args.dataset / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
