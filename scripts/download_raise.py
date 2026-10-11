#!/usr/bin/env python3
"""Select and download a stratified subset of the RAISE NEF images."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


REQUIRED_COLUMNS: tuple[str, ...] = (
    "File",
    "NEF",
    "File Size",
    "Device",
    "Keywords",
)
SELECTION_COLUMNS: tuple[str, ...] = (
    "File",
    "Device",
    "Keywords",
    "NEF URL",
    "expected size",
)
SIZE_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B)\s*$", re.I)
SELECTION_META = re.compile(r"^# download_raise seed=(-?\d+) count=(\d+)\s*$")
CHUNK_SIZE = 1024 * 1024
SIZE_TOLERANCE = int(0.05 * 1024**2) + 64 * 1024


@dataclass(frozen=True)
class Photo:
    """One RAISE catalog entry needed by the downloader."""

    file: str
    device: str
    keywords: str
    nef_url: str
    expected_size: str
    expected_bytes: int


@dataclass(frozen=True)
class DownloadResult:
    """The outcome of one file operation."""

    name: str
    status: str
    bytes_written: int = 0
    error: str = ""


class SelectionError(ValueError):
    """Raised when an existing selection cannot be safely reused."""


def parse_size(text: str) -> int:
    """Convert a human-readable catalog size to bytes."""
    match = SIZE_PATTERN.match(text)
    if match is None:
        raise ValueError(f"invalid file size {text!r}")
    value = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported file-size unit in {text!r}")
    return int(round(value * multipliers[unit]))


def column_indices(header: Sequence[str], required: Iterable[str]) -> Mapping[str, int]:
    """Find the first occurrence of each required column safely."""
    indices: dict[str, int] = {}
    for name in required:
        try:
            indices[name] = header.index(name)
        except ValueError as exc:
            raise ValueError(f"CSV is missing required column {name!r}") from exc
    return indices


def photo_from_values(values: Mapping[str, str]) -> Photo:
    """Build a photo record from catalog or selection values."""
    expected_size = values["expected size"]
    return Photo(
        file=values["File"].strip(),
        device=values["Device"].strip(),
        keywords=values["Keywords"].strip(),
        nef_url=values["NEF URL"].strip(),
        expected_size=expected_size.strip(),
        expected_bytes=parse_size(expected_size),
    )


def load_catalog(path: Path) -> list[Photo]:
    """Read the RAISE catalog without collapsing duplicate headers."""
    photos: list[Photo] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"catalog is empty: {path}") from exc
        indices = column_indices(header, REQUIRED_COLUMNS)
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            if max(indices.values()) >= len(row):
                raise ValueError(f"catalog row {line_number} has too few columns")
            values = {
                "File": row[indices["File"]],
                "Device": row[indices["Device"]],
                "Keywords": row[indices["Keywords"]],
                "NEF URL": row[indices["NEF"]],
                "expected size": row[indices["File Size"]],
            }
            try:
                photos.append(photo_from_values(values))
            except ValueError as exc:
                raise ValueError(f"catalog row {line_number}: {exc}") from exc
    if not photos:
        raise ValueError(f"catalog has no image rows: {path}")
    return photos


def stratified_selection(catalog: Sequence[Photo], count: int, seed: int) -> list[Photo]:
    """Select count photos with proportional Device/Keywords strata."""
    if count < 0:
        raise ValueError("count must not be negative")
    if count > len(catalog):
        raise ValueError(f"count {count} exceeds catalog size {len(catalog)}")
    groups: dict[tuple[str, str], list[Photo]] = defaultdict(list)
    for photo in catalog:
        groups[(photo.device, photo.keywords)].append(photo)

    total = len(catalog)
    targets: dict[tuple[str, str], int] = {}
    remainders: list[tuple[int, tuple[str, str]]] = []
    for key in sorted(groups):
        numerator = count * len(groups[key])
        targets[key] = numerator // total
        remainders.append((numerator % total, key))
    remaining = count - sum(targets.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        targets[key] += 1

    generator = random.Random(seed)
    selected: list[Photo] = []
    for key in sorted(groups):
        selected.extend(generator.sample(groups[key], targets[key]))
    generator.shuffle(selected)
    return selected


def selection_path(csv_path: Path) -> Path:
    """Return the selection manifest path beside the catalog."""
    return csv_path.parent / "selection.csv"


def read_selection(path: Path) -> tuple[int | None, int | None, list[Photo]]:
    """Read selection metadata and rows from a manifest."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        if not first_line:
            raise SelectionError(f"selection manifest is empty: {path}")
        metadata_match = SELECTION_META.match(first_line.rstrip("\n\r"))
        metadata: tuple[int | None, int | None]
        if metadata_match is None:
            metadata = (None, None)
            remaining = [first_line] + list(handle)
        else:
            metadata = (int(metadata_match.group(1)), int(metadata_match.group(2)))
            remaining = list(handle)
    reader = csv.reader(remaining)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise SelectionError(f"selection manifest has no header: {path}") from exc
    indices = column_indices(header, SELECTION_COLUMNS)
    rows: list[Photo] = []
    for line_number, row in enumerate(reader, start=2 if metadata[0] is None else 3):
        if not row or not any(cell.strip() for cell in row):
            continue
        if max(indices.values()) >= len(row):
            raise SelectionError(f"selection row {line_number} has too few columns")
        values = {name: row[indices[name]] for name in SELECTION_COLUMNS}
        try:
            rows.append(photo_from_values(values))
        except ValueError as exc:
            raise SelectionError(f"selection row {line_number}: {exc}") from exc
    return metadata[0], metadata[1], rows


def write_selection(path: Path, selected: Sequence[Photo], seed: int, count: int) -> None:
    """Write a reproducible selection manifest before downloads begin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# download_raise seed={seed} count={count}\n")
        writer = csv.writer(handle)
        writer.writerow(SELECTION_COLUMNS)
        for photo in selected:
            writer.writerow(
                (
                    photo.file,
                    photo.device,
                    photo.keywords,
                    photo.nef_url,
                    photo.expected_size,
                )
            )


def prepare_selection(
    catalog: Sequence[Photo],
    path: Path,
    count: int,
    seed: int,
    force: bool,
) -> list[Photo]:
    """Reuse or create a manifest, refusing accidental reselection."""
    if path.exists() and not force:
        old_seed, old_count, selected = read_selection(path)
        if old_seed != seed or old_count != count:
            raise SelectionError(
                f"{path} was made with seed={old_seed!r}, count={old_count!r}; "
                "use --force to replace it"
            )
        if len(selected) != count:
            raise SelectionError(f"{path} contains {len(selected)} rows, expected {count}")
        return selected
    selected = stratified_selection(catalog, count, seed)
    write_selection(path, selected, seed, count)
    return selected


def display_name(value: str) -> str:
    """Use a compact label for blank metadata values."""
    return value if value else "(blank)"


def print_summary(selected: Sequence[Photo], catalog: Sequence[Photo]) -> None:
    """Print selected and full-catalog Device and Keywords shares."""
    selected_devices = Counter(photo.device for photo in selected)
    full_devices = Counter(photo.device for photo in catalog)
    print(f"Selected {len(selected):,} of {len(catalog):,} images")
    print("Device counts (selected share / full-catalog share):")
    for device in sorted(full_devices, key=lambda value: (value == "", value.lower())):
        selected_count = selected_devices[device]
        selected_share = selected_count / len(selected) if selected else 0.0
        full_share = full_devices[device] / len(catalog)
        print(
            f"  {display_name(device)}: {selected_count:,} / {full_devices[device]:,} "
            f"({selected_share:.1%} / {full_share:.1%})"
        )

    keywords = Counter(photo.keywords for photo in selected)
    print("Top Keywords in selection:")
    for subject, amount in sorted(keywords.items(), key=lambda item: (-item[1], item[0]))[:20]:
        full_amount = sum(photo.keywords == subject for photo in catalog)
        print(f"  {display_name(subject)}: {amount:,} (full {full_amount:,})")
    expected_gb = sum(photo.expected_bytes for photo in selected) / 1_000_000_000
    print(f"Total expected size: {expected_gb:.2f} GB")


def destination_for(photo: Photo, output: Path) -> Path:
    """Derive a safe output filename from the NEF URL."""
    name = Path(unquote(urlsplit(photo.nef_url).path)).name
    if not name or name in {".", ".."}:
        name = f"{photo.file}.NEF"
    return output / name


def valid_image(path: Path, expected_bytes: int) -> bool:
    """Check the rounded catalog size and TIFF byte-order marker."""
    try:
        size = path.stat().st_size
        if abs(size - expected_bytes) > SIZE_TOLERANCE:
            return False
        with path.open("rb") as handle:
            return handle.read(4) in (b"II*\x00", b"MM\x00*")
    except (OSError, ValueError):
        return False


def download_one(photo: Photo, output: Path, timeout: float, retries: int) -> DownloadResult:
    """Download one NEF with validation and bounded retries."""
    destination = destination_for(photo, output)
    part = Path(f"{destination}.part")
    if valid_image(destination, photo.expected_bytes):
        return DownloadResult(destination.name, "skipped")
    try:
        part.unlink(missing_ok=True)
    except OSError:
        pass

    last_error = "download failed"
    for attempt in range(retries + 1):
        try:
            request = Request(photo.nef_url, headers={"User-Agent": "RAISE-downloader/1.0"})
            written = 0
            with urlopen(request, timeout=timeout) as response, part.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
            if not valid_image(part, photo.expected_bytes):
                raise ValueError("size or TIFF header check failed")
            part.replace(destination)
            return DownloadResult(destination.name, "downloaded", written)
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < retries:
                time.sleep(min(2.0, 0.25 * (2**attempt)))
    return DownloadResult(destination.name, "failed", error=last_error)


def remove_part_files(output: Path) -> None:
    """Remove temporary download files left in the output directory."""
    if not output.exists():
        return
    for part in output.glob("*.part"):
        try:
            part.unlink()
        except OSError:
            pass


def download_all(
    selected: Sequence[Photo],
    output: Path,
    workers: int,
    timeout: float,
    retries: int,
) -> tuple[int, int, list[str]]:
    """Download all selected files and report progress."""
    output.mkdir(parents=True, exist_ok=True)
    remove_part_files(output)
    started = time.monotonic()
    downloaded = 0
    skipped = 0
    failed: list[str] = []
    completed = 0
    total_bytes = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    futures: dict[Future[DownloadResult], Photo] = {
        executor.submit(download_one, photo, output, timeout, retries): photo for photo in selected
    }
    try:
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result.status == "downloaded":
                downloaded += 1
                total_bytes += result.bytes_written
            elif result.status == "skipped":
                skipped += 1
            else:
                failed.append(result.name)
            elapsed = max(time.monotonic() - started, 0.001)
            rate = total_bytes / elapsed / 1_000_000
            suffix = f" ({result.error})" if result.error else ""
            print(f"[{completed}/{len(selected)}] {result.status}: {result.name} — {rate:.1f} MB/s{suffix}")
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        remove_part_files(output)
        print("\nInterrupted; temporary files removed.", file=sys.stderr)
        raise
    else:
        executor.shutdown(wait=True)
        remove_part_files(output)
    return downloaded, skipped, failed


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI option."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer CLI option."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=nonnegative_int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=Path("ImageSets/RAISE/RAISE_all.csv"))
    parser.add_argument("--out", type=Path, default=Path("ImageSets/RAISE/NEF"))
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--retries", type=nonnegative_int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--force", action="store_true", help="replace an existing selection manifest")
    parser.add_argument("--dry-run", action="store_true", help="show selection statistics only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run selection, optional summary, and downloads."""
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2
    try:
        catalog = load_catalog(args.csv)
        if args.dry_run:
            selected = stratified_selection(catalog, args.count, args.seed)
            print_summary(selected, catalog)
            return 0
        manifest = selection_path(args.csv)
        selected = prepare_selection(catalog, manifest, args.count, args.seed, args.force)
        print(f"Selection manifest: {manifest} ({len(selected):,} files)")
        downloaded, skipped, failed = download_all(
            selected, args.out, args.workers, args.timeout, args.retries
        )
    except KeyboardInterrupt:
        return 130
    except (OSError, SelectionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Summary: downloaded={downloaded}, skipped={skipped}, failed={len(failed)}")
    if failed:
        print("Failed files: " + ", ".join(sorted(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
