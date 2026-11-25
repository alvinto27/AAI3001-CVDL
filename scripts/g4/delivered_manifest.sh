#!/usr/bin/env bash
# Run on the PC. READ-ONLY on the Pi: lists every delivered file under
# /srv/cvdl-results (size, and SHA-256 for *.ckpt) into a local file.
# The VM script reorg_data_layout.sh --clean-runs uses it to confirm that a
# per-epoch checkpoint on the VM is already on the Pi, byte for byte.
# Usage: delivered_manifest.sh OUTFILE [--host rpi400@rpi400-server]
set -euo pipefail
OUT=${1:?Usage: delivered_manifest.sh OUTFILE [--host USER@HOST]}
HOST=rpi400@rpi400-server
shift
while (($#)); do
    case "$1" in
        --host) HOST=${2:?missing host}; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
tmp=$(mktemp)
trap 'rm -f "$tmp" "$tmp.sha"' EXIT
# Only find and sha256sum run on the Pi. Nothing is written there.
ssh -o BatchMode=yes "$HOST" 'cd /srv/cvdl-results && find . -mindepth 2 -type f -printf "%s\t%P\n"' > "$tmp"
ssh -o BatchMode=yes "$HOST" 'cd /srv/cvdl-results && find . -mindepth 2 -type f -name "*.ckpt" -exec sha256sum {} +' > "$tmp.sha"
python3 - "$tmp" "$tmp.sha" > "$OUT" <<'PY'
import sys
sha = {}
for line in open(sys.argv[2], encoding="utf-8"):
    digest, path = line.rstrip("\n").split("  ", 1)
    sha[path[2:] if path.startswith("./") else path] = digest
for line in open(sys.argv[1], encoding="utf-8"):
    size, path = line.rstrip("\n").split("\t", 1)
    print(f"{size}\t{sha.get(path, '-')}\t{path}")
PY
echo "Wrote $(wc -l < "$OUT") entries to $OUT"
