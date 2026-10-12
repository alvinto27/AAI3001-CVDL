#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Rebuild the RAISE 512 dataset from the repository root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DELETE_INTERMEDIATE=0
SHUTDOWN_REQUESTED=0

usage() {
    cat <<'EOF'
Usage: scripts/rebuild_raise512_run1.sh [--delete-intermediate] [--shutdown]

Options:
  --delete-intermediate  Remove NEF files and tiles after all checks pass.
  --shutdown             Schedule shutdown in two minutes after the run.
EOF
}

while (($# > 0)); do
    case "$1" in
        --delete-intermediate)
            DELETE_INTERMEDIATE=1
            ;;
        --shutdown)
            SHUTDOWN_REQUESTED=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf 'error: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

STEP_NUMBER=0
STEP_NAME="startup"

stamp() {
    date --iso-8601=seconds
}

step_start() {
    STEP_NUMBER="$1"
    STEP_NAME="$2"
    printf '[%s] STEP %s/7 %s start\n' "$(stamp)" "$STEP_NUMBER" "$STEP_NAME"
}

step_done() {
    printf '[%s] STEP %s/7 %s done\n' "$(stamp)" "$STEP_NUMBER" "$STEP_NAME"
}

shutdown_if_requested() {
    if ((SHUTDOWN_REQUESTED == 0)); then
        return 0
    fi

    local current_user sessions gpu_processes
    current_user="${USER:-$(id -un)}"
    if ! sessions="$(who)"; then
        printf '[%s] WARNING cannot shut down: cannot check login sessions\n' "$(stamp)" >&2
        return 0
    fi
    if printf '%s\n' "$sessions" | awk -v me="$current_user" \
        'NF && $1 != me { found=1 } END { exit !found }' >/dev/null; then
        printf '[%s] WARNING cannot shut down: another user has an active login session\n' "$(stamp)" >&2
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '[%s] WARNING cannot shut down: nvidia-smi is unavailable\n' "$(stamp)" >&2
        return 0
    fi
    if ! gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits 2>/dev/null)"; then
        printf '[%s] WARNING cannot shut down: cannot check GPU processes\n' "$(stamp)" >&2
        return 0
    fi
    if [[ -n "$gpu_processes" ]]; then
        printf '[%s] WARNING cannot shut down: another process uses the GPU\n' "$(stamp)" >&2
        return 0
    fi

    if sudo -n /usr/sbin/shutdown -h +2 >/dev/null 2>&1; then
        printf '[%s] Shutdown scheduled in two minutes\n' "$(stamp)"
    else
        printf '[%s] WARNING cannot shut down: shutdown command failed\n' "$(stamp)" >&2
    fi
}

on_error() {
    local status=$?
    trap - ERR
    shutdown_if_requested
    printf 'REBUILD FAILED at step %s %s\n' "$STEP_NUMBER" "$STEP_NAME"
    exit "$status"
}

trap on_error ERR

step_start 1 download
uv run python scripts/download_raise.py \
    --selection data/raise/selection.csv \
    --out ImageSets/RAISE/NEF
step_done

step_start 2 prepare
uv run python LSB-Dataset-Generator/prepare.py \
    --input ImageSets/RAISE/NEF \
    --output ImageSets/RAISE/tiles512 \
    --tile 512 \
    --seed 0 \
    --png-compress-level 1
step_done

step_start 3 generate
uv run python LSB-Dataset-Generator/generate.py \
    --config configs/raise512-run1.yaml
step_done

step_start 4 verify
uv run python LSB-Dataset-Generator/generate.py \
    --verify Datasets/raise512-run1
step_done

step_start 5 build-labels
uv run python scripts/build_labels.py \
    --dataset Datasets/raise512-run1 \
    --tiles-manifest ImageSets/RAISE/tiles512/manifest.csv
step_done

step_start 6 copy-split
mkdir -p Datasets/raise512-run1/splits
cp -f data/raise/split_seed0.csv Datasets/raise512-run1/splits/split_seed0.csv
step_done

check_count() {
    local name="$1"
    local actual="$2"
    local expected="$3"
    if [[ "$actual" != "$expected" ]]; then
        printf 'count check failed: %s=%s, expected %s\n' "$name" "$actual" "$expected" >&2
        return 1
    fi
}

step_start 7 count-checks
clean_count="$(find Datasets/raise512-run1/clean -type f -name '*.png' -print | wc -l)"
stego_count="$(find Datasets/raise512-run1/stego -type f -name '*.png' -print | wc -l)"
mask_count="$(find Datasets/raise512-run1/masks -type f -name '*.png' -print | wc -l)"
label_count="$(awk 'NR > 1 { count++ } END { print count + 0 }' Datasets/raise512-run1/labels.csv)"
check_count clean "$clean_count" 49796
check_count stego "$stego_count" 49796
check_count masks "$mask_count" 49796
check_count labels "$label_count" 99592
step_done

if ((DELETE_INTERMEDIATE == 1)); then
    rm -rf ImageSets/RAISE/NEF ImageSets/RAISE/tiles512
    printf '[%s] Deleted intermediate NEF and tile folders\n' "$(stamp)"
fi

shutdown_if_requested
printf 'REBUILD DONE: clean=%s stego=%s masks=%s labels=%s\n' \
    "$clean_count" "$stego_count" "$mask_count" "$label_count"
