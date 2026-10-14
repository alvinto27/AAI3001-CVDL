#!/usr/bin/env bash
# First-boot setup and separate upload/cache/speed steps for rtx6000pro.
set -euo pipefail
STEP=${1:-}
[[ -n $STEP ]] || { echo 'Usage: setup.sh {base|upload|cache|speed} [--host IP] [--inbox-key FILE] [--workers N] [--streams N]' >&2; exit 2; }
shift
REPO_URL=https://github.com/alvinto27/AAI3001-CVDL.git
REPO_DIR="$HOME/AAI3001-CVDL"
SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CACHE_ROOT="$HOME/otf-cache/v1"
HOST=
INBOX_KEY="$HOME/.ssh/cvdl-g4-inbox"
SSH_KEY="$HOME/.ssh/cvdl-g4"
STREAMS=10
CACHE_WORKERS=$(( $(nproc) - 8 ))
(( CACHE_WORKERS < 1 )) && CACHE_WORKERS=1
while (($#)); do
    case "$1" in
        --host) HOST=${2:?missing host}; shift 2 ;;
        --inbox-key) INBOX_KEY=${2:?missing key}; shift 2 ;;
        --workers) CACHE_WORKERS=${2:?missing worker count}; shift 2 ;;
        --streams) STREAMS=${2:?missing stream count}; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$STEP" in
    base)
        [[ $(id -un) == yangxuan ]] || { echo 'Run base as yangxuan, not root' >&2; exit 2; }
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git rsync openssh-client
        if [[ ! -x "$HOME/.local/bin/uv" ]]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
        if [[ ! -d "$REPO_DIR/.git" ]]; then git clone --branch yx --single-branch "$REPO_URL" "$REPO_DIR"; fi
        [[ $(git -C "$REPO_DIR" remote get-url origin) == "$REPO_URL" ]] || { echo 'Wrong repository origin' >&2; exit 2; }
        [[ -z $(git -C "$REPO_DIR" status --porcelain) ]] || { echo 'Checkout is dirty; preserve it and clean it before setup' >&2; exit 2; }
        (cd "$REPO_DIR" && git switch yx && git pull --ff-only origin yx && "$HOME/.local/bin/uv" sync --locked)
        if ! command -v tailscale >/dev/null 2>&1; then curl -fsSL https://tailscale.com/install.sh | sh; fi
        sudo systemctl enable --now tailscaled
        if ! tailscale status >/dev/null 2>&1; then
            echo 'Approve the printed Tailscale login URL (no --ssh option).'
            sudo tailscale up --hostname=rtx6000pro
        fi
        [[ -f $INBOX_KEY ]] || { echo "Copy ~/.ssh/cvdl-g4-inbox from the PC to this VM first: $INBOX_KEY" >&2; exit 2; }
        mkdir -p "$HOME/.ssh" "$HOME/.local/lib/cvdl-g4"
        chmod 700 "$HOME/.ssh"
        if [[ $INBOX_KEY != "$HOME/.ssh/cvdl-g4-inbox" ]]; then
            install -m 600 "$INBOX_KEY" "$HOME/.ssh/cvdl-g4-inbox"
        else
            chmod 600 "$INBOX_KEY"
        fi
        install -m 755 "$REPO_DIR/scripts/g4/job.sh" "$HOME/.local/lib/cvdl-g4/job.sh"
        service=$(mktemp)
        sed -e "s|@USER@|$(id -un)|g" -e "s|@REPO@|$REPO_DIR|g" \
            "$REPO_DIR/scripts/g4/cvdl-g4-job.service" > "$service"
        sudo install -m 644 "$service" /etc/systemd/system/cvdl-g4-job.service
        rm -f "$service"
        sudo systemctl daemon-reload
        sudo systemctl enable cvdl-g4-job.service
        echo 'CHECK SUMMARY'
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        "$REPO_DIR/.venv/bin/python" -c 'import torch; print("CUDA available:",torch.cuda.is_available()); print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
        df -h "$HOME"
        echo 'Service enabled for the next boot. Maintenance mode stays up until the GCE maximum run duration.'
        ;;
    upload)
        [[ -n $HOST ]] || { echo 'upload needs --host EXTERNAL_IP' >&2; exit 2; }
        exec bash "$SOURCE_DIR/scripts/g4/upload_masters.sh" --host "$HOST" --streams "$STREAMS" --ssh-key "$SSH_KEY"
        ;;
    cache)
        [[ -x "$REPO_DIR/.venv/bin/python" ]] || { echo 'Run base first' >&2; exit 2; }
        [[ -d "$REPO_DIR/ImageSets/RAISE" ]] || { echo 'Run upload first' >&2; exit 2; }
        [[ $CACHE_WORKERS =~ ^[0-9]+$ ]] || { echo '--workers must be a positive integer' >&2; exit 2; }
        mkdir -p "$CACHE_ROOT"
        echo "Build cache with $CACHE_WORKERS workers (reserve about 1 GiB RAM per worker)."
        (cd "$REPO_DIR" && OMP_NUM_THREADS=1 .venv/bin/python scripts/build_cache.py \
            --repo-root "$REPO_DIR" --cache-root "$CACHE_ROOT" --workers "$CACHE_WORKERS" --allow-high-workers && \
            OMP_NUM_THREADS=1 .venv/bin/python scripts/check_cache.py \
            --repo-root "$REPO_DIR" --cache-root "$CACHE_ROOT" --gate verify && \
            .venv/bin/python scripts/g4/verify_cache.py "$CACHE_ROOT")
        ;;
    speed)
        [[ -x "$REPO_DIR/.venv/bin/python" ]] || { echo 'Run base first' >&2; exit 2; }
        (cd "$REPO_DIR" && sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && \
            OMP_NUM_THREADS=1 .venv/bin/python scripts/g4/speed_test.py \
                --cache-root "$CACHE_ROOT" --out "runs-out/g4-speed-$(date -u +%Y%m%dT%H%M%SZ).json")
        ;;
    *) echo 'Usage: setup.sh {base|upload|cache|speed}' >&2; exit 2 ;;
esac
