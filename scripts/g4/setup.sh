#!/usr/bin/env bash
# First-boot setup and separate upload/cache/speed steps for rtx6000pro.
set -euo pipefail
STEP=${1:-}
[[ -n $STEP ]] || { echo 'Usage: setup.sh {base|freeze-updates|upload|cache|speed} [--host IP] [--inbox-key FILE] [--workers N] [--streams N]' >&2; exit 2; }
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
# Only run this policy on the dedicated training VM, during maintenance.
# Package installation can restart services even when PID1 re-exec is harmless.
freeze_updates() {
    [[ $(id -un) == yangxuan && $(hostname -s) == rtx6000pro ]] || {
        echo 'freeze-updates requires yangxuan on rtx6000pro (maintenance only)' >&2; return 2;
    }
    local unit state load config
    state=$(sudo systemctl show cvdl-g4-job.service --property=ActiveState --value)
    [[ $state == inactive || $state == failed ]] || {
        echo 'Job unit is not quiescent; do not change package policy during a job' >&2; return 2;
    }
    # Stop new triggers, but NEVER kill a running apt/dpkg transaction.
    sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer
    for unit in apt-daily.service apt-daily-upgrade.service; do
        state=$(sudo systemctl show "$unit" --property=ActiveState --value)
        [[ $state == inactive || $state == failed ]] || {
            echo "$unit is busy: wait for it to finish, then rerun freeze-updates" >&2; return 2;
        }
    done
    sudo mkdir -p /etc/apt/apt.conf.d /etc/needrestart/conf.d
    printf '%s\n' 'APT::Periodic::Update-Package-Lists "0";' 'APT::Periodic::Unattended-Upgrade "0";' |
        sudo tee /etc/apt/apt.conf.d/zz-cvdl-no-auto-updates >/dev/null
    # Ubuntu24.04 needrestart uses -m u; a restart-mode variable alone is not
    # sufficient. Exclude only our one-shot job, including future installations.
    printf '%s\n' '$nrconf{override_rc}->{qr(^cvdl-g4-job\.service$)} = 0;' |
        sudo tee /etc/needrestart/conf.d/99-cvdl-job.conf >/dev/null
    local units=(apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service unattended-upgrades.service)
    sudo systemctl mask --now "${units[@]}"
    # Persistent /etc masks and policy survive reboot and pinned git checkouts.
    for unit in "${units[@]}"; do
        load=$(sudo systemctl show "$unit" --property=LoadState --value)
        state=$(sudo systemctl show "$unit" --property=ActiveState --value)
        [[ $load == masked && $state == inactive ]] || {
            echo "$unit is not masked/inactive: $load/$state" >&2; return 2;
        }
        printf '%s %s %s\n' "$unit" "$load" "$state"
    done
    config=$(apt-config dump)
    grep -Fxq 'APT::Periodic::Update-Package-Lists "0";' <<< "$config"
    grep -Fxq 'APT::Periodic::Unattended-Upgrade "0";' <<< "$config"
    sudo systemctl show cvdl-g4-job.service --property=Type,Restart,KillMode,TimeoutStopUSec,Requires,Wants,After,PartOf,BindsTo
    echo 'Automatic package updates are disabled. Apply security updates in planned maintenance, not during jobs.'
}
case "$STEP" in
    freeze-updates) freeze_updates ;;
    base)
        [[ $(id -un) == yangxuan ]] || { echo 'Run base as yangxuan, not root' >&2; exit 2; }
        freeze_updates
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git rsync openssh-client build-essential python3-dev
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
        # job.sh is a small bootstrap. It runs scripts/g4/job_run.sh from the checkout,
        # so a later repo change needs no new install. Keep a known-good copy of job_run.sh.
        install -m 755 "$REPO_DIR/scripts/g4/job.sh" "$HOME/.local/lib/cvdl-g4/job.sh"
        install -m 755 "$REPO_DIR/scripts/g4/job_run.sh" "$HOME/.local/lib/cvdl-g4/job_run.last-good"
        service=$(mktemp)
        sed -e "s|@USER@|$(id -un)|g" -e "s|@REPO@|$REPO_DIR|g" \
            "$REPO_DIR/scripts/g4/cvdl-g4-job.service" > "$service"
        sudo install -m 644 "$service" /etc/systemd/system/cvdl-g4-job.service
        rm -f "$service"
        # logind deletes a user's POSIX semaphores in /dev/shm when the last SSH
        # session ends (RemoveIPC=yes). That killed spawn workers of a running job.
        sudo mkdir -p /etc/systemd/logind.conf.d
        printf '[Login]\nRemoveIPC=no\n' | sudo tee /etc/systemd/logind.conf.d/cvdl-no-remove-ipc.conf >/dev/null
        sudo systemctl restart systemd-logind
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
        [[ -d "$REPO_DIR/Datasets/originals/raise" && -d "$REPO_DIR/Datasets/originals/div2k" ]] || { echo 'Run upload first' >&2; exit 2; }
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
    *) echo 'Usage: setup.sh {base|freeze-updates|upload|cache|speed}' >&2; exit 2 ;;
esac
