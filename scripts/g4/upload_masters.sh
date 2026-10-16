#!/usr/bin/env bash
# Upload the master photos with parallel, resumable rsync streams.
# Layout: Datasets/originals/{raise,div2k,alaska,lsdir} (same as the PC).
# RAISE and DIV2K: whole folders. ALASKA and LSDIR: only the files named by
# master_path in data/<source>/photo_manifest.csv (the unused ALASKA files stay on the PC).
# Remote: upload_masters.sh --host IP. Local-path mode (--dest-dir without
# --host) exists for PC tests only.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_DIR="$REPO_ROOT/Datasets/originals"
DEFAULT_DEST="/home/yangxuan/AAI3001-CVDL/Datasets/originals"
DEST_DIR=$DEFAULT_DEST
SSH_KEY="$HOME/.ssh/cvdl-g4"
REMOTE_USER=yangxuan
HOST=
STREAMS=10
FOLDERS=(raise div2k alaska lsdir)
WORK=
# Cheaper cipher than the default; the PC CPU was not the limit.
CIPHER="aes128-gcm""@openssh.com"
cleanup() { [[ -n $WORK ]] && rm -rf -- "$WORK"; }
trap cleanup EXIT
usage() { echo 'Usage: upload_masters.sh --host IP [--streams N] [--ssh-key FILE] | --dest-dir DIR (test) [--source-dir DIR] [--repo-root DIR] [--folders raise,div2k,alaska,lsdir]' >&2; exit 2; }
while (($#)); do
    case "$1" in
        --host) HOST=${2:?missing host}; shift 2 ;;
        --streams) STREAMS=${2:?missing count}; shift 2 ;;
        --ssh-key) SSH_KEY=${2:?missing key}; shift 2 ;;
        --dest-dir) DEST_DIR=${2:?missing dir}; shift 2 ;;
        --source-dir) SOURCE_DIR=${2:?missing dir}; shift 2 ;;
        --repo-root) REPO_ROOT=${2:?missing dir}; shift 2 ;;
        --folders) IFS=, read -r -a FOLDERS <<< "${2:?missing list}"; shift 2 ;;
        *) usage ;;
    esac
done
[[ $STREAMS =~ ^[0-9]+$ ]] && (( STREAMS >= 1 && STREAMS <= 32 )) || { echo '--streams must be 1..32' >&2; exit 2; }
if [[ -n $HOST ]]; then
    [[ -f $SSH_KEY ]] || { echo "Missing PC SSH key: $SSH_KEY" >&2; exit 2; }
    SSH_OPTS=(-i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new
              -o Compression=no -c "$CIPHER")
    TARGET="$REMOTE_USER@$HOST:$DEST_DIR/"
    run_dest() { ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$HOST" "$@"; }
    RSYNC_RSH="ssh ${SSH_OPTS[*]}"
else
    [[ -n ${DEST_DIR:-} && $DEST_DIR != $DEFAULT_DEST ]] || usage
    TARGET="$DEST_DIR/"
    run_dest() { bash -c "$*"; }
    RSYNC_RSH=
fi
for folder in "${FOLDERS[@]}"; do
    case $folder in raise|div2k|alaska|lsdir) ;; *) echo "Unknown folder $folder" >&2; exit 2 ;; esac
    [[ -d "$SOURCE_DIR/$folder" ]] || { echo "Missing master folder $folder" >&2; exit 2; }
done
WORK=$(mktemp -d)
# File list (paths relative to SOURCE_DIR, one per line) for each folder.
for folder in "${FOLDERS[@]}"; do
    case $folder in
        raise|div2k) (cd "$SOURCE_DIR" && find "$folder" -type f | LC_ALL=C sort) > "$WORK/folder.$folder" ;;
        alaska|lsdir) python3 - "$REPO_ROOT/data/$folder/photo_manifest.csv" "$folder" > "$WORK/folder.$folder" <<'PY'
import csv, sys
prefix = "Datasets/originals/" + sys.argv[2] + "/"
for row in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")):
    path = row["master_path"]
    if not path.startswith(prefix) or ".." in path.split("/"):
        sys.exit("bad master_path: " + path)
    print(path[len("Datasets/originals/"):])
PY
        ;;
    esac
    [[ -s "$WORK/folder.$folder" ]] || { echo "Empty file list for $folder" >&2; exit 2; }
    while IFS= read -r f; do [[ -f "$SOURCE_DIR/$f" ]] || { echo "Missing source file $f" >&2; exit 2; }; done < "$WORK/folder.$folder"
done
# Largest files first, dealt round-robin so every stream gets a similar load.
(cd "$SOURCE_DIR" && cat "$WORK"/folder.* | tr '\n' '\0' | xargs -0 stat -c '%s %n') | sort -rn \
    | awk -v n="$STREAMS" -v dir="$WORK" '{ sub(/^[0-9]+ /, ""); print > (dir "/list." (NR % n)) }'
if [[ -n $HOST ]]; then run_dest "mkdir -p '$DEST_DIR'"; else mkdir -p "$DEST_DIR"; fi
echo "Upload with $STREAMS streams (safe to resume; rerun after any failure)"
begin=$SECONDS
pids=(); names=()
for list in "$WORK"/list.*; do
    args=(-a --partial --append-verify --files-from="$list")
    [[ -n $RSYNC_RSH ]] && args+=(-e "$RSYNC_RSH")
    rsync "${args[@]}" -- "$SOURCE_DIR/" "$TARGET" 2> "$list.err" &
    pids+=($!); names+=("${list##*/}")
done
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "Stream ${names[$i]} failed:" >&2; cat "$WORK/${names[$i]}.err" >&2; failed=1
    fi
done
(( failed == 0 )) || { echo 'Upload FAILED; rerun the same command to resume.' >&2; exit 1; }
# Compare file count and total bytes per folder with the PC (same file list on both sides).
status=0
for folder in "${FOLDERS[@]}"; do
    local_stats=$(cd "$SOURCE_DIR" && tr '\n' '\0' < "$WORK/folder.$folder" | xargs -0 stat -c '%s' | awk '{ n++; s += $1 } END { print n + 0, s + 0 }')
    remote_stats=$(run_dest "cd '$DEST_DIR' && tr '\n' '\0' | xargs -0 stat -c '%s' | awk '{ n++; s += \$1 } END { print n + 0, s + 0 }'" < "$WORK/folder.$folder")
    if [[ $local_stats == "$remote_stats" ]]; then
        echo "OK $folder: files, bytes = $local_stats"
    else
        echo "MISMATCH $folder: PC=$local_stats target=$remote_stats" >&2; status=1
    fi
done
(( status == 0 )) || exit 1
echo "Upload complete and checked in $((SECONDS - begin)) s. The otf-cache is not copied; it is optional and absent by default."
