#!/usr/bin/env bash
# One-time data-layout change for the G4 VM. Run ON THE VM, in a maintenance boot.
#
#   bash reorg_data_layout.sh [--commit REV] [--apply] [--clean-runs --delivered-manifest FILE]
#
# Dry run is the default: it prints the inventory, the plan, and the checks it
# can already make, and changes nothing. --apply acts.
#
# Old layout                                   New layout (same as the PC)
#   ImageSets/RAISE                            Datasets/originals/raise
#   ImageSets/DIV2K_{train,valid}_HR           Datasets/originals/div2k/DIV2K_{train,valid}_HR
#   Datasets/alaska/tiff_various_color         Datasets/originals/alaska/tiff_various_color
#   Datasets/lsdir/hr                          Datasets/originals/lsdir/hr
#
# Only same-filesystem renames are used for the photos. The script refuses if a
# target already exists. It deletes only: an otf-cache (unless --keep-cache) and
# the clean/stego/masks images of Datasets/raise512-run1, if present. Everything
# else it does not know is listed and left alone. It never touches
# runs-out/.g4-state, ~/.local/lib/cvdl-g4, or anything outside the paths above.
# --clean-runs (separate, needs --apply to act) removes only per-epoch
# checkpoints and TensorBoard logs of job folders that the VM record marks
# delivered AND that match the read-only Pi manifest (scripts/g4/delivered_manifest.sh).
# After the moves it checks out the target commit (detached) and checks every
# master_path of that commit's data/*/photo_manifest.csv.
set -uo pipefail

REPO="$HOME/AAI3001-CVDL"
HOME_DIR="$HOME"
COMMIT=origin/yx
APPLY=0
CLEAN_RUNS=0
MANIFEST=
KEEP_CACHE=0
FETCH=1
SHA=1

usage() {
    echo 'Usage: reorg_data_layout.sh [--commit REV] [--apply] [--keep-cache] [--no-sha]' >&2
    echo '       [--clean-runs --delivered-manifest FILE] [--repo DIR] [--home DIR] [--no-fetch]' >&2
    exit 2
}
while (($#)); do
    case "$1" in
        --commit) COMMIT=${2:?missing revision}; shift 2 ;;
        --apply) APPLY=1; shift ;;
        --clean-runs) CLEAN_RUNS=1; shift ;;
        --delivered-manifest) MANIFEST=${2:?missing file}; shift 2 ;;
        --keep-cache) KEEP_CACHE=1; shift ;;
        --no-sha) SHA=0; shift ;;
        --repo) REPO=${2:?missing dir}; shift 2 ;;
        --home) HOME_DIR=${2:?missing dir}; shift 2 ;;
        --no-fetch) FETCH=0; shift ;;
        *) usage ;;
    esac
done
[[ $COMMIT =~ ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$ && $COMMIT != *..* ]] || { echo "Bad --commit: $COMMIT" >&2; exit 2; }
if ((CLEAN_RUNS)) && [[ -z $MANIFEST || ! -f $MANIFEST ]]; then
    echo '--clean-runs needs --delivered-manifest FILE (make it on the PC with scripts/g4/delivered_manifest.sh)' >&2
    exit 2
fi
[[ -n $MANIFEST ]] && MANIFEST=$(realpath -- "$MANIFEST")
[[ -d $REPO/.git ]] || { echo "Not a git checkout: $REPO" >&2; exit 2; }
REPO=$(cd "$REPO" && pwd -P)
HOME_DIR=$(cd "$HOME_DIR" && pwd -P)
cd "$REPO" || exit 2

MODE=DRY-RUN; ((APPLY)) && MODE=APPLY
say() { printf '%s\n' "$*"; }
head1() { printf '\n== %s ==\n' "$*"; }
REFUSALS=(); UNEXPECTED=(); PLAN=()
refuse() { REFUSALS+=("$*"); say "REFUSE: $*"; }
plan() { PLAN+=("$*"); say "  plan: $*"; }
size_of() { du -sh -- "$1" 2>/dev/null | cut -f1; }

# Everything that rm may touch must pass this check.
protected() { case "$1" in *.g4-state*|*cvdl-g4*) return 0 ;; esac; return 1; }
safe_rm() {
    local path real
    path=$1
    real=$(realpath -m -- "$path")
    if protected "$real"; then say "REFUSE to delete protected path $real"; return 1; fi
    case "$real" in
        "$REPO"/Datasets/*|"$HOME_DIR"/otf-cache|"$HOME_DIR"/otf-cache/*) ;;
        *) say "REFUSE to delete outside the allowed paths: $real"; return 1 ;;
    esac
    [[ $real != "$REPO/Datasets" && $real != "$REPO/Datasets/originals" && $real != "$REPO/Datasets/generated" ]] || { say "REFUSE: $real"; return 1; }
    rm -rf -- "$real"
}

say "G4 data-layout change: mode=$MODE repo=$REPO target=$COMMIT"
say "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------- 1. Inventory (read only) ----------
head1 'Inventory'
df -h "$REPO" | sed 's/^/  /'
for d in ImageSets/* Datasets/* "$HOME_DIR/otf-cache" Datasets/raise512-run1/* runs-out "$HOME_DIR/.local/lib/cvdl-g4"; do
    [[ -e $d ]] && printf '  %8s  %s\n' "$(size_of "$d")" "$d"
done
[[ -d runs-out ]] && for d in runs-out/.g4-state runs-out/*; do
    [[ -e $d ]] && printf '  %8s    %s\n' "$(size_of "$d")" "$d"
done
say '  Largest other items in the home folder (nothing is deleted here):'
du -sh -- "$HOME_DIR"/* "$HOME_DIR"/.cache 2>/dev/null | sort -h | tail -8 | sed 's/^/  /'

# ---------- 2. Target commit ----------
head1 'Target commit'
if ((FETCH)); then
    git fetch -q origin yx || say '  git fetch failed; using the local copy of the target'
fi
TARGET=$(git rev-parse --verify --quiet "$COMMIT^{commit}" || true)
if [[ -z $TARGET ]]; then
    refuse "target commit $COMMIT not found"
else
    say "  target $TARGET ($(git log -1 --format=%s "$TARGET" | cut -c1-70))"
    if ! git show "$TARGET:data/raise/photo_manifest.csv" 2>/dev/null | sed -n 2p | grep -q ',Datasets/originals/raise/'; then
        refuse "target $TARGET does not have the new master_path layout (needs commit 76f6257 or later)"
    fi
    git show "$TARGET:.gitignore" 2>/dev/null | grep -qx 'runs-out/' || refuse "target $TARGET does not ignore runs-out/ (the VM checkout would look dirty)"
fi
say "  current HEAD $(git rev-parse HEAD)"
if [[ -n $(git status --porcelain) ]]; then
    ((APPLY)) && refuse 'checkout is dirty; preserve it first' || say '  NOTE: checkout is dirty (apply would refuse)'
fi

# ---------- 3. Plan the moves ----------
head1 'Photo folders'
MOVES=(
    "ImageSets/RAISE|Datasets/originals/raise"
    "ImageSets/DIV2K_train_HR|Datasets/originals/div2k/DIV2K_train_HR"
    "ImageSets/DIV2K_valid_HR|Datasets/originals/div2k/DIV2K_valid_HR"
    "Datasets/alaska/tiff_various_color|Datasets/originals/alaska/tiff_various_color"
    "Datasets/lsdir/hr|Datasets/originals/lsdir/hr"
)
repo_dev=$(stat -c %d .)
TODO_MOVES=()
for pair in "${MOVES[@]}"; do
    old=${pair%%|*}; new=${pair##*|}
    if [[ -L $old || -L $new ]]; then refuse "$old or $new is a symlink"; continue; fi
    if [[ -e $old && -e $new ]]; then refuse "both $old and $new exist (target exists)"; continue; fi
    if [[ -e $old ]]; then
        if [[ $(stat -c %d "$old") != "$repo_dev" ]]; then refuse "$old is on another filesystem; a rename is not possible"; continue; fi
        say "  $old  ->  $new"
        plan "mv $old $new"
        TODO_MOVES+=("$pair")
    elif [[ -e $new ]]; then
        say "  already in the new place: $new"
    else
        say "  absent (neither $old nor $new)"
    fi
done

# raise512-run1: PNG images deleted on the PC; keep only the small files.
RUN1=Datasets/raise512-run1
RUN1_NEW=Datasets/generated/raise512-run1-records
RUN1_DEL=(); RUN1_MV=()
if [[ -d $RUN1 ]]; then
    head1 'raise512-run1 (image folders deleted on the PC; small files kept)'
    for e in "$RUN1"/* "$RUN1"/.[!.]*; do
        [[ -e $e ]] || continue
        case "${e##*/}" in
            clean|stego|masks) RUN1_DEL+=("$e"); plan "delete $e ($(size_of "$e"))" ;;
            labels.csv|metadata.csv|dataset_info.json|splits|records)
                if [[ -e $RUN1_NEW/${e##*/} ]]; then refuse "$RUN1_NEW/${e##*/} exists"; else RUN1_MV+=("$e"); plan "mv $e $RUN1_NEW/"; fi ;;
            *) UNEXPECTED+=("$e"); say "  UNEXPECTED (left alone): $e ($(size_of "$e"))" ;;
        esac
    done
fi

# otf-cache (deleted on the PC; absent by default; V3_SOURCE=cache is optional).
head1 'otf-cache'
CACHE_DEL=()
for c in "$HOME_DIR/otf-cache" "$REPO/Datasets/otf-cache"; do
    [[ -e $c ]] || continue
    if [[ -f $c/cache.json || -f $c/v1/cache.json ]]; then
        if ((KEEP_CACHE)); then say "  keep (--keep-cache): $c ($(size_of "$c"))"
        else CACHE_DEL+=("$c"); plan "delete cache $c ($(size_of "$c"))"; fi
    else
        UNEXPECTED+=("$c"); say "  UNEXPECTED (not a cache; left alone): $c"
    fi
done
((${#CACHE_DEL[@]})) || say '  none to delete'

# Anything else under ImageSets/ and Datasets/.
head1 'Unexpected items (listed, never deleted)'
for e in ImageSets/*; do
    [[ -e $e ]] || continue
    case "${e##*/}" in RAISE|DIV2K_train_HR|DIV2K_valid_HR|README.md) ;; *) UNEXPECTED+=("$e") ;; esac
done
for e in Datasets/*; do
    [[ -e $e ]] || continue
    case "${e##*/}" in README.md|alaska|lsdir|otf-cache|raise512-run1|originals|generated) ;; *) UNEXPECTED+=("$e") ;; esac
done
for e in Datasets/alaska/*; do [[ -e $e ]] && [[ ${e##*/} != tiff_various_color ]] && UNEXPECTED+=("$e"); done
for e in Datasets/lsdir/*;  do [[ -e $e ]] && [[ ${e##*/} != hr ]] && UNEXPECTED+=("$e"); done
if ((${#UNEXPECTED[@]})); then
    for e in "${UNEXPECTED[@]}"; do printf '  %8s  %s\n' "$(size_of "$e")" "$e"; done
else say '  none'; fi

# ---------- 4. Stop here if anything is refused ----------
if ((${#REFUSALS[@]})); then
    head1 'REFUSED'
    printf '  %s\n' "${REFUSALS[@]}"
    say 'Nothing was changed. Fix the cause and run again.'
    exit 2
fi

# ---------- 5. Apply ----------
if ((APPLY)); then
    head1 'Apply'
    mkdir -p Datasets/originals Datasets/generated || exit 1
    for pair in "${TODO_MOVES[@]}"; do
        old=${pair%%|*}; new=${pair##*|}
        mkdir -p "$(dirname "$new")" || exit 1
        [[ -e $new ]] && { say "REFUSE: $new appeared; stop"; exit 2; }
        mv -- "$old" "$new" || { say "mv failed: $old"; exit 1; }
        say "  moved $old -> $new"
    done
    if ((${#RUN1_MV[@]})); then
        mkdir -p "$RUN1_NEW" || exit 1
        for e in "${RUN1_MV[@]}"; do mv -- "$e" "$RUN1_NEW/" && say "  moved $e -> $RUN1_NEW/"; done
    fi
    for e in "${RUN1_DEL[@]}" "${CACHE_DEL[@]}"; do
        [[ -n $e ]] || continue
        safe_rm "$e" && say "  deleted $e"
    done
    if ! git checkout -q --detach "$TARGET"; then
        say "git checkout of $TARGET FAILED. The photos are already moved. Fix and run:  git checkout --detach $TARGET"
        exit 1
    fi
    say "  checkout: HEAD is now $(git rev-parse HEAD) (detached)"
    for e in ImageSets Datasets/alaska Datasets/lsdir Datasets/raise512-run1; do
        [[ -d $e ]] && { rmdir -- "$e" 2>/dev/null && say "  removed empty $e" || say "  kept $e (not empty)"; }
    done
fi

# ---------- 6. runs-out inventory and optional clean ----------
head1 'runs-out job folders'
RUNS_RC=0
python3 - "$REPO" "$APPLY" "$CLEAN_RUNS" "${MANIFEST:-}" <<'PY' || RUNS_RC=$?
import os, sys, hashlib
repo, apply_, clean, manifest = sys.argv[1], sys.argv[2] == "1", sys.argv[3] == "1", sys.argv[4]
root = os.path.join(repo, "runs-out")
if not os.path.isdir(root):
    print("  no runs-out/ folder"); sys.exit(0)
def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB": return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
pi = {}
if clean:
    for line in open(manifest, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3: pi[parts[2]] = (int(parts[0]), parts[1])
def record(job):
    path = os.path.join(root, ".g4-state", "jobs", job + ".record")
    try:
        return dict(l.rstrip("\n").split("=", 1) for l in open(path, encoding="utf-8") if "=" in l)
    except OSError:
        return {}
def candidate(rel):
    name = os.path.basename(rel)
    parts = rel.split("/")
    if name.endswith(".ckpt"):
        return not (name.startswith("last") or name.startswith("best"))
    return "tb" in parts and "logs" in parts
total_cand = 0; total_removed = 0
print(f"  {'job folder':58} {'size':>8} {'delivered':>9} {'per-epoch ckpt+tb':>18} {'action'}")
for job in sorted(os.listdir(root)):
    if job.startswith("."): continue
    top = os.path.join(root, job)
    if not os.path.isdir(top):
        print(f"  {job:58} {human(os.path.getsize(top)):>8}  (file, never touched)"); continue
    size = 0; cands = []
    for dirpath, _, files in os.walk(top):
        for f in files:
            p = os.path.join(dirpath, f)
            try: s = os.path.getsize(p)
            except OSError: continue
            size += s
            rel = os.path.relpath(p, root)
            if candidate(rel): cands.append((p, rel, s))
    rec = record(job); delivered = rec.get("delivered", "no (no record)")
    cbytes = sum(c[2] for c in cands); total_cand += cbytes
    action = "keep (list only)"
    if clean:
        if rec.get("delivered") != "yes":
            action = "SKIP: VM record not delivered"
        else:
            ok = []
            for p, rel, s in cands:
                entry = pi.get(rel)
                if not entry or entry[0] != s: continue
                if entry[1] != "-" and p.endswith(".ckpt"):
                    h = hashlib.sha256()
                    with open(p, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
                    if h.hexdigest() != entry[1]: continue
                ok.append((p, s))
            skipped = len(cands) - len(ok)
            if apply_:
                for p, s in ok:
                    os.remove(p); total_removed += s
                action = f"deleted {len(ok)} files ({human(sum(s for _, s in ok))}); {skipped} not on Pi, kept"
            else:
                action = f"would delete {len(ok)} files ({human(sum(s for _, s in ok))}); {skipped} not on Pi, kept"
    print(f"  {job[:58]:58} {human(size):>8} {delivered:>9} {len(cands):>4} / {human(cbytes):>10} {action}")
print(f"  per-epoch checkpoints and tb logs in total: {human(total_cand)}; removed now: {human(total_removed)}")
print("  runs-out/.g4-state is never touched. last/best checkpoints, status, logs and results are never removed.")
PY

# ---------- 7. Check the masters ----------
head1 'Check master_path of the target commit (existence, then size+SHA for ALASKA and LSDIR)'
VERIFY_RC=0
python3 - "$REPO" "$TARGET" "$SHA" "$APPLY" <<'PY' || VERIFY_RC=$?
import csv, hashlib, io, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
repo, target, do_sha, applied = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
OLD = [("Datasets/originals/raise/", "ImageSets/RAISE/"),
       ("Datasets/originals/div2k/DIV2K_train_HR/", "ImageSets/DIV2K_train_HR/"),
       ("Datasets/originals/div2k/DIV2K_valid_HR/", "ImageSets/DIV2K_valid_HR/"),
       ("Datasets/originals/alaska/", "Datasets/alaska/"),
       ("Datasets/originals/lsdir/", "Datasets/lsdir/")]
def locate(rel):
    p = os.path.join(repo, rel)
    if os.path.exists(p): return p, False
    for new, old in OLD:
        if rel.startswith(new):
            q = os.path.join(repo, old + rel[len(new):])
            if os.path.exists(q): return q, True
    return None, False
def records(path):
    text = subprocess.check_output(["git", "-C", repo, "show", f"{target}:{path}"], text=True)
    return list(csv.DictReader(io.StringIO(text, newline="")))
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""): h.update(chunk)
    return h.hexdigest()
bad_total = 0
for source, file in (("raise", "photo_manifest.csv"), ("div2k", "photo_manifest.csv"),
                     ("alaska", "provenance.csv"), ("lsdir", "provenance.csv")):
    start = time.time()
    rows = records(f"data/{source}/{file}")
    missing = size_bad = sha_bad = at_old = 0
    jobs = []
    for row in rows:
        rel = row["master_path"]
        if rel.startswith("/") or ".." in rel.split("/"): missing += 1; continue
        path, old = locate(rel)
        if path is None: missing += 1; continue
        at_old += old
        if "bytes" in row and row["bytes"] and os.path.getsize(path) != int(row["bytes"]): size_bad += 1; continue
        if do_sha and row.get("sha256"): jobs.append((path, row["sha256"]))
    checked = 0
    if jobs:
        with ThreadPoolExecutor(16) as pool:
            for ok in pool.map(lambda j: sha(j[0]) == j[1], jobs):
                checked += 1; sha_bad += (not ok)
    bad = missing + size_bad + sha_bad; bad_total += bad
    where = f", {at_old} still in the OLD place (dry run)" if at_old else ""
    note = "" if jobs else ("  (existence only: this record holds no SHA)" if source in ("raise", "div2k") else ("  (size only, --no-sha)" if not do_sha else ""))
    print(f"  {source:7} rows={len(rows):6} missing={missing} size_bad={size_bad} sha_checked={checked} sha_bad={sha_bad}{where}  [{time.time()-start:.0f}s]{note}")
sys.exit(1 if bad_total else 0)
PY

# ---------- 8. Summary ----------
head1 "Summary ($MODE)"
say "  moves planned/done: ${#TODO_MOVES[@]} of ${#MOVES[@]}; run1 images deleted: ${#RUN1_DEL[@]}; caches deleted: ${#CACHE_DEL[@]}; unexpected left alone: ${#UNEXPECTED[@]}"
if ((RUNS_RC)); then say '  runs-out report FAILED (see above).'; exit 1; fi
if ((VERIFY_RC)); then
    say '  Master check FAILED. Do not start a job. Read the lines above.'
    exit 1
fi
if ((APPLY)); then
    say "  Done. HEAD: $(git rev-parse --short HEAD) (detached). Master check passed."
else
    say '  Dry run only; nothing was changed. Run again with --apply to act.'
fi
exit 0
