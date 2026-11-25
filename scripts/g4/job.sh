#!/usr/bin/env bash
# Stable bootstrap for rtx6000pro. setup.sh installs this file; change it rarely.
# It updates the checkout, then runs scripts/g4/job_run.sh from that checkout.
# job_run.sh holds all job logic: record, delivery, timeout, shutdown trap.
set -uo pipefail

ROOT=${G4_REPO_ROOT:-"$HOME/AAI3001-CVDL"}
STATE_ROOT=${G4_STATE_ROOT:-"$ROOT/runs-out/.g4-state"}
LIB_DIR=${G4_LIB_DIR:-"$HOME/.local/lib/cvdl-g4"}
LAST_GOOD="$LIB_DIR/job_run.last-good"
TEST=0
META_FILE=
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --test) TEST=1 ;;
        --metadata-file) META_FILE=${args[i + 1]:-} ;;
    esac
done

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$STATE_ROOT/logs" "$STATE_ROOT/jobs"
note() { printf '%s bootstrap: %s\n' "$(stamp)" "$*" | tee -a "$STATE_ROOT/logs/boot.log"; }

# The next stage must never start another bootstrap.
if [[ -n ${CVDL_G4_STAGE:-} ]]; then
    note "loop guard: CVDL_G4_STAGE is set; stop"
    exit 2
fi
SELF_SHA=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
note "bootstrap sha256=$SELF_SHA test=$TEST"
cd "$ROOT" || { note "Cannot enter $ROOT"; exit 2; }

valid_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $1 != .* && $1 != *..* ]]; }
valid_commit() { [[ $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$ && $1 != *..* ]]; }
meta() {
    curl -fsS --connect-timeout 3 --max-time 10 -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null
}

# Production run mode only: bring the checkout to the job commit before job_run.sh starts.
# A failure here is not fatal: job_run.sh repeats the step, records the failure,
# delivers the error files and shuts down.
job_mode=off
if (( TEST == 0 )); then job_mode=$(meta job-mode || true); fi
if (( TEST == 0 )) && [[ $job_mode == run || $job_mode == resume ]]; then
    job_id=$(meta job-id || true)
    commit=$(meta job-commit || true)
    commit=${commit:-origin/yx}
    eligible=0
    if [[ -n $job_id ]] && valid_id "$job_id" && valid_commit "$commit"; then
        record="$STATE_ROOT/jobs/$job_id.record"
        if [[ $job_mode == run && ! -f $record ]]; then eligible=1
        elif [[ $job_mode == resume && -f $record ]]; then
            saved_commit=$(grep -m1 '^commit=' "$record" | cut -d= -f2-)
            saved_state=$(grep -m1 '^training-state=' "$record" | cut -d= -f2-)
            saved_attempt=$(grep -m1 '^attempt=' "$record" | cut -d= -f2-)
            requested=$(meta job-attempt || true)
            if [[ $saved_commit == "$commit" && $saved_attempt =~ ^(0|[1-9][0-9]{0,8})$ && $requested =~ ^(0|[1-9][0-9]{0,8})$ \
                && $saved_state =~ ^(starting|running|interrupted)$ ]] && (( requested == saved_attempt + 1 )); then eligible=1; fi
        fi
    fi
    if (( eligible )); then
        if [[ -n $(git status --porcelain) ]]; then
            note 'Dirty checkout; no update'
        elif ! git fetch -q origin yx; then
            note 'Git fetch failed; no update'
        elif ! git checkout -q --detach "$commit"; then
            note "Git checkout of $commit failed; no update"
        else
            note "Checkout at $(git rev-parse HEAD)"
        fi
    fi
fi

# Choose the script: the checked-out copy, else the last copy that passed a syntax check.
target="$ROOT/scripts/g4/job_run.sh"
source_name=checkout
if [[ -f $target ]] && bash -n "$target" 2>/dev/null; then
    # Keep a known-good copy in case a later checkout is broken.
    if ! cmp -s "$target" "$LAST_GOOD" 2>/dev/null; then
        mkdir -p "$LIB_DIR"
        tmp=$(mktemp "$LIB_DIR/.job_run.XXXXXX") && install -m 755 "$target" "$tmp" && mv -f "$tmp" "$LAST_GOOD" \
            && note "Saved last-good copy: $LAST_GOOD"
    fi
elif [[ -f $LAST_GOOD ]] && bash -n "$LAST_GOOD" 2>/dev/null; then
    note "Checked-out job_run.sh is missing or invalid; use last-good copy"
    target=$LAST_GOOD
    source_name=last-good
else
    note "No usable job_run.sh; stop"
    if (( TEST == 0 )) && [[ $(meta job-mode || true) != maintenance ]]; then sudo shutdown -h now; fi
    exit 2
fi
export CVDL_G4_STAGE=run CVDL_G4_SCRIPT_SOURCE=$source_name CVDL_G4_BOOTSTRAP_SHA=$SELF_SHA
exec /usr/bin/env bash "$target" "$@"
