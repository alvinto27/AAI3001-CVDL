#!/usr/bin/env bash
# Validate locally and check Pi/GCE read-only, then submit one pinned metadata job.
set -euo pipefail
ACTION=${1:-}
[[ $ACTION == run || $ACTION == resume || $ACTION == maintenance || $ACTION == off || $ACTION == status ]] || {
    echo 'Usage: submit.sh run --notebook PATH [--commit SHA] [--timeout-min N] [--dry-run]' >&2
    echo 'Resume most recent stopped job: submit.sh resume JOB [--dry-run]' >&2
    echo 'Speed tool only: submit.sh run --env-file scripts/g4/jobs/speed-1.env [options]' >&2
    exit 2
}
shift
RESUME_JOB=
if [[ $ACTION == resume ]]; then
    RESUME_JOB=${1:?resume needs a job ID}; shift
    [[ $RESUME_JOB =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $RESUME_JOB != *..* ]] || exit 2
    tail_args=("$@")
    for ((i=0;i<${#tail_args[@]};i++)); do
        case "${tail_args[i]}" in
            --dry-run) ;;
            --request-file|--start-limit) i=$((i+1)); [[ $i -lt ${#tail_args[@]} ]] || exit 2 ;;
            *) echo 'resume accepts only --dry-run, --request-file and --start-limit' >&2; exit 2 ;;
        esac
    done
fi
NOTEBOOK=
ENV_FILE=
COMMIT=HEAD
TIMEOUT=100
MAX_RUN=120
DRY=0
REQUEST_FILE=
START_LIMIT=15
while (($#)); do
    case "$1" in
        --notebook) NOTEBOOK=${2:?missing notebook}; shift 2 ;;
        --env-file) ENV_FILE=${2:?missing speed env file}; shift 2 ;;
        --commit) COMMIT=${2:?missing commit}; shift 2 ;;
        --timeout-min) TIMEOUT=${2:?missing timeout}; shift 2 ;;
        --max-run-minutes) MAX_RUN=${2:?missing maximum}; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --request-file) REQUEST_FILE=${2:?missing request file}; shift 2 ;;
        --start-limit) START_LIMIT=${2:?missing start limit}; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")/../.." rev-parse --show-toplevel)
cd "$ROOT"
if [[ -n ${CVDL_G4_CONTROLLER_PYTHON:-} ]]; then
    [[ $CVDL_G4_CONTROLLER_PYTHON == /* && ( $ACTION == resume || $ACTION == status ) ]] || {
        echo 'native controller is resume/status only' >&2; exit 2;
    }
fi
if [[ -n $REQUEST_FILE ]]; then
    [[ $ACTION == resume && -n ${CVDL_G4_CONTROLLER_PYTHON:-} ]] || {
        echo 'pending requests require explicit native resume controller' >&2; exit 2;
    }
    extra=(); (( DRY )) && extra+=(--dry-run)
    exec "$CVDL_G4_CONTROLLER_PYTHON" scripts/g4/resume_request.py "$RESUME_JOB" \
        --request-file "$REQUEST_FILE" --start-limit "$START_LIMIT" "${extra[@]}"
fi
[[ $START_LIMIT == 15 ]] || { echo '--start-limit requires --request-file' >&2; exit 2; }
INSTANCE=rtx6000pro
ZONE=us-central1-c
TEMP_ENV=
TEMP_META=
TEMP_FIELDS=
TEMP_GATE=
ATTEMPT=0
trap 'rm -f -- ${TEMP_ENV:+"$TEMP_ENV"} ${TEMP_META:+"$TEMP_META"} ${TEMP_FIELDS:+"$TEMP_FIELDS"} ${TEMP_GATE:+"$TEMP_GATE"}' EXIT
if [[ $ACTION == resume ]]; then
    command -v gcloud >/dev/null || { echo 'Install and authenticate gcloud first' >&2; exit 2; }
    TEMP_META=$(mktemp); TEMP_FIELDS=$(mktemp)
    gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --format=json > "$TEMP_META"
    if [[ -n ${CVDL_G4_CONTROLLER_PYTHON:-} ]]; then
        "$CVDL_G4_CONTROLLER_PYTHON" scripts/g4/resume.py metadata "$RESUME_JOB" < "$TEMP_META" > "$TEMP_FIELDS"
    else
        podman exec -i -w /work aai3001 python scripts/g4/resume.py metadata "$RESUME_JOB" < "$TEMP_META" > "$TEMP_FIELDS"
    fi
    mapfile -t retained < "$TEMP_FIELDS"
    [[ ${#retained[@]} == 5 ]] || exit 2
    COMMIT=${retained[0]}; NOTEBOOK=${retained[1]}; TIMEOUT=${retained[2]}; MAX_RUN=${retained[3]}; ATTEMPT=${retained[4]}
fi
if [[ $ACTION == run || $ACTION == resume ]]; then
    [[ -z $(git status --porcelain) ]] || { echo 'Dirty checkout: commit or remove changes before submission (also dry-run)' >&2; exit 2; }
    [[ $COMMIT =~ ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$ && $COMMIT != *..* ]] || { echo 'Invalid commit' >&2; exit 2; }
    COMMIT=$(git rev-parse --verify "$COMMIT^{commit}")
    git merge-base --is-ancestor "$COMMIT" refs/remotes/origin/yx || {
        echo 'Commit is not published on origin/yx; push it before submission (also dry-run)' >&2; exit 2;
    }
    [[ $TIMEOUT =~ ^[0-9]+$ && $MAX_RUN =~ ^[0-9]+$ ]] && \
        (( TIMEOUT > 0 && MAX_RUN >= 16 && TIMEOUT <= MAX_RUN - 15 )) || {
        echo 'timeout must leave at least 15 minutes before the GCE max run duration' >&2; exit 2;
    }
    if [[ -n $ENV_FILE ]]; then
        [[ -z $NOTEBOOK && -f $ENV_FILE ]] && grep -qx 'V3_SPEED_MATRIX=on' "$ENV_FILE" || {
            echo '--env-file is reserved for the speed matrix; notebook jobs use --notebook PATH' >&2; exit 2;
        }
        ENV_FILE=$(realpath "$ENV_FILE")
        NOTEBOOK=$(grep -m1 '^V3_NOTEBOOK=' "$ENV_FILE" | cut -d= -f2-)
        [[ $NOTEBOOK == notebooks/05_xunet_v4_nonorm.ipynb ]] || { echo 'Speed tool needs notebook 05 selector' >&2; exit 2; }
        podman exec -i -w /work aai3001 python scripts/g4/preflight.py --commit "$COMMIT" --speed-env-stdin < "$ENV_FILE"
        label=g4-speed
        echo 'Speed tool: scripts/run_notebook.py dispatches scripts/speed_matrix.py; legacy tool env retained.'
    else
        [[ -n $NOTEBOOK ]] || { echo 'run needs --notebook PATH' >&2; exit 2; }
        # Python, including validation in --dry-run, runs only in the ROCm container.
        if [[ -n ${CVDL_G4_CONTROLLER_PYTHON:-} ]]; then
            "$CVDL_G4_CONTROLLER_PYTHON" scripts/g4/preflight.py --commit "$COMMIT" --notebook "$NOTEBOOK" --resume-identity-only
        else
            podman exec -w /work aai3001 python scripts/g4/preflight.py --commit "$COMMIT" --notebook "$NOTEBOOK"
        fi
        label=$(basename "$NOTEBOOK" .ipynb)
        label=$(printf '%s' "$label" | tr -cs 'A-Za-z0-9_-' '-' | cut -c1-50)
        [[ $label =~ ^[A-Za-z0-9] ]] || label="nb-$label"
    fi
    if [[ $ACTION == resume ]]; then JOB_ID=$RESUME_JOB
    else JOB_ID="${label}-$(date -u +%Y%m%dT%H%M%SZ)-$(printf '%04d' "$((RANDOM % 10000))")"; fi
    if [[ -z $ENV_FILE ]]; then
        TEMP_ENV=$(mktemp)
        printf 'V3_NOTEBOOK=%s\nCVDL_JOB_ID=%s\n' "$NOTEBOOK" "$JOB_ID" > "$TEMP_ENV"
        ENV_FILE=$TEMP_ENV
        printf 'job-env (generated; no experiment values):\n'
        while IFS= read -r line; do printf '  %s\n' "$line"; done < "$ENV_FILE"
    fi
    printf 'job-id=%s\njob-commit=%s\n' "$JOB_ID" "$COMMIT"
fi
if [[ $ACTION == run || $ACTION == resume ]]; then
    # Query on the host: its existing admin SSH and gcloud credentials stay there.
    # Python parsing runs in the container, or natively for the Pi controller.
    TEMP_GATE=$(mktemp)
    if ! bash scripts/g4/launch_probe.sh > "$TEMP_GATE"; then : > "$TEMP_GATE"; fi
    if [[ -n ${CVDL_G4_CONTROLLER_PYTHON:-} ]]; then
        "$CVDL_G4_CONTROLLER_PYTHON" scripts/g4/launch_gate.py < "$TEMP_GATE"
    else
        podman exec -i -w /work aai3001 python scripts/g4/launch_gate.py < "$TEMP_GATE"
    fi
fi
command_line() { printf '+ '; printf '%q ' "$@"; printf '\n'; }
execute() { if (( DRY )); then command_line "$@"; else "$@"; fi; }
if (( ! DRY )); then command -v gcloud >/dev/null || { echo 'Install and authenticate gcloud first' >&2; exit 2; }; fi
if [[ $ACTION == status ]]; then
    execute gcloud compute instances describe "$INSTANCE" --zone="$ZONE" \
        '--format=value(status,networkInterfaces[0].accessConfigs[0].natIP)'
    exit
fi
if (( DRY )); then
    command_line gcloud compute instances describe "$INSTANCE" --zone="$ZONE" '--format=value(status)'
else
    current=$(gcloud compute instances describe "$INSTANCE" --zone="$ZONE" '--format=value(status)')
    [[ $current == TERMINATED || $current == STOPPED ]] || {
        echo "VM status is $current. Stop it before submitting metadata for the next boot." >&2; exit 2;
    }
fi
if [[ $ACTION == run || $ACTION == resume ]]; then
    execute gcloud compute instances add-metadata "$INSTANCE" --zone="$ZONE" \
        --metadata="job-mode=$ACTION,job-id=$JOB_ID,job-commit=$COMMIT,job-timeout-min=$TIMEOUT,max-run-minutes=$MAX_RUN,job-protocol=resume-v1,job-attempt=$ATTEMPT" \
        --metadata-from-file="job-env=$ENV_FILE"
else
    execute gcloud compute instances add-metadata "$INSTANCE" --zone="$ZONE" --metadata="job-mode=$ACTION"
fi
execute gcloud compute instances start "$INSTANCE" --zone="$ZONE"
