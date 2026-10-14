#!/usr/bin/env bash
# Submit one metadata job and boot rtx6000pro. --dry-run needs no gcloud install.
set -euo pipefail
ACTION=${1:-}
[[ $ACTION == run || $ACTION == maintenance || $ACTION == off || $ACTION == status ]] || {
    echo 'Usage: submit.sh {run|maintenance|off|status} [--env-file FILE] [--commit SHA] [--timeout-min 100] [--dry-run]' >&2
    exit 2
}
shift
ENV_FILE=
COMMIT=origin/yx
TIMEOUT=100
MAX_RUN=120
DRY=0
while (($#)); do
    case "$1" in
        --env-file) ENV_FILE=${2:?missing env file}; shift 2 ;;
        --commit) COMMIT=${2:?missing commit}; shift 2 ;;
        --timeout-min) TIMEOUT=${2:?missing timeout}; shift 2 ;;
        --max-run-minutes) MAX_RUN=${2:?missing maximum}; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
INSTANCE=rtx6000pro
ZONE=us-central1-c
if [[ $ACTION == run ]]; then
    [[ -f $ENV_FILE ]] || { echo 'run needs --env-file FILE' >&2; exit 2; }
    ENV_FILE=$(realpath "$ENV_FILE")
    RUN_NAME=$(grep -m1 '^V3_RUN_NAME=' "$ENV_FILE" | cut -d= -f2- || true)
    [[ $RUN_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ && $RUN_NAME != *..* ]] || {
        echo 'Set a simple V3_RUN_NAME in the env file' >&2; exit 2;
    }
    [[ $COMMIT =~ ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$ && $COMMIT != *..* ]] || exit 2
    [[ $TIMEOUT =~ ^[0-9]+$ && $MAX_RUN =~ ^[0-9]+$ ]] && \
        (( TIMEOUT > 0 && MAX_RUN >= 16 && TIMEOUT <= MAX_RUN - 15 )) || {
        echo 'timeout must leave at least 15 minutes before the GCE max run duration' >&2; exit 2;
    }
    JOB_ID="${RUN_NAME}-$(date -u +%Y%m%dT%H%M%SZ)-$(printf '%04d' "$RANDOM")"
    [[ ${#JOB_ID} -le 101 ]] || { echo 'Run name is too long for a job-id' >&2; exit 2; }
    printf 'job-id=%s\n' "$JOB_ID"
fi
command_line() { printf '+ '; printf '%q ' "$@"; printf '\n'; }
execute() {
    if (( DRY )); then command_line "$@"; else "$@"; fi
}
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
if [[ $ACTION == run ]]; then
    execute gcloud compute instances add-metadata "$INSTANCE" --zone="$ZONE" \
        --metadata="job-mode=run,job-id=$JOB_ID,job-commit=$COMMIT,job-timeout-min=$TIMEOUT,max-run-minutes=$MAX_RUN" \
        --metadata-from-file="job-env=$ENV_FILE"
else
    execute gcloud compute instances add-metadata "$INSTANCE" --zone="$ZONE" --metadata="job-mode=$ACTION"
fi
execute gcloud compute instances start "$INSTANCE" --zone="$ZONE"
