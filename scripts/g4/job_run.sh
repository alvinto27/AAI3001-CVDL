#!/usr/bin/env bash
# Job logic for rtx6000pro. The installed bootstrap (job.sh) runs the checked-out copy of this file.
# Production uses GCE instance attributes; --test is PC-only.
set -uo pipefail

ROOT=${G4_REPO_ROOT:-"$HOME/AAI3001-CVDL"}
STATE_ROOT=${G4_STATE_ROOT:-"$ROOT/runs-out/.g4-state"}
MODE=production
META_FILE=
INBOX_HOST=rpi400-server
INBOX_USER=cvdl-inbox
INBOX_KEY="$HOME/.ssh/cvdl-g4-inbox"
RETRY_SECONDS=300
while (($#)); do
    case "$1" in
        --test) MODE=test; shift ;;
        --metadata-file|--inbox-host|--inbox-key|--retry-seconds)
            (($# >= 2)) || { echo "missing value for $1" >&2; exit 2; }
            case "$1" in
                --metadata-file) META_FILE=$2 ;;
                --inbox-host) INBOX_HOST=$2 ;;
                --inbox-key) INBOX_KEY=$2 ;;
                --retry-seconds) RETRY_SECONDS=$2 ;;
            esac
            shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ $MODE == production && ( -n $META_FILE || $INBOX_HOST != rpi400-server || $INBOX_KEY != "$HOME/.ssh/cvdl-g4-inbox" || $RETRY_SECONDS != 300 ) ]]; then
    echo "test overrides require --test" >&2; exit 2
fi
[[ $MODE == production || -f $META_FILE ]] || { echo "--test requires --metadata-file" >&2; exit 2; }
[[ $RETRY_SECONDS =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$STATE_ROOT/jobs" "$STATE_ROOT/env" "$STATE_ROOT/logs"
cd "$ROOT" || exit 2
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$STATE_ROOT/logs/boot.log"; }
# Record which copy of this script runs, for the boot log.
SELF_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
note "job_run copy: ${CVDL_G4_SCRIPT_SOURCE:-direct} path=$SELF_PATH sha256=$(sha256sum < "$SELF_PATH" | cut -d' ' -f1) bootstrap-sha256=${CVDL_G4_BOOTSTRAP_SHA:-none}"

metadata() {
    local key=$1 dest=$2
    if [[ $MODE == test ]]; then
        local value
        value=$(grep -m1 "^${key}=" "$META_FILE" || true)
        [[ -n $value ]] || return 1
        value=${value#*=}
        if [[ $key == job-env ]]; then
            # The test file points at a local env file; do not execute or source it.
            [[ -f $value ]] || return 1
            cp -- "$value" "$dest"
        else
            printf '%s' "$value" > "$dest"
        fi
    else
        curl -fsS --connect-timeout 3 --max-time 10 \
            -H 'Metadata-Flavor: Google' \
            "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$key" \
            -o "$dest" 2>/dev/null
    fi
}
meta_value() {
    local key=$1 fallback=$2 path="$STATE_ROOT/env/meta-$$-$1"
    if metadata "$key" "$path"; then
        META_RESULT=$(<"$path")
    else
        META_RESULT=$fallback
    fi
    rm -f "$path"
}
valid_id() { [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $1 != .* && $1 != *..* ]]; }
valid_commit() { [[ $1 =~ ^[a-zA-Z0-9][a-zA-Z0-9._/-]*$ && $1 != *..* ]]; }
field() { grep -m1 "^$2=" "$STATE_ROOT/jobs/$1.record" 2>/dev/null | cut -d= -f2- || true; }
helper() {
    if [[ $MODE == test ]]; then
        local converted=() arg
        for arg in "$@"; do converted+=("${arg/#$ROOT\//\/work\/}"); done
        podman exec -w /work aai3001 python scripts/g4/resume.py "${converted[@]}"
    else "$ROOT/.venv/bin/python" "$ROOT/scripts/g4/resume.py" "$@"; fi
}
write_record() {
    local id=$1 status=$2 started=$3 ended=$4 code=$5 delivered=$6
    local path="$STATE_ROOT/jobs/$id.record"
    local commit notebook attempt env_sha training
    commit=$(field "$id" commit); notebook=$(field "$id" notebook); attempt=$(field "$id" attempt); env_sha=$(field "$id" env-sha256)
    training=${7:-$(field "$id" training-state)}
    if [[ $id == "${JOB_ID:-}" && ${OWN_JOB:-0} == 1 ]]; then
        commit=$COMMIT; notebook=$NOTEBOOK; attempt=$ATTEMPT; env_sha=$ENV_SHA
    fi
    printf 'job-id=%s\nstate=%s\nstarted-at=%s\nended-at=%s\nexit-code=%s\ndelivered=%s\nprotocol=resume-v1\ncommit=%s\nnotebook=%s\nattempt=%s\nenv-sha256=%s\ntraining-state=%s\n' \
        "$id" "$status" "$started" "$ended" "$code" "$delivered" "$commit" "$notebook" "$attempt" "$env_sha" "$training" | helper record "$path"
}
shutdown_vm() {
    if [[ $MODE == test ]]; then note 'TEST SHUTDOWN -h now'; else note 'shutdown -h now'; sudo shutdown -h now; fi
}
deliver() {
    local id=$1 out="$ROOT/runs-out/$1" began=$SECONDS end remain connect rc spent
    [[ -d $out ]] || { note "Pending $id has no folder; leave pending"; return 1; }
    if [[ ! -f $INBOX_KEY ]]; then note "Missing inbox key: $INBOX_KEY"; return 1; fi
    if (( DELIVERY_BUDGET <= 0 )); then note "Pi retry budget exhausted for $id"; return 1; fi
    end=$((SECONDS + DELIVERY_BUDGET))
    local sync_options=() attempt_number
    attempt_number=$(field "$id" attempt)
    if [[ $attempt_number =~ ^(0|[1-9][0-9]{0,8})$ ]] && (( attempt_number > 0 )); then sync_options=(--delete-delay); fi
    local ready=0 ready_end=$((SECONDS + 30))
    (( ready_end > end )) && ready_end=$end
    while (( SECONDS < ready_end )); do
        if { [[ $MODE == test ]] || tailscale status >/dev/null 2>&1; } && \
           getent ahostsv4 "$INBOX_HOST" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if (( ready == 0 )); then
        note "Tailscale or $INBOX_HOST unavailable; leave $id pending"
        spent=$((SECONDS - began))
        DELIVERY_BUDGET=$((DELIVERY_BUDGET - spent))
        return 1
    fi
    while (( SECONDS < end )); do
        remain=$((end - SECONDS))
        connect=5; (( remain < connect )) && connect=$remain
        note "Upload $id (up to ${remain}s left)"
        if rsync -r -t --partial "${sync_options[@]}" --timeout="$remain" \
            -e "ssh -i $INBOX_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=$connect -o StrictHostKeyChecking=accept-new" \
            -- "$out" "$INBOX_USER@$INBOX_HOST:."; then
            note "Delivered $id"
            spent=$((SECONDS - began))
            DELIVERY_BUDGET=$((DELIVERY_BUDGET - spent))
            return 0
        else
            rc=$?
            note "Upload $id failed (rsync exit $rc)"
        fi
        (( SECONDS >= end )) && break
        if [[ $MODE == test ]]; then sleep 1; else sleep 5; fi
    done
    spent=$((SECONDS - began))
    DELIVERY_BUDGET=$((DELIVERY_BUDGET - spent))
    note "Upload $id retry deadline reached"
    return 1
}
deliver_pending() {
    local path id started ended code
    shopt -s nullglob
    for path in "$STATE_ROOT"/jobs/*.record; do
        id=${path##*/}; id=${id%.record}
        valid_id "$id" || continue
        if [[ $(field "$id" training-state) =~ ^(starting|running)$ ]]; then
            local recovered
            recovered=$(helper outcome "$ROOT/runs-out/$id" unknown) || recovered=failed
            write_record "$id" pending "$(field "$id" started-at)" "$(stamp)" unknown no "$recovered"
            helper attempt "$STATE_ROOT/jobs/$id.attempts.jsonl" "$ROOT/runs-out/$id" "$id" "$(field "$id" attempt)" end 'abrupt stop detected on next boot; time is detection time' || true
            [[ -d "$ROOT/runs-out/$id" && -f "$STATE_ROOT/jobs/$id.attempts.jsonl" ]] && cp --remove-destination "$STATE_ROOT/jobs/$id.attempts.jsonl" "$ROOT/runs-out/$id/attempts.jsonl"
        fi
        [[ $(field "$id" delivered) == yes ]] && continue
        note "Resend pending $id before new job"
        started=$(field "$id" started-at); ended=$(field "$id" ended-at); code=$(field "$id" exit-code)
        [[ -n $ended ]] || ended=$(stamp)  # Detection time after an abrupt stop.
        [[ -n $code ]] || code=unknown
        if deliver "$id"; then
            write_record "$id" delivered "$started" "$ended" "$code" yes
        else
            write_record "$id" pending "$started" "$ended" "$code" no
        fi
    done
}

meta_value job-mode off
JOB_MODE=$META_RESULT
if [[ $JOB_MODE == maintenance ]]; then
    note 'Maintenance mode: no upload, no shutdown (GCE max duration still applies).'
    exit 0
fi
DELIVERY_BUDGET=$RETRY_SECONDS
if [[ $JOB_MODE != run && $JOB_MODE != resume ]]; then
    deliver_pending
    note "Job mode $JOB_MODE: pending results sent if possible; stop"
    shutdown_vm
    [[ $JOB_MODE == off ]] && exit 0 || exit 2
fi
JOB_ID=
STARTED=
EXIT_CODE=0
RUN_STARTED=0
RUN_PID=
MONITOR_PID=
RUN_ENV=
COMMIT=
NOTEBOOK=
ATTEMPT=0
ENV_SHA=
OWN_JOB=0
RESUMING=0
FAILURE_REASON=
check_update_units() {
    local unit load active bad=()
    for unit in apt-daily.timer apt-daily-upgrade.timer apt-daily.service apt-daily-upgrade.service unattended-upgrades.service; do
        if ! load=$(systemctl show "$unit" --property=LoadState --value); then load=query-failed; fi
        if ! active=$(systemctl show "$unit" --property=ActiveState --value); then active=query-failed; fi
        if [[ $load != masked || $active != inactive ]]; then
            bad+=("$unit(LoadState=$load,ActiveState=$active)")
        fi
    done
    if ((${#bad[@]})); then
        FAILURE_REASON="automatic-update policy refused: ${bad[*]}"
        note "$FAILURE_REASON"
        printf '%s %s\n' "$(stamp)" "$FAILURE_REASON" >> "$STATE_ROOT/logs/$JOB_ID-run.log"
        return 1
    fi
    note 'Automatic-update units are masked and inactive'
}
cleanup() {
    local rc=$?
    trap - EXIT
    if (( RUN_STARTED == 0 || rc != 0 )); then EXIT_CODE=$rc; fi
    if [[ -n $RUN_PID ]] && kill -0 "$RUN_PID" 2>/dev/null; then
        kill -TERM "$RUN_PID" 2>/dev/null || true
        wait "$RUN_PID" 2>/dev/null || true
    fi
    if [[ -n $MONITOR_PID ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    if (( OWN_JOB )) && [[ -n $JOB_ID ]] && valid_id "$JOB_ID" && [[ -f "$STATE_ROOT/jobs/$JOB_ID.record" ]]; then
        local out="$ROOT/runs-out/$JOB_ID" finished
        mkdir -p "$out"
        # Notebook 05 keeps the full cell output only in run.log. Preserve a
        # small console record only if the runner failed before run.log existed,
        # or a pre-run policy refusal occurred (resume can have an old run.log).
        if [[ $NOTEBOOK == notebooks/04_baseline_kv_xunet_v3.ipynb || ! -f $out/run.log || -n $FAILURE_REASON ]]; then
            [[ -f "$STATE_ROOT/logs/$JOB_ID-run.log" ]] && cp "$STATE_ROOT/logs/$JOB_ID-run.log" "$out/job-console.log"
        fi
        [[ -n $RUN_ENV && -f $RUN_ENV ]] && cp "$RUN_ENV" "$out/job.env"
        finished=$(stamp)
        local training reason
        training=$(helper outcome "$out" "$EXIT_CODE") || training=failed
        reason="$training exit=$EXIT_CODE"
        [[ -z $FAILURE_REASON ]] || reason+="; $FAILURE_REASON"
        helper attempt "$STATE_ROOT/jobs/$JOB_ID.attempts.jsonl" "$out" "$JOB_ID" "$ATTEMPT" end "$reason" || true
        [[ -f "$STATE_ROOT/jobs/$JOB_ID.attempts.jsonl" ]] && cp --remove-destination "$STATE_ROOT/jobs/$JOB_ID.attempts.jsonl" "$out/attempts.jsonl"
        if deliver "$JOB_ID"; then
            write_record "$JOB_ID" delivered "$STARTED" "$finished" "$EXIT_CODE" yes "$training"
        else
            write_record "$JOB_ID" pending "$STARTED" "$finished" "$EXIT_CODE" no "$training"
        fi
    fi
    note "Job process exit=$rc, runner exit=$EXIT_CODE"
    shutdown_vm
}
trap cleanup EXIT
trap 'note "Job service received SIGTERM"; EXIT_CODE=143; exit 143' TERM
trap 'note "Job service received SIGINT"; EXIT_CODE=130; exit 130' INT
# Pending uploads are attempted even when this boot has no valid or new job-id.
deliver_pending
meta_value job-id ''
JOB_ID=$META_RESULT
if [[ -z $JOB_ID ]]; then note 'No job-id; stop'; exit 0; fi
if ! valid_id "$JOB_ID"; then note 'Invalid job-id; stop'; JOB_ID=; exit 2; fi
exec 9>"$STATE_ROOT/jobs/$JOB_ID.lock"
if ! flock -n 9; then note 'Job already locked'; JOB_ID=; exit 2; fi
if [[ $JOB_MODE == run && -f "$STATE_ROOT/jobs/$JOB_ID.record" ]]; then
    note "Job-id $JOB_ID already recorded; do not run again"; JOB_ID=; exit 0
fi
if [[ $JOB_MODE == resume && ! -f "$STATE_ROOT/jobs/$JOB_ID.record" ]]; then note 'Resume has no job record'; JOB_ID=; exit 2; fi
STARTED=$(stamp)
if [[ $JOB_MODE == run ]]; then
    OWN_JOB=1
    if ! write_record "$JOB_ID" starting "$STARTED" '' '' no starting; then OWN_JOB=0; exit 2; fi
    if ! helper attempt "$STATE_ROOT/jobs/$JOB_ID.attempts.jsonl" "$ROOT/runs-out/$JOB_ID" "$JOB_ID" 0 start run; then exit 2; fi
fi
meta_value job-commit origin/yx
COMMIT=$META_RESULT
if ! valid_commit "$COMMIT"; then note 'Invalid commit'; COMMIT=; exit 2; fi
meta_value job-timeout-min 100
TIMEOUT_MIN=$META_RESULT
meta_value max-run-minutes 120
MAX_RUN_MIN=$META_RESULT
if [[ ! $TIMEOUT_MIN =~ ^[0-9]+([.][0-9]+)?$ || ! $MAX_RUN_MIN =~ ^[0-9]+$ ]] || \
   ! awk -v t="$TIMEOUT_MIN" -v m="$MAX_RUN_MIN" 'BEGIN { exit !(t > 0 && m >= 16 && t <= m - 15) }'; then
    note "Invalid job timeout: $TIMEOUT_MIN min; max-run=$MAX_RUN_MIN min; must be <= max-run minus 15"
    exit 2
fi
meta_value job-attempt 0
ATTEMPT=$META_RESULT
[[ $ATTEMPT =~ ^(0|[1-9][0-9]{0,8})$ ]] || { note 'Invalid attempt'; exit 2; }
if [[ $JOB_MODE == resume ]]; then
    RESUMING=1
    saved_attempt=$(field "$JOB_ID" attempt)
    if [[ $(field "$JOB_ID" job-id) != "$JOB_ID" || $(field "$JOB_ID" protocol) != resume-v1 || $(field "$JOB_ID" training-state) != interrupted \
        || $(field "$JOB_ID" commit) != "$COMMIT" || ! $saved_attempt =~ ^(0|[1-9][0-9]{0,8})$ ]] \
        || (( ATTEMPT != saved_attempt + 1 )); then note 'Resume record/state/commit/attempt mismatch'; JOB_ID=; exit 2; fi
    STARTED=$(field "$JOB_ID" started-at)
else
    (( ATTEMPT == 0 )) || { note 'Fresh run must have attempt zero'; exit 2; }
fi
RUN_ENV="$STATE_ROOT/env/$JOB_ID.env"
(( RESUMING )) && RUN_ENV="$STATE_ROOT/env/$JOB_ID-attempt-$ATTEMPT.env"
if ! metadata job-env "$RUN_ENV"; then note 'Missing job-env metadata'; exit 2; fi
chmod 600 "$RUN_ENV"
# Metadata is data, never shell code. Validate against the job checkout below.
NOTEBOOK=$(grep -m1 '^V3_NOTEBOOK=' "$RUN_ENV" | cut -d= -f2- || true)
ENV_SHA=$(printf '%s' "$(<"$RUN_ENV")" | sha256sum | cut -d' ' -f1)
if (( RESUMING )) && [[ $(field "$JOB_ID" notebook) != "$NOTEBOOK" || $(field "$JOB_ID" env-sha256) != "$ENV_SHA" ]]; then
    note 'Resume notebook/settings env mismatch'; JOB_ID=; exit 2
fi
OWN_JOB=1
if ! write_record "$JOB_ID" starting "$STARTED" '' '' no starting; then note 'Cannot persist attempt claim'; OWN_JOB=0; exit 2; fi
if (( RESUMING )) && ! helper attempt "$STATE_ROOT/jobs/$JOB_ID.attempts.jsonl" "$ROOT/runs-out/$JOB_ID" "$JOB_ID" "$ATTEMPT" start "$JOB_MODE"; then
    note 'Could not record attempt start'; exit 2
fi
# The attempt is now ours. Refuse before checkout/package work or training;
# EXIT cleanup must still record, deliver and shut down this failed attempt.
if ! check_update_units; then exit 2; fi
if (( RESUMING )) && ! helper checkpoint "$ROOT/runs-out/$JOB_ID" > "$STATE_ROOT/jobs/$JOB_ID-checkpoint-in.json"; then
    note 'Resume needs a readable full last checkpoint; refuse fresh start'; exit 2
fi
if [[ $MODE == production ]]; then
    if [[ -n $(git status --porcelain) ]]; then note 'Dirty checkout; refuse git checkout'; exit 2; fi
    # The bootstrap has normally done this already. Bash reads a script while it runs,
    # so a checkout that changes this file now would be unsafe: stop in that case.
    BEFORE_HEAD=$(git rev-parse HEAD)
    if ! git fetch origin yx; then note 'Git fetch failed'; exit 2; fi
    if ! git checkout --detach "$COMMIT"; then note 'Git checkout failed'; exit 2; fi
    if [[ $(git rev-parse HEAD) != "$BEFORE_HEAD" ]]; then
        note "Checkout moved from $BEFORE_HEAD to $(git rev-parse HEAD) after job_run.sh started; stop. Use a new job-id."
        exit 2
    fi
    if ! "$HOME/.local/bin/uv" sync --locked; then note 'uv sync --locked failed'; exit 2; fi
else
    note "TEST: skip checkout and uv sync for $COMMIT (keep PC ROCm container intact)"
fi
if [[ $MODE == test ]]; then
    [[ $RUN_ENV == "$ROOT/"* ]] || { note 'Test state root must be inside the checkout'; exit 2; }
    env_in_container="/work/${RUN_ENV#"$ROOT"/}"
    validation=(podman exec -w /work aai3001 python scripts/g4/preflight.py --job-env "$env_in_container" --job-id "$JOB_ID" --commit HEAD)
else
    validation=("$ROOT/.venv/bin/python" scripts/g4/preflight.py --job-env "$RUN_ENV" --job-id "$JOB_ID" --commit "$COMMIT")
fi
if ! NOTEBOOK=$("${validation[@]}"); then note 'Invalid pinned notebook or job-env; refuse training'; exit 2; fi
note "Run $JOB_ID commit=$COMMIT timeout=${TIMEOUT_MIN}m notebook=$NOTEBOOK"
if [[ $MODE == test ]]; then
    [[ $RUN_ENV == "$ROOT/"* ]] || { note 'Test state root must be inside the checkout'; exit 2; }
    env_in_container="/work/${RUN_ENV#"$ROOT"/}"
    runner=(podman exec -w /work aai3001 timeout --signal=TERM --kill-after=20s "${TIMEOUT_MIN}m" python scripts/run_notebook.py --notebook "$NOTEBOOK" --env-file "$env_in_container" --output "runs-out/$JOB_ID")
else
    runner=(env CVDL_EXPECTED_PROFILE=g4 "$ROOT/.venv/bin/python" scripts/run_notebook.py --notebook "$NOTEBOOK" --env-file "$RUN_ENV" --output "runs-out/$JOB_ID" --host-monitor on)
fi
if (( RESUMING )); then runner+=(--resume); fi
if ! write_record "$JOB_ID" running "$STARTED" '' '' no running; then note 'Cannot persist running state'; exit 2; fi
RUN_STARTED=1
if [[ $MODE == test ]]; then
    "${runner[@]}" >> "$STATE_ROOT/logs/$JOB_ID-run.log" 2>&1 &
else
    timeout --signal=TERM --kill-after=30s "${TIMEOUT_MIN}m" "${runner[@]}" >> "$STATE_ROOT/logs/$JOB_ID-run.log" 2>&1 &
fi
RUN_PID=$!
# Do not put telemetry in the runner folder until it has started: it must be empty.
while kill -0 "$RUN_PID" 2>/dev/null; do
    if [[ -f "$ROOT/runs-out/$JOB_ID/status.json" ]]; then
        (
            trap 'jobs -pr | xargs -r kill 2>/dev/null || true; exit' TERM INT
            while :; do
                if [[ $MODE == test ]]; then
                    printf '%s,TEST (no NVIDIA on PC)\n' "$(stamp)"
                else
                    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
                        --format=csv,noheader,nounits || true
                fi
                sleep 60 &
                wait $!
            done
        ) > "$ROOT/runs-out/$JOB_ID/nvidia.csv" 2>&1 &
        MONITOR_PID=$!
        break
    fi
    sleep 0.5
done
if wait "$RUN_PID"; then EXIT_CODE=0; else EXIT_CODE=$?; fi
RUN_PID=
note "Runner $JOB_ID exit=$EXIT_CODE"
exit "$EXIT_CODE"
