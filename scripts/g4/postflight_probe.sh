#!/usr/bin/env bash
# Host-side read transport for postflight.py (keeps SSH/gcloud credentials out
# of the ROCm container). No remote files/scripts are created.
set -uo pipefail
job=${1:-}
[[ $job =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $job != *..* ]] || exit 2
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ssh_args=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no -o ConnectTimeout=10 \
    -o ConnectionAttempts=1 -o ServerAliveInterval=5 -o ServerAliveCountMax=1 rpi400@rpi400-server)
pi=$(timeout --kill-after=2s 120s ssh "${ssh_args[@]}" "python3 -B - $job --collect" < "$here/postflight.py"); pi_rc=$?
cloud=$(timeout --kill-after=2s 40s gcloud compute instances describe rtx6000pro \
    --zone=us-central1-c --project=personal-compute-511601 --format=json --quiet); cloud_rc=$?
watchdog=$(timeout --kill-after=2s 20s ssh "${ssh_args[@]}" 'sudo -n cat /var/lib/cvdl-watchdog/status.json'); watchdog_rc=$?
config=$(timeout --kill-after=2s 20s ssh "${ssh_args[@]}" 'cat /etc/cvdl-watchdog/watchdog.json'); config_rc=$?
printf '%s\n%s\n---CVDL-POSTFLIGHT---\n%s\n%s\n---CVDL-POSTFLIGHT---\n%s\n%s\n---CVDL-POSTFLIGHT---\n%s\n%s\n' \
    "$pi_rc" "$pi" "$cloud_rc" "$cloud" "$watchdog_rc" "$watchdog" "$config_rc" "$config"
