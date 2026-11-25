#!/usr/bin/env bash
# Transport only: bounded read-only queries. Parsing/reporting is in launch_gate.py.
# Run on the PC host (where admin SSH/gcloud are configured), or natively on Pi.
set -uo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
user=$(sed -n 's/^INBOX_USER=\([a-z_][a-z0-9_-]*\)$/\1/p' "$here/job_run.sh")
[[ $user =~ ^[a-z_][a-z0-9_-]*$ ]] || { echo 'Missing or malformed INBOX_USER configuration' >&2; exit 2; }
host=$(hostname -s) || { echo 'Cannot identify local host for Pi check' >&2; exit 2; }
[[ -n $host ]] || exit 2
if [[ $host == rpi400-server ]]; then
    pi=$(timeout --kill-after=2s 20s bash "$here/pi_launch_check.sh" "$user"); pi_rc=$?
else
    pi=$(timeout --kill-after=2s 20s ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no \
        -o ConnectTimeout=10 -o ConnectionAttempts=1 \
        -o ServerAliveInterval=5 -o ServerAliveCountMax=1 rpi400@rpi400-server \
        "bash -s -- $user" < "$here/pi_launch_check.sh"); pi_rc=$?
fi
cloud=$(timeout --kill-after=2s 40s gcloud compute instances describe rtx6000pro \
    --zone=us-central1-c --project=personal-compute-511601 --format=json --quiet); cloud_rc=$?
printf '%s\n%s\n---CVDL-GCE---\n%s\n%s\n' "$pi_rc" "$pi" "$cloud_rc" "$cloud"
