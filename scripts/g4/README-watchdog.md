# Pi G4 watchdog

This controller resumes one named **future** job. It never submits a fresh job.
Default: unarmed. No VM SSH key or training packages are needed on the Pi.
The VM still checks its full saved recipe/checkpoint before training. Native
controller preflight checks identity only, not the full PC settings report.

Limits: 3 resume intents; 15 total start calls; 240-second capacity retry delay;
$20 **estimated compute** per job, including the original boot. The default
$2/hour planning rate and $0.50 margin are not a billing guarantee. Reserve the
actual five-hour GCE STOP backstop before each resume. One unknown stopped-disk
state may get one guarded boot; a terminal disk record still cannot retrain.
Do not use `submit.sh off` to stop: it **boots** the VM in off mode.

## 1. Audit FIRST (PC, read-only)

Use administrator credentials. Do not create a key until the user has reviewed
this audit. Missing access/incomplete analysis means **STOP**, not approval.
VM metadata control can give access to the VM's attached account credentials.

```sh
P=personal-compute-511601; Z=us-central1-c; VM=rtx6000pro
A="$HOME/.local/state/cvdl-watchdog-audit"
mkdir -p "$A"; chmod 700 "$A"
gcloud compute instances describe "$VM" --project="$P" --zone="$Z" \
  --format=json > "$A/vm.json"
gcloud projects get-iam-policy "$P" --format=json > "$A/project-policy.json"
gcloud projects get-ancestors "$P" --format=json > "$A/ancestors.json"
for F in $(gcloud projects get-ancestors "$P" --filter='type=folder' --format='value(id)'); do
  gcloud resource-manager folders get-iam-policy "$F" --format=json > "$A/folder-$F.json"
done
for O in $(gcloud projects get-ancestors "$P" --filter='type=organization' --format='value(id)'); do
  gcloud organizations get-iam-policy "$O" --format=json > "$A/org-$O.json"
done
PN=$(gcloud projects describe "$P" --format='value(projectNumber)')
for VSA in $(gcloud compute instances describe "$VM" --project="$P" --zone="$Z" --format='value(serviceAccounts.email)'); do
  gcloud iam service-accounts get-iam-policy "$VSA" --project="$P" --format=json > "$A/$VSA-policy.json"
  gcloud asset analyze-iam-policy --project="$P" \
    --full-resource-name="//cloudresourcemanager.googleapis.com/projects/$PN" \
    --identity="serviceAccount:$VSA" --expand-groups --expand-roles \
    --analyze-service-account-impersonation --show-response --format=json > "$A/$VSA-effective.json"
done
```

Review `vm.json` account emails AND scopes, direct/inherited/group/custom-role
rights and impersonation paths. Expand custom roles with
`gcloud iam roles describe ROLE --project="$P" --format=json` (or
`--organization=ORG` for an organization role). For predefined roles use
`gcloud iam roles describe roles/ROLE --format=json`.
Owner/Editor/Compute Admin, IAM policy control, broad project writes or broad
account impersonation: **STOP and report before creating ANY watchdog key**.
An absent account is acceptable only after verifying `serviceAccounts` is empty.
Asset analysis must have `fullyExplored=true`, including impersonation results;
false/missing results mean STOP. Do not enable APIs or change VM rights as part
of this audit. Save the user's explicit approval outside Git. Do not copy ADC.

## 2. IAM and key (PC, ONLY after audit approval)

The first binding controls only this instance. Zonal-operation reads must be
project-wide; list filtering is not an IAM restriction. No broad predefined
Compute role, OS Login, IAP, Service Account User or Token Creator is needed.
Existing role/account names require review; do not overwrite them blindly.

```sh
(
set -eu
read -r -p 'After complete audit and user approval, type AUDIT-APPROVED: ' ACK
[ "$ACK" = AUDIT-APPROVED ] || exit 1
SA="cvdl-g4-watchdog@$P.iam.gserviceaccount.com"
gcloud iam service-accounts create cvdl-g4-watchdog --project="$P" \
  --display-name='Pi G4 watchdog'
gcloud iam roles create cvdlWatchdogInstance --project="$P" --stage=GA \
  --title='G4 instance control' \
  --permissions=compute.instances.get,compute.instances.start,compute.instances.stop,compute.instances.setMetadata
gcloud compute instances add-iam-policy-binding "$VM" --project="$P" --zone="$Z" \
  --member="serviceAccount:$SA" --role="projects/$P/roles/cvdlWatchdogInstance"
gcloud iam roles create cvdlWatchdogOperations --project="$P" --stage=GA \
  --title='Zonal operation reads' --permissions=compute.zoneOperations.get,compute.zoneOperations.list
gcloud projects add-iam-policy-binding "$P" --member="serviceAccount:$SA" \
  --role="projects/$P/roles/cvdlWatchdogOperations" --condition=None
(
  set -eu; umask 077; d=$(mktemp -d)
  trap 'rm -f "$d/key.json"; rmdir "$d"' EXIT
  gcloud iam service-accounts keys create "$d/key.json" --iam-account="$SA" --project="$P"
  ssh rpi400@rpi400-server 'umask 077; mkdir -p ~/.cache/cvdl-watchdog-setup; chmod 700 ~/.cache/cvdl-watchdog-setup'
  scp "$d/key.json" rpi400@rpi400-server:.cache/cvdl-watchdog-setup/gcp-key.json
)
)
```

Record the key ID for rotation. Revoke with `gcloud iam service-accounts keys
delete KEY_ID --iam-account="cvdl-g4-watchdog@$P.iam.gserviceaccount.com" --project="$P"`. The runtime key is root:root
600, loaded with systemd credentials; it never belongs in Git or the inbox.

## 3. Install on Pi (user sudo; NOT a VM install)

Set `W` to the **reviewed/pushed watchdog commit SHA from the handoff**, not HEAD.
The official ARM64 590.0.1 archive/hash below was checked on 2026-10-24. Recheck
Google's versioned listing before installation; stop on mismatch. Debian13's
Python3.13 is supported. Do not run uv or install Torch.
The unversioned file changes with each Google release; if the check fails, re-read the published hash at https://cloud.google.com/sdk/docs/downloads-versioned-archives and never skip the check.

```sh
(
set -eu
W=REVIEWED_WATCHDOG_COMMIT_SHA
umask 077
cd "$HOME/.cache/cvdl-watchdog-setup"
curl -fL --retry 3 -o gcloud-arm.tar.gz \
  https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-arm.tar.gz
printf '%s  gcloud-arm.tar.gz\n' c9e8618c9257b3289deb7cb5483479230d5d96230bd68a80fce17f592d0d3012 | sha256sum -c -
sudo useradd --system --user-group --create-home --home-dir /var/lib/cvdl-watchdog --shell /usr/sbin/nologin cvdl-watchdog
sudo chmod 700 /var/lib/cvdl-watchdog
sudo install -d -m 755 /opt/cvdl-watchdog /etc/cvdl-watchdog
sudo tar -xzf gcloud-arm.tar.gz -C /opt/cvdl-watchdog
sudo git clone --branch yx --single-branch https://github.com/alvinto27/AAI3001-CVDL.git /opt/cvdl-watchdog/repo
sudo git -C /opt/cvdl-watchdog/repo checkout --detach "$W"
# Public code/SDK only; root's restrictive umask must not hide them from the service.
sudo chmod -R a+rX /opt/cvdl-watchdog
sudo install -o root -g root -m 600 gcp-key.json /etc/cvdl-watchdog/gcp-key.json
sudo install -m 644 /opt/cvdl-watchdog/repo/scripts/g4/watchdog.json /etc/cvdl-watchdog/watchdog.json
sudo install -m 644 /opt/cvdl-watchdog/repo/scripts/g4/cvdl-watchdog.service /etc/systemd/system/cvdl-watchdog.service
sudo systemd-analyze verify /etc/systemd/system/cvdl-watchdog.service
sudo systemctl daemon-reload
sudo systemctl enable --now cvdl-watchdog.service
rm -f gcp-key.json gcloud-arm.tar.gz
cd "$HOME"; rmdir "$HOME/.cache/cvdl-watchdog-setup"
)
```

## 4. Unarmed acceptance (Pi; cloud reads only)

```sh
systemctl status cvdl-watchdog.service --no-pager
sudo cat /etc/cvdl-watchdog/watchdog.json
sudo cat /var/lib/cvdl-watchdog/status.json
sudo journalctl -u cvdl-watchdog.service -n 30 --no-pager
sudo -u cvdl-watchdog test -r /srv/cvdl-results
F=$(find /srv/cvdl-results -maxdepth 2 -name status.json -print -quit)
test -n "$F" && sudo -u cvdl-watchdog test -r "$F"
CLOUDSDK_PYTHON=/usr/bin/python3 /opt/cvdl-watchdog/google-cloud-sdk/bin/gcloud --version
sudo -u cvdl-watchdog env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory \
  GIT_CONFIG_VALUE_0=/opt/cvdl-watchdog/repo \
  /usr/bin/python3 -S /opt/cvdl-watchdog/repo/scripts/g4/preflight.py \
  --resume-identity-only --commit ORIGINAL_PUBLISHED_SHA --notebook ORIGINAL_NOTEBOOK
```

Require `UNARMED` / `read-only get/operations succeeded`, no adopted jobs, no
writes to the VM. Verify ordinary delivered status readability as cvdl-watchdog.
The identity smoke uses a known published commit; run it in the trusted checkout
with the service's exact Git safe.directory setting if your login differs from
root. No fake preemption/VM start/stop is an acceptance test. Check unarmed
restart/logout persistence. Only after this, record audit approval:

```sh
sudo python3 -c 'import json; p="/etc/cvdl-watchdog/watchdog.json"; c=json.load(open(p)); c["audit_approved"]=True; open(p,"w").write(json.dumps(c,indent=2)+"\n")'
```

## 5. Arm ONE future job; status and disable

Submit that future job on the PC through the normal reviewed full-preflight
workflow. The Pi checkout must contain its original commit under origin/yx;
update the stopped controller's checkout explicitly if necessary. Only boots
newer than the Pi's first unarmed installation and metadata attempt0 can be
newly enrolled. An old/current run is refused. Enrollment pins numeric VM ID,
commit, notebook hash, env, limits and price policy. Counters/latches survive
restart/disarm; never delete state to reset them.

```sh
JOB=EXACT_FUTURE_JOB_ID
sudo systemctl stop cvdl-watchdog.service
sudo git -C /opt/cvdl-watchdog/repo fetch origin yx
sudo chmod -R a+rX /opt/cvdl-watchdog/repo/.git
sudo python3 /opt/cvdl-watchdog/repo/scripts/g4/watchdog.py \
  --config /etc/cvdl-watchdog/watchdog.json --arm "$JOB"
sudo systemctl start cvdl-watchdog.service
sudo cat /var/lib/cvdl-watchdog/status.json
# Disable control; this does NOT stop the VM:
sudo systemctl disable --now cvdl-watchdog.service
sudo python3 /opt/cvdl-watchdog/repo/scripts/g4/watchdog.py \
  --config /etc/cvdl-watchdog/watchdog.json --disarm
```

From the PC: `ssh -t rpi400@rpi400-server 'sudo cat
/var/lib/cvdl-watchdog/status.json'` or `systemctl status`/`journalctl -u
cvdl-watchdog.service`. Logs: `/var/lib/cvdl-watchdog/watchdog.jsonl*` (4 ×5MiB).
For an authorized emergency stop use `gcloud compute instances stop rtx6000pro
--project=personal-compute-511601 --zone=us-central1-c`, then describe until
TERMINATED. API failures leave a visible alert; the GCE backstop is independent.

## Tests (PC only)

Use `podman exec -w /work aai3001 python scripts/required_test_level.py`, then the
required CPU level and FULL before arming. `test_g4_watchdog.py` uses fake cloud
commands/state/time, not a real VM or Pi. Keep the existing resume/checkpoint
gates; controller changes cannot bypass them.
