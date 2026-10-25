# Pi host changes

This is a record of system changes on the Raspberry Pi 400 data host (`rpi400@rpi400-server`). The Pi holds the 1 TB USB SSD at `/srv`, the ALASKA and LSDIR masters, and `/srv/cvdl-results`.

## 2026-10-25: G4 watchdog service (unarmed)

Installed by the user with `sudo` from `scripts/g4/README-watchdog.md` section 3 (watchdog commit `3ac965d`). Plan: `runs/.scratch/pi-watchdog-plan-2026-10-24/PLAN.md`.

| Item | State |
| --- | --- |
| User `cvdl-watchdog` | system user, no login shell; home `/var/lib/cvdl-watchdog` (mode 700; state, `status.json`, `watchdog.jsonl`) |
| `/opt/cvdl-watchdog/` | root-owned: `google-cloud-sdk` 590.0.1 (ARM, published SHA-256 checked) and `repo` (detached at `3ac965d`) |
| `/etc/cvdl-watchdog/gcp-key.json` | root, mode 600; key `89b47d32…` of the service account `cvdl-g4-watchdog`; loaded with systemd `LoadCredential` |
| `/etc/cvdl-watchdog/watchdog.json` | `armed: false`, `audit_approved: false` |
| `cvdl-watchdog.service` | enabled and active; `Restart=on-failure`; `NoNewPrivileges`, `ProtectSystem=strict` |
| Acceptance (2026-10-25 07:34Z) | phase `UNARMED`, "read-only get/operations succeeded", status refreshed about every 77 s; `/srv/cvdl-results` and a delivered `status.json` readable by the service user; resume identity preflight on the Pi exit 0; VM metadata unchanged |

Google Cloud side (project `personal-compute-511601`): service account `cvdl-g4-watchdog`; custom role `cvdlWatchdogInstance` (instances get/start/stop/setMetadata) bound on `rtx6000pro` only; custom role `cvdlWatchdogOperations` (zoneOperations get/list) on the project. The organisation rule `iam.disableServiceAccountKeyCreation` was switched off for the project only to create the key, then restored (effective: enforced). The audit of the VM's attached account: no project or organisation roles, empty account policy, limited scopes (record outside Git, `~/.local/state/cvdl-watchdog-audit/`).

Disable: `sudo systemctl disable --now cvdl-watchdog.service` (does not stop the VM). Revoke the key: `gcloud iam service-accounts keys delete 89b47d32fd68d8bddd9b3cc2c27f1dd386dcd777 --iam-account=cvdl-g4-watchdog@personal-compute-511601.iam.gserviceaccount.com`.

## 2026-10-16: swap and journal after a freeze

On the morning of 2026-10-16 the Pi froze during two parallel large reads. The cause is unknown, because the journal was volatile and the log of that boot was lost. The Pi came back up at about 10:44 +08.

### 1. Swap file on the USB SSD removed

| item | before | after |
|---|---|---|
| `/srv/swapfile` | 8 GiB, priority -2, on the USB SSD | `swapoff` done, file deleted (8 GiB freed on `/srv`) |
| `/etc/fstab` line `/srv/swapfile none swap sw,pri=-2 0 0` | active | commented out with a dated note (not deleted); backup `/etc/fstab.bak-2026-10-16` |
| zram `/dev/zram0` | 2 GiB, priority 100 | unchanged |
| total swap (`free -m`) | 10,239 MiB | 2,047 MiB |

After the change, `systemctl daemon-reload` was run. Then `findmnt --verify` gave 0 errors.

**Note:** `rpi-swap` (`/etc/rpi/swap.conf`, all defaults) manages the zram device and its writeback file `/var/swap`. That file is 2 GiB on the SD card root (`/dev/mmcblk0p2`), not on the SSD. It is unchanged.

### 2. Persistent journal

The package drop-in `/usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf` sets `Storage=volatile`. A later drop-in overrides it:

```ini
# /etc/systemd/journald.conf.d/90-cvdl-persistent.conf
[Journal]
Storage=persistent
SystemMaxUse=200M
```

The steps were:
1. `systemd-tmpfiles --create --prefix /var/log/journal`
2. `systemctl restart systemd-journald`
3. `journalctl --flush`

After these steps:
- The journal writes to `/var/log/journal/<machine-id>/system.journal`. This is on the SD card, not the SSD.
- It used 16 MB at first.
- `journalctl --list-boots` shows the current boot. Later boots are kept up to 200 MB.

### 3. USB and SSD check (report only, nothing changed)

- **Driver:** the SSD bridge is a Realtek `0bda:9201` (Ugreen). It uses the `uas` driver at 5000M on bus 2.
- **Errors since boot:** `dmesg` shows no USB reset, `uas` abort or I/O error.
- **Power and heat:** `vcgencmd get_throttled` gave `0x0` (no under-voltage or throttling), and the temperature was 45.7 °C.
- **Quirks:** no `usb-storage.quirks` setting is on the kernel command line. This was not changed.

Not touched: `/srv/cvdl-results`, sshd, users, Tailscale. There was no reboot.
