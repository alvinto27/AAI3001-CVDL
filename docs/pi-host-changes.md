# Pi host changes

This is a record of system changes on the Raspberry Pi 400 data host (`rpi400@rpi400-server`). The Pi holds the 1 TB USB SSD at `/srv`, the ALASKA and LSDIR masters, and `/srv/cvdl-results`.

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
