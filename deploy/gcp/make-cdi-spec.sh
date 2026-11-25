#!/bin/sh
# Write an NVIDIA CDI spec that Podman 4.9 (Ubuntu 24.04) can read. Run as root.
# nvidia-ctk 1.20 writes cdiVersion 0.7.0; Podman 4.9 reads up to 0.6.0. The only 0.7.0
# field is additionalGids; the /dev/nvidia* nodes are mode 0666, so the groups are not needed.
# The refresh units would write a 0.7.0 spec to /var/run/cdi at boot, so they stay masked.
# Run again after an NVIDIA driver update.
set -eu
systemctl disable --now nvidia-cdi-refresh.path nvidia-cdi-refresh.service 2>/dev/null || true
systemctl mask nvidia-cdi-refresh.path nvidia-cdi-refresh.service >/dev/null
rm -f /var/run/cdi/nvidia.yaml
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
python3 - <<'PY'
import re
p = "/etc/cdi/nvidia.yaml"; s = open(p).read()
s = s.replace("cdiVersion: 0.7.0", "cdiVersion: 0.6.0", 1)
s = re.sub(r"\n( *)additionalGids:\n(?:\1 *- \d+\n)+", "\n", s)
open(p, "w").write(s)
PY
nvidia-ctk cdi list
