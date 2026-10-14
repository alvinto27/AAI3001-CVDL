"""Compare the new G4 cache records to the PC and VM reference digest."""
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = "bde842a00a9060d2a3b5e6c9e214a5d6f976a52f3e494f503a1290a93c8ae482"
root = Path(sys.argv[1])
records = sorted(path for path in root.glob("*.json") if path.name != "cache.json")
if len(records) != 1900:
    raise SystemExit(f"Expected 1,900 cache records, got {len(records)}")
rows = {}
for path in records:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["photo_id"] != path.stem or record["photo_id"] in rows:
        raise SystemExit(f"Invalid cache identity: {path}")
    record.pop("master_mtime_ns", None)  # This value differs across machines.
    serialized = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rows[path.stem] = hashlib.sha256(serialized).hexdigest()
digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
print(f"Normalized cache records: {len(rows)}, sha256={digest}, reference PC/VM={EXPECTED}")
if digest != EXPECTED:
    raise SystemExit("G4 cache records differ from PC/VM (excluding master_mtime_ns)")
