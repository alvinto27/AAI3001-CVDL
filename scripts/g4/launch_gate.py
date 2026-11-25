"""Read-only launch checks. No training imports, network writes or privilege changes."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    'mount_source': 'Pi /srv is mounted from /dev/sda1',
    'filesystem_rw': 'Pi /srv filesystem is read-write (superblock options)',
    'upload_owner_write': 'Pi upload user owns the folder with write permission (mode bits)',
    'srv_search': 'Pi /srv is searchable for upload user (mode bits)',
    'results_search': 'Pi /srv/cvdl-results is a searchable directory for upload user (mode bits)',
    'duration': 'GCE maxRunDuration is exactly 18000 s',
    'termination': 'GCE termination action is STOP',
}


class LaunchGateError(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('invalid JSON constant')


def validate(raw, *, stream=None):
    stream = sys.stderr if stream is None else stream
    checks = {key: False for key in LABELS}
    try:
        pi, cloud = raw.split('\n---CVDL-GCE---\n')
        pi_rc, *lines = pi.splitlines()
        # No missing/duplicate/unknown keys, extra output, or non-boolean answers.
        expected = set(LABELS) - {'duration', 'termination'}
        rows = [line.split('=') for line in lines]
        if (pi_rc == '0' and len(rows) == len(expected)
                and all(len(row) == 2 and row[1] in ('0', '1') for row in rows)
                and {row[0] for row in rows} == expected):
            checks.update({key: value == '1' for key, value in rows})
        cloud_rc, document = cloud.split('\n', 1)
        if cloud_rc == '0':
            scheduling = json.loads(document, object_pairs_hook=unique_object, parse_constant=invalid_constant)['scheduling']
            duration = scheduling.get('maxRunDuration', {})
            seconds = duration.get('seconds')
            nanos = duration.get('nanos', 0)
            checks['duration'] = ((type(seconds) is int and seconds == 18000 or type(seconds) is str and seconds == '18000')
                                  and type(nanos) is int and nanos == 0)
            checks['termination'] = scheduling.get('instanceTerminationAction') == 'STOP'
    except (ValueError, KeyError, TypeError, AttributeError):
        # Leave unperformed/unparseable checks failed. Never synthesize PASS.
        pass
    for key, label in LABELS.items():
        print(f'launch gate {"PASS" if checks[key] else "FAIL"}: {label}', file=stream)
    failed = [LABELS[key] for key in checks if not checks[key]]
    if failed:
        raise LaunchGateError('launch refused: ' + '; '.join(failed))
    return checks


def check_launch():
    try:
        done = subprocess.run(['bash', str(ROOT / 'scripts/g4/launch_probe.sh')],
                              text=True, capture_output=True, timeout=70, check=True)
        raw = done.stdout
    except (OSError, subprocess.SubprocessError):
        raw = ''
    return validate(raw)


def main():
    try:
        validate(sys.stdin.read())
        return 0
    except LaunchGateError:
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
