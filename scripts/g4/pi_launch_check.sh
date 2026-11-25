#!/usr/bin/env bash
# Same unprivileged metadata check locally and over the existing admin SSH.
# No access(2), UID switch, sudo or probe writes. Mount flags after '-' are
# filesystem superblock flags, not the watchdog sandbox's per-mount ro view.
set -uo pipefail
user=${1:-}
[[ $user =~ ^[a-z_][a-z0-9_-]*$ ]] || exit 2
mount_source=0; filesystem_rw=0; upload_owner_write=0; srv_search=0; results_search=0
mount=$(awk '$5=="/srv" { n++; for(i=7;i<=NF;i++) if($i=="-" && NF==i+3) print $(i+2),$(i+3) } END { if(n!=1) exit 1 }' /proc/self/mountinfo) || mount=
if [[ $mount =~ ^([^[:space:]]+)\ ([^[:space:]]+)$ ]]; then
    [[ ${BASH_REMATCH[1]} == /dev/sda1 ]] && mount_source=1
    options=,${BASH_REMATCH[2]},
    [[ $options == *,rw,* && $options != *,ro,* ]] && filesystem_rw=1
fi
uid=$(id -u "$user" 2>/dev/null) || uid=
groups=$(id -G "$user" 2>/dev/null) || groups=
# Search follows the owner/group/other class for the named upload user.
inspect_dir() {
    local path=$1 row mode owner gid shift
    row=$(stat -L -c '%f %u %g' -- "$path" 2>/dev/null) || return 1
    [[ $row =~ ^([0-9a-fA-F]+)\ ([0-9]+)\ ([0-9]+)$ && $uid =~ ^[0-9]+$ && $groups =~ ^[0-9]+(\ [0-9]+)*$ ]] || return 1
    read -r mode owner gid <<< "$row"
    mode=$((16#$mode))
    (( (mode & 0170000) == 0040000 )) || return 1
    if [[ $path == /srv/cvdl-results && $owner == "$uid" ]] && (( mode & 0200 )); then upload_owner_write=1; fi
    if [[ $owner == "$uid" ]]; then shift=6
    elif [[ " $groups " == *" $gid "* ]]; then shift=3
    else shift=0; fi
    (( (mode >> shift) & 1 ))
}
inspect_dir /srv && srv_search=1
inspect_dir /srv/cvdl-results && results_search=1
printf 'mount_source=%s\nfilesystem_rw=%s\nupload_owner_write=%s\nsrv_search=%s\nresults_search=%s\n' \
    "$mount_source" "$filesystem_rw" "$upload_owner_write" "$srv_search" "$results_search"
