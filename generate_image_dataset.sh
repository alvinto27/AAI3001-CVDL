#!/usr/bin/env bash

# Generate paired clean/stego PNG data from a flat directory of raw images.
#
# Usage:
#   ./generate_image_dataset.sh [INPUT_DIR] [OUTPUT_DIR]
#
# With no arguments, all paths are relative to the folder containing this script.
# Environment overrides: INPUT_DIR, OUTPUT_DIR, STEGO_CLI, CPU_COUNT, ERROR_LOG.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="${SCRIPT_DIR}"
readonly INPUT_DIR="${1:-${INPUT_DIR:-${PROJECT_ROOT}/dataset/input}}"
readonly OUTPUT_DIR="${2:-${OUTPUT_DIR:-${PROJECT_ROOT}/dataset/output}}"
readonly CLEAN_DIR="${OUTPUT_DIR}/clean"
readonly STEGO_DIR="${OUTPUT_DIR}/stego"
readonly ERROR_LOG="${ERROR_LOG:-${OUTPUT_DIR}/generation_error.log}"
readonly ERROR_LOCK="${ERROR_LOG}.lock"
readonly STEGO_CLI="${STEGO_CLI:-${PROJECT_ROOT}/stego_cli}"
readonly CPU_COUNT="${CPU_COUNT:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN || printf '1')}"

initialize_directories() {
    mkdir -p -- "${INPUT_DIR}" "${CLEAN_DIR}" "${STEGO_DIR}"
    touch -- "${ERROR_LOG}" "${ERROR_LOCK}"
}

log_error() {
    local image_path="$1"
    local message="$2"
    local record
    record="$(date -u +'%Y-%m-%dT%H:%M:%SZ') | ${image_path} | ${message}"

    if command -v flock >/dev/null 2>&1; then
        {
            flock -x 9
            printf '%s\n' "${record}" >> "${ERROR_LOG}"
        } 9>> "${ERROR_LOCK}"
    else
        printf '%s\n' "${record}" >> "${ERROR_LOG}"
    fi
}

find_imagemagick() {
    if command -v magick >/dev/null 2>&1; then
        printf 'magick\n'
    # Windows also has a disk utility named convert.exe. Only accept convert
    # when its version output identifies it as ImageMagick.
    elif command -v convert >/dev/null 2>&1 \
        && convert -version 2>&1 | grep -qi 'ImageMagick'; then
        printf 'convert\n'
    else
        printf 'ImageMagick is required (expected magick or convert).\n' >&2
        return 1
    fi
}

preprocess_image() {
    local source_path="$1"
    local filename stem output_path
    filename="$(basename -- "${source_path}")"
    stem="${filename%.*}"
    output_path="${INPUT_DIR}/${stem}.png"

    if [[ -e "${output_path}" && "${output_path}" != "${source_path}" ]]; then
        log_error "${source_path}" "PNG conversion skipped: ${output_path} already exists"
        return 0
    fi

    if [[ "${IM_COMMAND}" == "magick" ]]; then
        if ! magick "${source_path}" -alpha off -colorspace sRGB -type TrueColor -define png:color-type=2 "${output_path}"; then
            log_error "${source_path}" "ImageMagick conversion failed"
            return 0
        fi
    elif ! convert "${source_path}" -alpha off -colorspace sRGB -type TrueColor -define png:color-type=2 "${output_path}"; then
        log_error "${source_path}" "ImageMagick conversion failed"
        return 0
    fi

    if [[ ! -s "${output_path}" ]]; then
        log_error "${source_path}" "ImageMagick produced an empty PNG"
    fi
}

preprocess_images() {
    # COIL-100 is commonly distributed as PNG files, so do not require
    # ImageMagick unless there is actually something to convert.
    if ! find "${INPUT_DIR}" -maxdepth 1 -type f ! -iname '*.png' -print -quit \
        | grep -q .; then
        return 0
    fi

    IM_COMMAND="$(find_imagemagick)"
    export IM_COMMAND

    find "${INPUT_DIR}" -maxdepth 1 -type f ! -iname '*.png' -print0 \
        | xargs -0 -r -P "${CPU_COUNT}" -n 1 bash -c 'preprocess_image "$1"' _
}

random_integer() {
    local minimum="$1"
    local maximum="$2"
    local random_value
    random_value="$(od -An -N4 -tu4 /dev/urandom)"
    random_value="${random_value//[[:space:]]/}"
    printf '%d\n' "$((minimum + random_value % (maximum - minimum + 1)))"
}

process_image() {
    local image_path="$1"
    local filename basename_no_ext clean_output stego_output
    local copy_pid copy_status cli_status
    local PAYLOAD_SIZE LSB_DEPTH
    local -a stego_command

    filename="$(basename -- "${image_path}")"
    basename_no_ext="${filename%.*}"
    # Randomize payload bytes and LSB depth independently for every image.
    PAYLOAD_SIZE="$(random_integer 100 2000)"
    LSB_DEPTH="$(random_integer 1 4)"
    clean_output="${CLEAN_DIR}/${basename_no_ext}_clean.png"
    stego_output="${STEGO_DIR}/${basename_no_ext}_stego_lsb${LSB_DEPTH}.png"

    if [[ "${STEGO_CLI}" == "${PROJECT_ROOT}/stego_cli" ]]; then
        stego_command=(perl "${STEGO_CLI}")
    else
        stego_command=("${STEGO_CLI}")
    fi

    # Copy the clean pair while stego_cli works on the same source image.
    cp -f -- "${image_path}" "${clean_output}" &
    copy_pid=$!

    cli_status=0
    "${stego_command[@]}" --embed \
        --in "${image_path}" \
        --out "${stego_output}" \
        --size "${PAYLOAD_SIZE}" \
        --depth "${LSB_DEPTH}" \
        || cli_status=$?

    copy_status=0
    wait "${copy_pid}" || copy_status=$?

    if ((copy_status != 0)) || [[ ! -s "${clean_output}" ]]; then
        log_error "${image_path}" "clean copy failed with status ${copy_status}"
    fi

    if ((cli_status != 0)); then
        log_error "${image_path}" "stego_cli failed with status ${cli_status} (payload=${PAYLOAD_SIZE}, lsb=${LSB_DEPTH})"
    elif [[ ! -s "${stego_output}" ]]; then
        log_error "${image_path}" "stego_cli produced an empty output (payload=${PAYLOAD_SIZE}, lsb=${LSB_DEPTH})"
    fi

    return 0
}

main() {
    if (($# > 2)); then
        printf 'Usage: %s [INPUT_DIR] [OUTPUT_DIR]\n' "${0##*/}" >&2
        return 2
    fi

    initialize_directories

    if [[ ! "${CPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'CPU_COUNT must be a positive integer.\n' >&2
        return 2
    fi
    if [[ "${STEGO_CLI}" == "${PROJECT_ROOT}/stego_cli" ]]; then
        if [[ ! -r "${STEGO_CLI}" ]] || ! command -v perl >/dev/null 2>&1; then
            printf 'The project stego_cli or Perl was not found: %s\n' "${STEGO_CLI}" >&2
            return 2
        fi
    elif [[ ! -x "${STEGO_CLI}" ]] && ! command -v "${STEGO_CLI}" >/dev/null 2>&1; then
        printf 'stego_cli is not executable or was not found: %s\n' "${STEGO_CLI}" >&2
        return 2
    fi

    IM_COMMAND=""
    export PROJECT_ROOT INPUT_DIR OUTPUT_DIR CLEAN_DIR STEGO_DIR ERROR_LOG ERROR_LOCK STEGO_CLI
    export -f log_error preprocess_image random_integer process_image

    preprocess_images

    find "${INPUT_DIR}" -maxdepth 1 -type f -iname '*.png' -print0 \
        | xargs -0 -r -P "${CPU_COUNT}" -n 1 bash -c 'process_image "$1"' _

    printf 'Dataset generation complete.\nClean: %s\nStego: %s\nErrors: %s\n' \
        "${CLEAN_DIR}" "${STEGO_DIR}" "${ERROR_LOG}"
}

main "$@"
