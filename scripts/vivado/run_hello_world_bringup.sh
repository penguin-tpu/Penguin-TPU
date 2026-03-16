#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SERIAL_PORT="${SERIAL_PORT:-/dev/ttyUSB0}"
SERIAL_BAUD="${SERIAL_BAUD:-115200}"
SERIAL_TIMEOUT="${SERIAL_TIMEOUT:-6}"
PROGRAM_RETRIES="${PROGRAM_RETRIES:-3}"
CLEAN_FIRST=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            SERIAL_PORT="$2"
            shift 2
            ;;
        --baud)
            SERIAL_BAUD="$2"
            shift 2
            ;;
        --timeout)
            SERIAL_TIMEOUT="$2"
            shift 2
            ;;
        --program-retries)
            PROGRAM_RETRIES="$2"
            shift 2
            ;;
        --skip-clean)
            CLEAN_FIRST=0
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

cd "${REPO_ROOT}"

if [[ "${CLEAN_FIRST}" -eq 1 ]]; then
    rm -rf "${REPO_ROOT}/VivadoProject"
    rm -f "${REPO_ROOT}"/vivado*.log "${REPO_ROOT}"/vivado*.jou
fi

vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/1_create_project.tcl"
vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/2_add_files.tcl"
vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/3_generate_bitstream.tcl"

for ((attempt = 1; attempt <= PROGRAM_RETRIES; attempt += 1)); do
    if vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/4_program_device.tcl"; then
        break
    fi

    if [[ "${attempt}" -eq "${PROGRAM_RETRIES}" ]]; then
        echo "programming failed after ${PROGRAM_RETRIES} attempts" >&2
        exit 1
    fi

    echo "programming attempt ${attempt} failed; retrying..." >&2
    sleep 1
done

uv run python "${REPO_ROOT}/scripts/vivado/read_uart_hello.py" \
    --port "${SERIAL_PORT}" \
    --baud "${SERIAL_BAUD}" \
    --timeout "${SERIAL_TIMEOUT}" \
    --expect "Hello World"
