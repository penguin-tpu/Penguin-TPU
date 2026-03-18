#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGET="${TARGET:-uart_hello}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyUSB0}"
SERIAL_BAUD="${SERIAL_BAUD:-115200}"
SERIAL_TIMEOUT="${SERIAL_TIMEOUT:-6}"
PROGRAM_RETRIES="${PROGRAM_RETRIES:-3}"
STEP_TIMEOUT="${STEP_TIMEOUT:-300}"
BITSTREAM_TIMEOUT="${BITSTREAM_TIMEOUT:-1800}"
PROGRAM_TIMEOUT="${PROGRAM_TIMEOUT:-120}"
CLEAN_FIRST=1

run_with_timeout() {
    local timeout_s="$1"
    local description="$2"
    shift 2

    if ! command -v timeout >/dev/null 2>&1; then
        echo "'timeout' is required for bounded FPGA flow execution" >&2
        exit 2
    fi

    timeout --foreground "${timeout_s}s" "$@"
    local status=$?
    if [[ "${status}" -eq 124 ]]; then
        echo "${description} timed out after ${timeout_s}s" >&2
    fi
    return "${status}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
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
        --step-timeout)
            STEP_TIMEOUT="$2"
            shift 2
            ;;
        --bitstream-timeout)
            BITSTREAM_TIMEOUT="$2"
            shift 2
            ;;
        --program-timeout)
            PROGRAM_TIMEOUT="$2"
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

if [[ "${TARGET}" == "sclar_core" ]]; then
    TARGET="scalar_core"
fi

case "${TARGET}" in
    uart_hello)
        EXPECT_MESSAGE="Hello World"
        MIN_OCCURRENCES=2
        MIN_PERIOD=0.90
        MAX_PERIOD=1.10
        ;;
    scalar_core)
        EXPECT_MESSAGE="Hello World"
        MIN_OCCURRENCES=1
        MIN_PERIOD=""
        MAX_PERIOD=""
        ;;
    *)
        echo "unsupported target: ${TARGET}" >&2
        exit 2
        ;;
esac

cd "${REPO_ROOT}"

if [[ "${CLEAN_FIRST}" -eq 1 ]]; then
    rm -rf "${REPO_ROOT}/VivadoProject"
    rm -f "${REPO_ROOT}"/vivado*.log "${REPO_ROOT}"/vivado*.jou
fi

export PENGUIN_VIVADO_TARGET="${TARGET}"

run_with_timeout "${STEP_TIMEOUT}" "vivado project creation" \
    vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/1_create_project.tcl"
run_with_timeout "${STEP_TIMEOUT}" "vivado add-files step" \
    vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/2_add_files.tcl"
run_with_timeout "${STEP_TIMEOUT}" "vivado ip-generation step" \
    vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/3_generate_vivado_ip.tcl"
run_with_timeout "${BITSTREAM_TIMEOUT}" "vivado bitstream generation" \
    vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/4_generate_bitstream.tcl"

for ((attempt = 1; attempt <= PROGRAM_RETRIES; attempt += 1)); do
    if run_with_timeout "${PROGRAM_TIMEOUT}" "vivado device programming" \
        vivado -mode batch -source "${REPO_ROOT}/scripts/vivado/5_program_device.tcl"; then
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
    --expect "${EXPECT_MESSAGE}" \
    --min-occurrences "${MIN_OCCURRENCES}" \
    ${MIN_PERIOD:+--min-period "${MIN_PERIOD}"} \
    ${MAX_PERIOD:+--max-period "${MAX_PERIOD}"}
