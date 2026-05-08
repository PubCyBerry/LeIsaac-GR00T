#!/usr/bin/env bash
# scripts/usbipd.sh
# Windows Git Bash에서 SO-101 Leader/Follower 암 및 카메라를 WSL2에 attach/detach한다.
#
# 사용법:
#   bash scripts/usbipd.sh list
#   bash scripts/usbipd.sh attach
#   bash scripts/usbipd.sh detach
#   bash scripts/usbipd.sh status
#   bash scripts/usbipd.sh setup-udev
#
# 요구사항:
#   - Windows: usbipd-win (winget install usbipd)
#   - WSL2: sudo apt install linux-tools-generic hwdata

set -euo pipefail

# ── 색상 ──────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
err()     { echo -e "${RED}[ERR]${RESET}   $*" >&2; }
section() { echo -e "\n${BOLD}$*${RESET}"; }

# ── VID:PID 테이블 (SO-101 암 시리얼 어댑터) ─────────────────────────────────
# 99-feetech.rules 와 동일한 목록
SERIAL_VIDPIDS=("1a86:7523" "1a86:55d3" "10c4:ea60" "0403:6001" "0403:6014")

# ── usbipd 경로 확인 ──────────────────────────────────────────────────────────
USBIPD="$(command -v usbipd.exe 2>/dev/null || command -v usbipd 2>/dev/null || true)"
if [[ -z "$USBIPD" ]]; then
    err "usbipd를 찾을 수 없습니다."
    err "  winget install --interactive --exact dorssel.usbipd-win"
    exit 1
fi

# ── usbipd list 파싱 ──────────────────────────────────────────────────────────
# 출력 예:
#   BUSID  VID:PID    DEVICE                          STATE
#   1-1    1a86:7523  USB Serial Device (CH340)        Not shared
#   1-3    0c45:6366  USB 2.0 Camera                   Attached - 3-1 WSL

parse_devices() {
    # busid|vidpid|desc|state 형식의 배열을 전역 변수에 저장
    DEVICES=()
    local raw
    raw=$("$USBIPD" list 2>/dev/null) || { err "usbipd list 실패"; exit 1; }

    while IFS= read -r line; do
        # BUSID 패턴: N-N 형식으로 시작하는 행만 처리
        [[ "$line" =~ ^[0-9]+-[0-9]+ ]] || continue

        local busid vidpid desc state
        busid=$(echo "$line" | awk '{print $1}')
        vidpid=$(echo "$line" | awk '{print $2}')

        # VID:PID가 hex:hex 패턴인지 확인
        [[ "$vidpid" =~ ^[0-9a-fA-F]{4}:[0-9a-fA-F]{4}$ ]] || continue

        # STATE 판별: grep 기반 (bash =~ 에서 \( \) 처리 복잡성 회피)
        if echo "$line" | grep -qi 'Attached'; then
            state="Attached"
        elif echo "$line" | grep -qi 'Not shared'; then
            state="Not shared"
        elif echo "$line" | grep -qi 'Shared'; then
            state="Shared"
        elif echo "$line" | grep -qi 'Not connected'; then
            state="Not connected"
        else
            state="Unknown"
        fi

        # DEVICE 설명: 3번째 필드부터 추출 후 state 키워드 제거
        desc=$(echo "$line" | awk '{for(i=3;i<=NF;i++) printf $i" "; print ""}')
        desc=$(echo "$desc" \
            | sed 's/[[:space:]]*Attached[[:print:]]*//' \
            | sed 's/[[:space:]]*Not shared[[:print:]]*//' \
            | sed 's/[[:space:]]*Not connected$//' \
            | sed 's/[[:space:]]*Shared$//' \
            | sed 's/[[:space:]]*$//')

        DEVICES+=("${busid}|${vidpid}|${desc}|${state}")
    done <<< "$raw"
}

# 장치가 시리얼 어댑터(암)인지 확인
is_serial() {
    local vidpid="$1" desc="$2"
    for vp in "${SERIAL_VIDPIDS[@]}"; do
        [[ "$vidpid" == "$vp" ]] && return 0
    done
    echo "$desc" | grep -qiE 'CH34|CP21|FTDI|FT232|ttyACM|ttyUSB|Serial' && return 0
    return 1
}

# 장치가 카메라인지 확인
is_camera() {
    local desc="$1"
    echo "$desc" | grep -qiE 'UVC|Camera|Webcam|Video[[:space:]]+Device' && return 0
    return 1
}

# STATE 문자열이 "Attached" 상태인지 확인
is_attached() {
    echo "$1" | grep -qi 'Attached' && return 0
    return 1
}

# ── 서브커맨드: list ───────────────────────────────────────────────────────────
cmd_list() {
    section "USB 장치 목록"
    parse_devices

    printf "%-8s %-12s %-42s %s\n" "BUSID" "VID:PID" "DEVICE" "STATE"
    printf '%s\n' "$(printf '─%.0s' {1..90})"

    local found=0
    for entry in "${DEVICES[@]}"; do
        IFS='|' read -r busid vidpid desc state <<< "$entry"
        local tag=""
        if is_serial "$vidpid" "$desc"; then
            tag="${GREEN}[ARM]${RESET}"
            found=1
        elif is_camera "$desc"; then
            tag="${CYAN}[CAM]${RESET}"
            found=1
        fi
        if [[ -n "$tag" ]]; then
            printf "${BOLD}%-8s${RESET} %-12s %-42s %-20s %b\n" \
                "$busid" "$vidpid" "${desc:0:40}" "$state" "$tag"
        else
            printf "%-8s %-12s %-42s %s\n" \
                "$busid" "$vidpid" "${desc:0:40}" "$state"
        fi
    done

    [[ $found -eq 0 ]] && warn "SO-101 암 또는 카메라 후보를 감지하지 못했습니다."
    echo ""
}

# ── 서브커맨드: status ────────────────────────────────────────────────────────
cmd_status() {
    section "WSL2 Attach 상태"
    parse_devices

    local any=0
    for entry in "${DEVICES[@]}"; do
        IFS='|' read -r busid vidpid desc state <<< "$entry"
        if is_serial "$vidpid" "$desc" || is_camera "$desc"; then
            any=1
            if is_attached "$state"; then
                ok "  $busid  $desc  →  ${GREEN}${state}${RESET}"
            else
                warn "  $busid  $desc  →  ${state}"
            fi
        fi
    done
    [[ $any -eq 0 ]] && warn "SO-101 암 또는 카메라 후보를 감지하지 못했습니다."
    echo ""
}

# ── 공통: bind + attach 실행 ─────────────────────────────────────────────────
do_attach_device() {
    local busid="$1" state="$2" label="$3"

    if is_attached "$state"; then
        ok "$label ($busid): 이미 Attached 상태"
        return
    fi

    # bind가 필요한 경우
    if echo "$state" | grep -qi 'Not shared'; then
        info "$label ($busid): bind 중..."
        "$USBIPD" bind --busid "$busid" 2>&1 || {
            err "bind 실패 — Administrator 권한으로 Git Bash를 실행해 주세요."
            return 1
        }
    fi

    info "$label ($busid): WSL2에 attach 중..."
    "$USBIPD" attach --wsl --busid "$busid" 2>&1 && ok "$label ($busid): attach 완료" || {
        err "$label ($busid): attach 실패"
        return 1
    }
}

# ── 서브커맨드: attach ────────────────────────────────────────────────────────
cmd_attach() {
    section "WSL2 Attach"

    # Administrator 권한 확인 (없어도 진행은 허용)
    if ! net.exe session &>/dev/null; then
        warn "Administrator 권한이 없습니다. bind 단계에서 오류가 발생할 수 있습니다."
        warn "  → Git Bash를 '관리자 권한으로 실행' 후 재시도하세요."
    fi

    parse_devices

    # 시리얼 장치(암) 수집
    local serial_entries=()
    local cam_entries=()
    for entry in "${DEVICES[@]}"; do
        IFS='|' read -r busid vidpid desc state <<< "$entry"
        is_serial "$vidpid" "$desc" && serial_entries+=("$entry") || true
        is_camera "$desc" && cam_entries+=("$entry") || true
    done

    # ── 암 처리 ──────────────────────────────────────────────────────────────
    local leader_busid="" follower_busid=""
    local leader_state="" follower_state=""

    if [[ ${#serial_entries[@]} -eq 0 ]]; then
        warn "시리얼 장치(암)를 감지하지 못했습니다. Leader/Follower 연결을 확인하세요."
    elif [[ ${#serial_entries[@]} -eq 2 ]]; then
        # busid 오름차순 → 첫 번째=Leader, 두 번째=Follower
        IFS='|' read -r leader_busid _ _ leader_state <<< "${serial_entries[0]}"
        IFS='|' read -r follower_busid _ _ follower_state <<< "${serial_entries[1]}"
        info "시리얼 장치 2개 자동 할당:"
        info "  Leader   → busid $leader_busid"
        info "  Follower → busid $follower_busid"
    else
        echo ""
        echo "시리얼 장치가 ${#serial_entries[@]}개 감지되었습니다. Leader를 선택하세요:"
        local idx=0
        for entry in "${serial_entries[@]}"; do
            IFS='|' read -r b v d s <<< "$entry"
            printf "  [%d] busid=%-8s  %-12s  %s\n" "$idx" "$b" "$v" "$d"
            idx=$(( idx + 1 ))
        done
        read -rp "Leader 번호: " sel_l
        read -rp "Follower 번호: " sel_f
        IFS='|' read -r leader_busid _ _ leader_state <<< "${serial_entries[$sel_l]}"
        IFS='|' read -r follower_busid _ _ follower_state <<< "${serial_entries[$sel_f]}"
    fi

    [[ -n "$leader_busid"   ]] && do_attach_device "$leader_busid"   "$leader_state"   "Leader"
    [[ -n "$follower_busid" ]] && do_attach_device "$follower_busid" "$follower_state" "Follower"

    # ── 카메라 처리 ───────────────────────────────────────────────────────────
    if [[ ${#cam_entries[@]} -eq 0 ]]; then
        warn "카메라 장치를 감지하지 못했습니다."
    elif [[ ${#cam_entries[@]} -eq 1 ]]; then
        IFS='|' read -r cb _ cd cs <<< "${cam_entries[0]}"
        do_attach_device "$cb" "$cs" "Camera(${cd:0:20})"
    else
        echo ""
        echo "카메라 장치가 ${#cam_entries[@]}개 감지되었습니다. 역할을 지정하세요:"
        local idx=0
        for entry in "${cam_entries[@]}"; do
            IFS='|' read -r b v d s <<< "$entry"
            printf "  [%d] busid=%-8s  %-12s  %s\n" "$idx" "$b" "$v" "$d"
            idx=$(( idx + 1 ))
        done
        read -rp "Belly(Top) 카메라 번호: " sel_belly
        read -rp "Wrist 카메라 번호 (없으면 엔터): " sel_wrist

        IFS='|' read -r belly_b _ belly_d belly_s <<< "${cam_entries[$sel_belly]}"
        do_attach_device "$belly_b" "$belly_s" "cam_top(${belly_d:0:15})"

        if [[ -n "$sel_wrist" ]]; then
            IFS='|' read -r wrist_b _ wrist_d wrist_s <<< "${cam_entries[$sel_wrist]}"
            do_attach_device "$wrist_b" "$wrist_s" "cam_wrist(${wrist_d:0:15})"
        fi
    fi

    echo ""
    info "WSL2 내 장치 확인:"
    wsl.exe -- bash -c "ls /dev/ttyACM* /dev/ttyUSB* /dev/video* 2>/dev/null || echo '  (장치 없음)'"
    echo ""
}

# ── 서브커맨드: detach ────────────────────────────────────────────────────────
cmd_detach() {
    section "WSL2 Detach (Windows로 반환)"
    parse_devices

    local detached=0
    for entry in "${DEVICES[@]}"; do
        IFS='|' read -r busid vidpid desc state <<< "$entry"
        if (is_serial "$vidpid" "$desc" || is_camera "$desc") && is_attached "$state"; then
            info "detach: $busid  $desc"
            "$USBIPD" detach --busid "$busid" && ok "  → 완료" || warn "  → 실패"
            detached=$(( detached + 1 ))
        fi
    done

    [[ $detached -eq 0 ]] && warn "Attached 상태인 SO-101/카메라 장치가 없습니다."
    echo ""
}

# ── 서브커맨드: setup-udev ────────────────────────────────────────────────────
cmd_setup_udev() {
    section "WSL2 udev 고정 심볼릭 링크 설정"
    echo "이 단계에서는 WSL2 내에 /etc/udev/rules.d/99-so101.rules 를 생성합니다."
    echo "실행 후에는 포트 번호가 바뀌어도 /dev/so101_leader, /dev/so101_follower,"
    echo "/dev/cam_top, /dev/cam_wrist 가 항상 올바른 장치를 가리킵니다."
    echo ""

    # 장치가 attach 되어 있는지 확인
    info "WSL2 내 ttyACM 장치 확인 중..."
    local tty_list
    tty_list=$(wsl.exe -- bash -c "ls /dev/ttyACM* 2>/dev/null || true")
    if [[ -z "$tty_list" ]]; then
        err "/dev/ttyACM* 장치가 WSL2에 없습니다. 먼저 'bash scripts/usbipd.sh attach'를 실행하세요."
        exit 1
    fi

    # ── 시리얼 넘버 수집 ─────────────────────────────────────────────────────
    section "1) SO-101 암 역할 지정"
    local tty_array=()
    while IFS= read -r dev; do
        [[ -z "$dev" ]] && continue
        tty_array+=("$dev")
    done <<< "$tty_list"

    echo "감지된 ttyACM 장치:"
    local idx=0
    for dev in "${tty_array[@]}"; do
        local sn
        sn=$(wsl.exe -- bash -c \
            "udevadm info -a -n ${dev} 2>/dev/null | grep 'ATTRS{serial}' | head -1 | sed 's/.*==\"//;s/\"//'")
        printf "  [%d] %-15s  시리얼 넘버: %s\n" "$idx" "$dev" "${sn:-<없음>}"
        tty_array[$idx]="${dev}|${sn}"
        idx=$(( idx + 1 ))
    done
    echo ""

    local leader_dev="" leader_sn="" follower_dev="" follower_sn=""
    if [[ ${#tty_array[@]} -eq 2 ]]; then
        IFS='|' read -r leader_dev leader_sn <<< "${tty_array[0]}"
        IFS='|' read -r follower_dev follower_sn <<< "${tty_array[1]}"
        echo "장치 2개 감지 — 역할을 수동으로 지정하세요."
    fi

    read -rp "Leader 번호 선택 [0-$((${#tty_array[@]}-1))]: " sel_l
    IFS='|' read -r leader_dev leader_sn <<< "${tty_array[$sel_l]}"

    read -rp "Follower 번호 선택 [0-$((${#tty_array[@]}-1))]: " sel_f
    IFS='|' read -r follower_dev follower_sn <<< "${tty_array[$sel_f]}"

    if [[ -z "$leader_sn" || -z "$follower_sn" ]]; then
        err "시리얼 넘버를 읽지 못했습니다. udevadm 설치 여부를 확인하세요."
        err "  WSL2: sudo apt install -y udev"
        exit 1
    fi

    ok "Leader:   $leader_dev  (SN=$leader_sn)"
    ok "Follower: $follower_dev  (SN=$follower_sn)"

    # ── 카메라 KERNELS 수집 ──────────────────────────────────────────────────
    section "2) 카메라 역할 지정"
    info "WSL2 내 video 장치 확인 중..."
    local video_list
    video_list=$(wsl.exe -- bash -c "ls /dev/video[02468] 2>/dev/null || true")

    local cam_top_kernels="" cam_wrist_kernels=""

    if [[ -z "$video_list" ]]; then
        warn "짝수 /dev/video* 장치가 없습니다. 카메라 udev 규칙을 건너뜁니다."
    else
        local vid_array=()
        while IFS= read -r dev; do
            [[ -z "$dev" ]] && continue
            vid_array+=("$dev")
        done <<< "$video_list"

        echo "감지된 video 장치 (캡처 노드):"
        local idx=0
        for dev in "${vid_array[@]}"; do
            local kn
            kn=$(wsl.exe -- bash -c \
                "udevadm info -a -n ${dev} 2>/dev/null | grep -m1 'KERNELS==\"[0-9]' | sed 's/.*==\"//;s/\"//'")
            printf "  [%d] %-15s  KERNELS: %s\n" "$idx" "$dev" "${kn:-<없음>}"
            vid_array[$idx]="${dev}|${kn}"
            idx=$(( idx + 1 ))
        done
        echo ""

        if [[ ${#vid_array[@]} -ge 1 ]]; then
            read -rp "cam_top (Belly/전면) 번호 선택 [0-$((${#vid_array[@]}-1))]: " sel_top
            local top_dev top_k
            IFS='|' read -r top_dev top_k <<< "${vid_array[$sel_top]}"
            cam_top_kernels="$top_k"
            ok "cam_top:   $top_dev  (KERNELS=$top_k)"
        fi

        if [[ ${#vid_array[@]} -ge 2 ]]; then
            read -rp "cam_wrist (손목) 번호 선택 [없으면 엔터]: " sel_wrist
            if [[ -n "$sel_wrist" ]]; then
                local wrist_dev wrist_k
                IFS='|' read -r wrist_dev wrist_k <<< "${vid_array[$sel_wrist]}"
                cam_wrist_kernels="$wrist_k"
                ok "cam_wrist: $wrist_dev  (KERNELS=$wrist_k)"
            fi
        fi
    fi

    # ── udev rules 파일 생성 ─────────────────────────────────────────────────
    section "3) udev rules 생성 및 설치"

    local rules_content
    rules_content="# /etc/udev/rules.d/99-so101.rules
# SO-101 암 고정 심볼릭 링크 — scripts/usbipd.sh setup-udev 로 생성

# Leader arm (SN: ${leader_sn})
SUBSYSTEM==\"tty\", ATTRS{serial}==\"${leader_sn}\", SYMLINK+=\"so101_leader\", MODE=\"0666\", GROUP=\"dialout\"

# Follower arm (SN: ${follower_sn})
SUBSYSTEM==\"tty\", ATTRS{serial}==\"${follower_sn}\", SYMLINK+=\"so101_follower\", MODE=\"0666\", GROUP=\"dialout\"
"

    if [[ -n "$cam_top_kernels" ]]; then
        rules_content+="
# Belly/Top camera (KERNELS: ${cam_top_kernels})
SUBSYSTEM==\"video4linux\", KERNELS==\"${cam_top_kernels}\", ATTR{index}==\"0\", SYMLINK+=\"cam_top\"
"
    fi
    if [[ -n "$cam_wrist_kernels" ]]; then
        rules_content+="
# Wrist camera (KERNELS: ${cam_wrist_kernels})
SUBSYSTEM==\"video4linux\", KERNELS==\"${cam_wrist_kernels}\", ATTR{index}==\"0\", SYMLINK+=\"cam_wrist\"
"
    fi

    echo ""
    echo "생성될 규칙 파일 내용:"
    echo "─────────────────────────────────────────────"
    echo "$rules_content"
    echo "─────────────────────────────────────────────"
    read -rp "WSL2에 설치하시겠습니까? [y/N]: " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "취소되었습니다."; exit 0; }

    echo "$rules_content" | wsl.exe -- bash -c "sudo tee /etc/udev/rules.d/99-so101.rules > /dev/null"
    wsl.exe -- bash -c "sudo udevadm control --reload-rules && sudo udevadm trigger"

    # ── 결과 확인 ────────────────────────────────────────────────────────────
    section "4) 설치 확인"
    info "심볼릭 링크:"
    wsl.exe -- bash -c "ls -l /dev/so101_* /dev/cam_* 2>/dev/null || echo '  (심볼릭 링크 없음 — 장치 재연결 후 재확인 필요)'"

    # .env에 반영할 값 출력
    section "5) .env 업데이트 안내"
    echo "아래 값을 .env 파일에 반영하세요:"
    echo ""
    echo "  LEADER_PORT=/dev/so101_leader"
    echo "  FOLLOWER_PORT=/dev/so101_follower"

    if [[ -n "$cam_top_kernels" ]]; then
        local top_vidnum
        top_vidnum=$(wsl.exe -- bash -c \
            "readlink /dev/cam_top 2>/dev/null | sed 's/video//' || echo ''")
        echo "  BELLY_CAM_DEV=/dev/cam_top"
        if [[ "$top_vidnum" =~ ^[0-9]+$ ]]; then
            echo "  BELLY_CAM_META_DEV=/dev/video$(( top_vidnum + 1 ))   # cam_top → video${top_vidnum}"
            echo "  BELLY_CAM_INDEX=${top_vidnum}"
        else
            echo "  BELLY_CAM_META_DEV=<장치 재연결 후 확인: ls -l /dev/cam_top>"
            echo "  BELLY_CAM_INDEX=<장치 재연결 후 확인>"
        fi
    fi
    if [[ -n "$cam_wrist_kernels" ]]; then
        local wrist_vidnum
        wrist_vidnum=$(wsl.exe -- bash -c \
            "readlink /dev/cam_wrist 2>/dev/null | sed 's/video//' || echo ''")
        echo "  WRIST_CAM_DEV=/dev/cam_wrist"
        if [[ "$wrist_vidnum" =~ ^[0-9]+$ ]]; then
            echo "  WRIST_CAM_META_DEV=/dev/video$(( wrist_vidnum + 1 ))  # cam_wrist → video${wrist_vidnum}"
            echo "  WRIST_CAM_INDEX=${wrist_vidnum}"
        else
            echo "  WRIST_CAM_META_DEV=<장치 재연결 후 확인: ls -l /dev/cam_wrist>"
            echo "  WRIST_CAM_INDEX=<장치 재연결 후 확인>"
        fi
    fi

    echo ""
    ok "setup-udev 완료. 이후 'bash scripts/usbipd.sh attach' 실행 시 udev 규칙이 자동 적용됩니다."
    echo ""
}

# ── 도움말 ────────────────────────────────────────────────────────────────────
usage() {
    echo ""
    echo -e "${BOLD}사용법:${RESET}  bash scripts/usbipd.sh <subcommand>"
    echo ""
    echo "  list        USB 장치 목록 출력 (SO-101 암·카메라 하이라이트)"
    echo "  attach      Leader·Follower·카메라를 WSL2에 attach"
    echo "  detach      Attached 장치를 Windows로 반환"
    echo "  status      현재 attach 상태 요약"
    echo "  setup-udev  WSL2에 udev 고정 심볼릭 링크 규칙 설치"
    echo "              (/dev/so101_leader, /dev/so101_follower, /dev/cam_top, /dev/cam_wrist)"
    echo ""
    echo -e "${BOLD}순서:${RESET}"
    echo "  1. bash scripts/usbipd.sh attach       # 최초 또는 매 세션마다"
    echo "  2. bash scripts/usbipd.sh setup-udev   # 최초 1회 — 고정 포트 설정"
    echo "  3. docker compose up teleop            # .env에 고정 경로 반영 후"
    echo "  4. bash scripts/usbipd.sh detach       # 작업 종료 후 Windows 반환"
    echo ""
}

# ── 메인 ──────────────────────────────────────────────────────────────────────
SUBCMD="${1:-help}"
shift || true

case "$SUBCMD" in
    list)       cmd_list ;;
    attach)     cmd_attach ;;
    detach)     cmd_detach ;;
    status)     cmd_status ;;
    setup-udev) cmd_setup_udev ;;
    help|--help|-h) usage ;;
    *)
        err "알 수 없는 서브커맨드: $SUBCMD"
        usage
        exit 1
        ;;
esac
