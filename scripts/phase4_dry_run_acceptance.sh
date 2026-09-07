#!/usr/bin/env bash
# Phase 4 dry-run acceptance: load each theory controller via bridge CLI.
# Usage (WSL or Git Bash):
#   bash scripts/phase4_dry_run_acceptance.sh
# Optional: PYTHON=python3 STEPS=40

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export UAM_ROOT="${UAM_ROOT:-$ROOT}"
export PYTHONPATH="$UAM_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python3}"
STEPS="${STEPS:-40}"
PASS=0
FAIL=0

echo "=== Phase 4 Dry-Run Acceptance ==="
echo "UAM_ROOT=$UAM_ROOT PYTHON=$PYTHON STEPS=$STEPS"

run_one() {
  local name="$1"
  local flag="$2"
  local needle="$3"
  local log
  log="$(mktemp)"
  echo "Testing $name (--$flag)..."
  set +e
  "$PYTHON" -u "$UAM_ROOT/ros_nodes/transport_theory_bridge.py" \
    --dry_run --steps "$STEPS" --no_accept --"$flag" \
    >"$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]] && grep -q "$needle" "$log"; then
    echo "PASS $name dry-run ($needle)"
    PASS=$((PASS + 1))
  else
    echo "FAIL $name dry-run (rc=$rc, need '$needle')"
    tail -n 20 "$log" || true
    FAIL=$((FAIL + 1))
  fi
  rm -f "$log"
}

run_one CSCBF cscbf "CSCBF active"
run_one MUBF mubf "MUBF active"
run_one SC-RISE scrise "SC-RISE active"
run_one ATAC atac "ATAC active"

echo "--- combo A: CSCBF+MUBF ---"
LOG="$(mktemp)"
"$PYTHON" -u "$UAM_ROOT/ros_nodes/transport_theory_bridge.py" \
  --dry_run --steps "$STEPS" --no_accept --cscbf --mubf >"$LOG" 2>&1
if grep -q "CSCBF active" "$LOG" && grep -q "MUBF active" "$LOG"; then
  echo "PASS combo A"; PASS=$((PASS + 1))
else
  echo "FAIL combo A"; FAIL=$((FAIL + 1)); tail -n 15 "$LOG" || true
fi
rm -f "$LOG"

echo "--- combo B: CSCBF+SC-RISE ---"
LOG="$(mktemp)"
"$PYTHON" -u "$UAM_ROOT/ros_nodes/transport_theory_bridge.py" \
  --dry_run --steps "$STEPS" --no_accept --cscbf --scrise >"$LOG" 2>&1
if grep -q "CSCBF active" "$LOG" && grep -q "SC-RISE active" "$LOG"; then
  echo "PASS combo B"; PASS=$((PASS + 1))
else
  echo "FAIL combo B"; FAIL=$((FAIL + 1)); tail -n 15 "$LOG" || true
fi
rm -f "$LOG"

echo "--- combo C: MUBF+ATAC ---"
LOG="$(mktemp)"
"$PYTHON" -u "$UAM_ROOT/ros_nodes/transport_theory_bridge.py" \
  --dry_run --steps "$STEPS" --no_accept --mubf --atac >"$LOG" 2>&1
if grep -q "MUBF active" "$LOG" && grep -q "ATAC active" "$LOG"; then
  echo "PASS combo C"; PASS=$((PASS + 1))
else
  echo "FAIL combo C"; FAIL=$((FAIL + 1)); tail -n 15 "$LOG" || true
fi
rm -f "$LOG"

echo "--- combo D: ALL ---"
LOG="$(mktemp)"
"$PYTHON" -u "$UAM_ROOT/ros_nodes/transport_theory_bridge.py" \
  --dry_run --steps "$STEPS" --no_accept --cscbf --mubf --scrise --atac >"$LOG" 2>&1
if grep -q "CSCBF active" "$LOG" && grep -q "MUBF active" "$LOG" \
  && grep -q "SC-RISE active" "$LOG" && grep -q "ATAC active" "$LOG"; then
  echo "PASS combo D"; PASS=$((PASS + 1))
else
  echo "FAIL combo D"; FAIL=$((FAIL + 1)); tail -n 15 "$LOG" || true
fi
rm -f "$LOG"

echo "=== Summary: PASS=$PASS FAIL=$FAIL ==="
[[ "$FAIL" -eq 0 ]]
