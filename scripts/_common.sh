#!/usr/bin/env bash
# Shared helpers for SoccerStack scripts

source_env() {
  if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
  fi
}

init_counters() {
  pass_count=0
  warn_count=0
  fail_count=0
}

hit_pass() { pass_count=$((pass_count + 1)); }
hit_warn() { warn_count=$((warn_count + 1)); }
hit_fail() { fail_count=$((fail_count + 1)); }

# check_exists "file|dir" "path" "label"
# Uses global STRICT variable (0=warn on missing, 1=fail on missing)
check_exists() {
  local kind="$1"
  local path="$2"
  local label="$3"
  local ok=0
  if [[ "$kind" == "file" && -f "$path" ]]; then ok=1; fi
  if [[ "$kind" == "dir" && -d "$path" ]]; then ok=1; fi
  if [[ "$ok" -eq 1 ]]; then
    echo "[PASS] $label -> $path"
    hit_pass
  else
    if [[ "${STRICT:-0}" -eq 1 ]]; then
      echo "[FAIL] $label 缺失 -> $path"
      hit_fail
    else
      echo "[WARN] $label 缺失 -> $path"
      hit_warn
    fi
  fi
}
