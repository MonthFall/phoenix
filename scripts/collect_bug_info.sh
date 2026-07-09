#!/usr/bin/env bash
#
# collect_bug_info.sh — produce a structured Markdown bug report for Phoenix.
#
# Safe: read-only system inspection only. No writes, no module changes.
# Paste the output into a GitHub issue (see .github/ISSUE_TEMPLATE/bug_report.md).
#
set -euo pipefail

echo "## Phoenix bug environment report"
echo
echo "### Kernel"
uname -a || true
echo
echo "### OS"
{ [ -f /etc/os-release ] && grep -E '^(NAME|VERSION)=' /etc/os-release; } || true
echo
echo "### NVIDIA driver"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
echo
echo "### Loaded NVIDIA-related modules"
lsmod 2>/dev/null | grep -i nvidia || echo "(none)"
echo
echo "### Phoenix kernel module status"
lsmod 2>/dev/null | grep -i phxfs || echo "phxfs not loaded"
echo
echo "### Phoenix char devices"
ls -l /dev/phxfs* 2>/dev/null || echo "no /dev/phxfs* devices"
echo
echo "### Recent phxfs dmesg (last 40 lines)"
dmesg 2>/dev/null | grep -i phxfs | tail -n 40 || echo "dmesg unavailable"
echo
echo "### libphoenix presence"
ldconfig -p 2>/dev/null | grep -i phoenix || echo "libphoenix not found in ldconfig"
echo
echo "### Reproduction steps"
echo "<describe how to reproduce the issue>"
echo
echo "### Expected vs actual"
echo "<what you expected> / <what happened>"
