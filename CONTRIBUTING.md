# Contributing to Phoenix

Thanks for your interest in Phoenix! This guide covers how to report issues and how to propose changes.

## Reporting bugs

1. Run the collector to gather a structured environment report:
   ```shell
   bash scripts/collect_bug_info.sh
   ```
   It prints a Markdown block with kernel version, NVIDIA driver, loaded modules, recent `dmesg`, and `phxfs` status.
2. Open an issue using the **Bug report** template (`.github/ISSUE_TEMPLATE/bug_report.md`) and paste the collector output plus clear reproduction steps.

> A future MCP server will aggregate these reports across users to build a shared bug knowledge base (see `doc/roadmap.md`).

## Development setup

Follow [doc/install.md](doc/install.md) to build the kernel module, library, and benchmarks. Use the `tencent-backup` branch only as a historical reference for the SC'25 artifact; active development happens on `main`.

## Pull requests

- Keep changes focused; describe the motivation and the testing you performed.
- For kernel-module changes, include `dmesg` output showing a successful `insmod` and any new test results from `test/`.
- Run the relevant benchmark/example to confirm no regression before opening a PR.

## Code style

- C: follow the existing style in `module/` and `libphoenix/` (see `CMakeLists.txt` warning flags).
- Python: 4-space indent, typed where reasonable.
- Document new public APIs in `doc/`.
