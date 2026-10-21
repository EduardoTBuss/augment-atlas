"""SYNTHETIC stand-in for nvidia-smi used by the sentinel tests. Not real output.

Mode from FAKE_NVIDIA_SMI_MODE (default ``ok``):
  ok        queries print 2 synthetic GPUs; ``-lms N`` streams every N ms forever
  timeout   every invocation sleeps (the caller's timeout/watchdog must fire)
  stall     the stream emits FAKE_NVIDIA_SMI_STALL_AFTER ticks (default 3), then
            stays alive without output; with FAKE_NVIDIA_SMI_STATE=<file> only the
            first launch stalls, later launches stream normally
  exit / exit-9  the stream emits FAKE_NVIDIA_SMI_STALL_AFTER ticks, then exits 9;
            queries exit 9 immediately
  na        values contain [N/A] and [Not Supported]
Other knobs: FAKE_NVIDIA_SMI_GPUS (default 2), FAKE_NVIDIA_SMI_APPS=1 (one compute app on GPU 0),
FAKE_NVIDIA_SMI_SLOW_TIMEOUT=1 (only non-stream queries sleep, the stream works).
"""
import math
import os
import sys
import time

MODE = os.environ.get("FAKE_NVIDIA_SMI_MODE", "ok")
GPUS = int(os.environ.get("FAKE_NVIDIA_SMI_GPUS", "2"))
STALL_AFTER = int(os.environ.get("FAKE_NVIDIA_SMI_STALL_AFTER", "3"))
UNITS = {"power.draw": "W", "power.limit": "W", "fan.speed": "%", "utilization.gpu": "%", "memory.used": "MiB",
         "memory.total": "MiB", "clocks.sm": "MHz", "clocks.mem": "MHz"}


def uuid(index):
    return f"GPU-00000000-0000-0000-0000-{index:012d}"


def value(field, index, tick, units):
    phase = tick * 0.5
    table = {
        "index": str(index), "uuid": uuid(index), "name": "SYNTHETIC GPU",
        "temperature.gpu": str(40 + index + tick % 3),
        "power.draw": f"{100.0 + 50.0 * index + 10.0 * math.sin(phase):.2f}",
        "power.limit": "250.00", "fan.speed": str(30 + index), "utilization.gpu": str(90 - index),
        "memory.used": str(1000 + index), "memory.total": "12288", "clocks.sm": "1189", "clocks.mem": "3505",
        "pstate": "P2", "compute_cap": "5.2", "pcie.link.gen.current": "3", "pcie.link.width.current": "16",
        "clocks_event_reasons.active": "0x0000000000000004",
    }
    text = table.get(field, "[N/A]")
    if MODE == "na" and field in {"power.draw", "fan.speed"}:
        text = "[N/A]" if field == "power.draw" else "[Not Supported]"
    if units and field in UNITS and not text.startswith("["):
        text += " " + UNITS[field]
    return text


def emit(fields, tick, units):
    for index in range(GPUS):
        sys.stdout.write(", ".join(value(field, index, tick, units) for field in fields) + "\n")
    sys.stdout.flush()


def launch_number():
    state = os.environ.get("FAKE_NVIDIA_SMI_STATE")
    if not state:
        return 1
    try:
        with open(state, "r", encoding="utf-8") as handle:
            count = int(handle.read().strip() or "0")
    except (OSError, ValueError):
        count = 0
    with open(state, "w", encoding="utf-8") as handle:
        handle.write(str(count + 1))
    return count + 1


Q_TEXT = """
==============NVSMI LOG==============

Timestamp                                              : SYNTHETIC
Driver Version                                         : 000.00.00
CUDA Version                                           : 0.0

Attached GPUs                                          : {count}
"""
Q_GPU = """GPU 00000000:{bus:02X}:00.0
    Performance State                                  : P2
    Clocks Event Reasons
        Idle                                           : Not Active
        SW Power Cap                                   : Active
        HW Slowdown                                    : Not Active
    ECC Mode
        Current                                        : N/A
        Pending                                        : N/A
    Temperature
        GPU Current Temp                               : 41 C
        GPU Shutdown Temp                              : 97 C
        GPU Slowdown Temp                              : 92 C
        GPU Target Temperature                         : 83 C
    GPU Power Readings
        Instantaneous Power Draw                       : 101.00 W
        Current Power Limit                            : 250.00 W
        Default Power Limit                            : 250.00 W
        Min Power Limit                                : 150.00 W
        Max Power Limit                                : 275.00 W
"""


def main(argv):
    query = next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--query-gpu=")), None)
    fmt = next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--format=")), "csv")
    loop_ms = int(argv[argv.index("-lms") + 1]) if "-lms" in argv else None
    if MODE == "timeout" or (os.environ.get("FAKE_NVIDIA_SMI_SLOW_TIMEOUT") == "1" and loop_ms is None):
        time.sleep(3600)
        return 0
    if MODE in {"exit", "exit-9"} and loop_ms is None:
        return 9
    if "--version" in argv:
        print("NVIDIA-SMI version  : 000.00.00 (SYNTHETIC)\nDRIVER version      : 000.00.00\nCUDA Version        : 0.0")
        return 0
    if "-q" in argv:
        sys.stdout.write(Q_TEXT.format(count=GPUS) + "\n".join(Q_GPU.format(bus=0x17 + index) for index in range(GPUS)))
        return 0
    if any(arg.startswith("--query-compute-apps") for arg in argv):
        if os.environ.get("FAKE_NVIDIA_SMI_APPS") == "1":
            print(uuid(0))
        return 0
    if query is None:
        print("SYNTHETIC fake nvidia-smi: unsupported arguments", file=sys.stderr)
        return 2
    fields = query.split(",")
    units = "nounits" not in fmt
    if "noheader" not in fmt:
        print(", ".join(f + (f" [{UNITS[f]}]" if units and f in UNITS else "") for f in fields))
    if loop_ms is None:
        emit(fields, 0, units)
        return 0
    stall = MODE == "stall" and launch_number() == 1
    tick = 0
    while True:
        if (stall or MODE in {"exit", "exit-9"}) and tick >= STALL_AFTER:
            if stall:
                time.sleep(3600)
            return 9
        emit(fields, tick, units)
        tick += 1
        time.sleep(loop_ms / 1000.0)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(0)
