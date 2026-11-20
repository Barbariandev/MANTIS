# MIT License
# Copyright (c) 2024 MANTIS
"""Run slots: the same share of the machine for every LOOP run, and the
round schedule that fits every entry into the day.

A round's entries run at most ``slots`` at a time; the rest wait in
line.  Every run gets the same share, whatever else is running:

* CPU cores of its own (a cpuset no other run is on),
* a memory cap that covers its scratch space and its experiments,
* a cap on processes,

enforced by the kernel through one cgroup per run (cgroup v2), which
the run's sandbox and its harness experiments both sit in.  A run that
outgrows its share is killed, alone.

Every run in a round also gets the same wall clock: the round's run
window split into waves of ``slots`` runs,

    run_seconds = min(LOOP_MAX_RUN_SECONDS, window / ceil(entries / slots))

so the last wave ends inside the window however many entries came in.
200 entries on 10 slots in 20 hours is an hour each; the extreme case,
one entry for every UID (256), still fits.

    python -m autoresearch.slots      # what this machine holds
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

CGROUP_ROOT = "/sys/fs/cgroup"
PARENT = "loop-runs"                  # every run's cgroup lives under it
CONTROLLERS = ("cpuset", "cpu", "memory", "pids")
MAX_ENTRIES = 256                     # one per UID: the most a round holds
RESERVED_CPUS = 2                     # kept for the validator itself
RESERVED_MEMORY_MB = 8192
WINDOW_HOURS = 20.0                   # the rest of the day: audit, publish
MAX_RUN_SECONDS = 43200


def window_seconds() -> float:
    """How long a round's runs have, all of them (``config.LOOP_ROUND_RUN_HOURS``)."""
    try:
        import config
        hours = float(config.LOOP_ROUND_RUN_HOURS)
    except (ImportError, AttributeError):
        hours = WINDOW_HOURS
    return hours * 3600.0


def max_run_seconds() -> int:
    """The longest any one run gets (``config.LOOP_MAX_RUN_SECONDS``)."""
    try:
        import config
        return int(config.LOOP_MAX_RUN_SECONDS)
    except (ImportError, AttributeError):
        return MAX_RUN_SECONDS


def run_seconds(entries: int, slots: int, window: float,
                cap: float) -> int:
    """One run's wall clock in a round of ``entries``: the window split
    into waves of ``slots``.  Runs taken in turn as slots free up end
    within ``waves * run_seconds``, so inside the window."""
    waves = max(1, math.ceil(max(0, int(entries)) / max(1, int(slots))))
    return int(min(float(cap), float(window) / waves))


@dataclass(frozen=True)
class RunShape:
    """What one run gets.  A run mostly waits on its models: an
    experiment takes seconds of one core, so a core of its own keeps
    every run's experiments as fast as they get."""
    cpus: int = 1
    memory_mb: int = 4096           # scratch and experiments included
    scratch_mb: int = 64            # its writable /tmp (memory-backed)
    pids: int = 512

    @classmethod
    def from_config(cls) -> "RunShape":
        try:
            import config
            return cls(
                cpus=int(config.LOOP_RUN_CPUS),
                memory_mb=int(config.LOOP_RUN_MEMORY_MB),
                scratch_mb=int(config.LOOP_RUN_SCRATCH_MB),
                pids=int(config.LOOP_RUN_PIDS),
            )
        except (ImportError, AttributeError):
            return cls()

    from_env = from_config

    def describe(self) -> str:
        return (f"{cores(self.cpus)}, {self.memory_mb / 1024:g} GB memory "
                f"({self.scratch_mb} MB of it scratch), {self.pids} "
                "processes")


def cores(n: int) -> str:
    return f"{n} core" + ("" if n == 1 else "s")


def host_cpus() -> List[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:                       # not Linux
        return list(range(os.cpu_count() or 1))


def host_memory_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def fit(shape: RunShape, cpus: int, memory_mb: int) -> int:
    """How many runs of ``shape`` fit side by side next to what the
    validator keeps for itself."""
    return max(0, min((cpus - RESERVED_CPUS) // shape.cpus,
                      (memory_mb - RESERVED_MEMORY_MB) // shape.memory_mb))


def cpuset(cpus: List[int]) -> str:
    return ",".join(str(c) for c in sorted(cpus))


# ------------------------------------------------------------- cgroups

def _write(path: str, value: str) -> None:
    with open(path, "w") as f:
        f.write(value)


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def cgroups_ready(root: str = CGROUP_ROOT) -> str:
    """"" when this process can give every run a cgroup of its own;
    otherwise why it cannot."""
    if not os.path.exists(os.path.join(root, "cgroup.controllers")):
        return f"no cgroup v2 hierarchy at {root}"
    if os.geteuid() != 0:
        return "not running as root"
    missing = set(CONTROLLERS) - set(
        _read(os.path.join(root, "cgroup.controllers")).split())
    if missing:
        return f"cgroup controllers missing: {' '.join(sorted(missing))}"
    if not os.access(root, os.W_OK):
        return f"{root} is read-only here"
    return ""


def _enable(path: str) -> None:
    """Hand the controllers down to ``path``'s children."""
    have = set(_read(os.path.join(path, "cgroup.subtree_control")).split())
    want = [c for c in CONTROLLERS if c not in have]
    if want:
        _write(os.path.join(path, "cgroup.subtree_control"),
               " ".join("+" + c for c in want))


def _procs(path: str) -> List[int]:
    """Every process in the cgroup at ``path`` and below."""
    out: List[int] = []
    for dirpath, _dirs, _files in os.walk(path):
        try:
            out += [int(p) for p in _read(
                os.path.join(dirpath, "cgroup.procs")).split()]
        except (OSError, ValueError):
            pass
    return out


def _kill(path: str) -> None:
    """SIGKILL every process in the cgroup at ``path`` and below."""
    try:
        _write(os.path.join(path, "cgroup.kill"), "1")
        return
    except OSError:                    # before Linux 5.14
        pass
    for pid in _procs(path):
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _remove(path: str, wait: float = 10.0) -> bool:
    """rmdir ``path`` and its children, deepest first, waiting for
    killed processes to leave."""
    deadline = time.monotonic() + wait
    dirs = [d for d, _sub, _f in os.walk(path)][::-1]
    for d in dirs:
        while True:
            try:
                os.rmdir(d)
                break
            except FileNotFoundError:
                break
            except OSError:
                if time.monotonic() > deadline:
                    logger.warning("cgroup %s is still busy", d)
                    return False
                time.sleep(0.05)
    return True


class Envelope:
    """One run's cgroup, ``<root>/loop-runs/<name>``: the limits sit on
    it, and its two leaves, ``sandbox`` (the pipeline) and ``compute``
    (the experiments it asks for), share them."""

    def __init__(self, name: str, cpus: List[int], shape: RunShape,
                 root: str = CGROUP_ROOT):
        self.name, self.cpus, self.shape = name, list(cpus), shape
        self.rel = f"/{PARENT}/{name}"
        self.path = root + self.rel
        self.used: Dict[str, float] = {}
        os.mkdir(self.path)
        try:
            _write(os.path.join(self.path, "cpuset.cpus"), cpuset(cpus))
            _write(os.path.join(self.path, "memory.max"),
                   str(shape.memory_mb << 20))
            if os.path.exists(os.path.join(self.path, "memory.swap.max")):
                _write(os.path.join(self.path, "memory.swap.max"), "0")
            _write(os.path.join(self.path, "pids.max"), str(shape.pids))
            # out of memory: the whole run goes, not one of its processes
            _write(os.path.join(self.path, "memory.oom.group"), "1")
            _enable(self.path)
        except Exception:
            _remove(self.path)
            raise

    @property
    def cpuset(self) -> str:
        return cpuset(self.cpus)

    def leaf_rel(self, which: str) -> str:
        """``which`` leaf, as a path from the cgroup root (for runsc)."""
        return f"{self.rel}/{which}"

    def join(self, which: str) -> List[str]:
        """An argv prefix: the process moves itself into the ``which``
        leaf, then becomes the command — or never starts at all."""
        leaf = os.path.join(self.path, which)
        os.makedirs(leaf, exist_ok=True)
        return ["/bin/sh", "-c", 'echo 0 > "$1" || exit 125; shift; '
                'exec "$@"', "sh", os.path.join(leaf, "cgroup.procs")]

    def usage(self) -> Dict[str, float]:
        """CPU seconds, peak memory, and whether it ran out of memory."""
        out: Dict[str, float] = {}
        for name, key, field in (("cpu.stat", "usage_usec", "cpu_seconds"),
                                 ("memory.events", "oom_kill", "oom_kills")):
            try:
                for line in _read(os.path.join(self.path, name)).split("\n"):
                    k, _, value = line.partition(" ")
                    if k == key:
                        out[field] = (round(int(value) / 1e6, 1)
                                      if key == "usage_usec" else int(value))
            except (OSError, ValueError):
                pass
        try:
            out["memory_peak_mb"] = int(_read(
                os.path.join(self.path, "memory.peak"))) >> 20
        except (OSError, ValueError):
            pass
        return out

    def close(self) -> Dict[str, float]:
        """Kill whatever is still in the run, remove its cgroup, and
        return what it used."""
        self.used = self.usage()
        _kill(self.path)
        _remove(self.path)
        return self.used


class Slots:
    """``count`` runs side by side, each on cores of its own.  The first
    ``RESERVED_CPUS`` cores stay with the validator."""

    def __init__(self, shape: RunShape, count: Optional[int] = None,
                 cpus: Optional[List[int]] = None,
                 memory_mb: Optional[int] = None,
                 root: str = CGROUP_ROOT):
        cpus = host_cpus() if cpus is None else sorted(cpus)
        memory_mb = host_memory_mb() if memory_mb is None else memory_mb
        room = fit(shape, len(cpus), memory_mb)
        if room < 1:
            raise ValueError(
                f"this machine holds no LOOP run: one needs {shape.cpus} "
                f"cores and {shape.memory_mb} MB on top of the "
                f"{RESERVED_CPUS} cores and {RESERVED_MEMORY_MB} MB kept "
                f"for the validator (it has {len(cpus)} cores, "
                f"{memory_mb} MB)")
        if count is not None and count > room:
            logger.warning("LOOP_RUN_SLOTS=%d does not fit this machine; "
                           "running %d at a time", count, room)
        self.count = room if count is None else max(1, min(count, room))
        self.shape, self.root = shape, root
        spare = cpus[RESERVED_CPUS:]
        self._free: "queue.Queue[List[int]]" = queue.Queue()
        for i in range(self.count):
            self._free.put(spare[i * shape.cpus:(i + 1) * shape.cpus])
        _enable(root)
        parent = os.path.join(root, PARENT)
        os.makedirs(parent, exist_ok=True)
        _enable(parent)
        # the validator wins a core it shares with a run
        _write(os.path.join(parent, "cpu.weight"), "50")
        self._sweep(parent)

    @classmethod
    def from_config(cls) -> "Slots":
        """Configured slots from ``config.py`` (``LOOP_RUN_SLOTS``, ``RunShape``)."""
        try:
            import config
            want = config.LOOP_RUN_SLOTS
        except (ImportError, AttributeError):
            want = None
        return cls(RunShape.from_config(),
                   int(want) if want is not None else None)

    from_env = from_config

    @staticmethod
    def _sweep(parent: str) -> None:
        """Remove the empty cgroups a previous process left behind.  One
        with processes still in it is somebody else's run: left alone."""
        for entry in os.scandir(parent):
            if entry.is_dir() and not _procs(entry.path):
                _remove(entry.path, wait=1.0)

    def reap(self, run_id: str) -> int:
        """Kill what a previous process left running for ``run_id`` — a
        hard stop leaves its sandbox behind, on cores this process hands
        out again — and remove its cgroup.  Returns how many processes
        were killed."""
        killed = 0
        for entry in os.scandir(os.path.join(self.root, PARENT)):
            if entry.is_dir() and entry.name.rsplit("-", 1)[0] == run_id:
                killed += len(_procs(entry.path))
                _kill(entry.path)
                _remove(entry.path)
        if killed:
            logger.warning("run %s: killed %d processes a previous "
                           "process left running", run_id, killed)
        return killed

    @contextlib.contextmanager
    def held(self, run_id: str) -> Iterator[Envelope]:
        """A slot's cores, in a fresh cgroup for this run, until the
        block ends; then everything left in it is killed."""
        cpus = self._free.get()
        try:
            env = Envelope(f"{run_id}-{secrets.token_hex(3)}", cpus,
                           self.shape, self.root)
        except Exception:
            self._free.put(cpus)
            raise
        try:
            yield env
        finally:
            env.close()
            self._free.put(cpus)


# ------------------------------------------------ experiments in the run

_here = threading.local()


@contextlib.contextmanager
def compute_in(envelope: Optional[Envelope]) -> Iterator[None]:
    """Experiments started on this thread inside the block run in
    ``envelope``'s compute leaf: on the run's share, not the host's."""
    before = getattr(_here, "envelope", None)
    _here.envelope = envelope
    try:
        yield
    finally:
        _here.envelope = before


def compute_prefix() -> List[str]:
    """The argv prefix that puts an experiment in the current run's
    cgroup (empty outside a run)."""
    env = getattr(_here, "envelope", None)
    return env.join("compute") if env is not None else []


# ----------------------------------------------------------------- report

def _clock(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    return f"{seconds // 60} min"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    argparse.ArgumentParser(
        description="What this machine holds: slots, every run's share, "
                    "and every run's wall clock by round size.").parse_args(
        argv)
    from .sealed import Sandbox
    try:
        import config
        want = config.LOOP_RUN_SLOTS
        kind = str(config.LOOP_SANDBOX).strip().lower()
    except (ImportError, AttributeError):
        want = None
        kind = "gvisor"
    shape = RunShape.from_config()
    cpus, memory_mb = host_cpus(), host_memory_mb()
    room = fit(shape, len(cpus), memory_mb)
    count = max(0, min(int(want), room)) if want is not None else room
    window, cap = window_seconds(), max_run_seconds()
    why = cgroups_ready()
    print(f"machine     {len(cpus)} cores, {memory_mb / 1024:.0f} GB; "
          f"{RESERVED_CPUS} cores and {RESERVED_MEMORY_MB // 1024} GB "
          "kept for the validator")
    print(f"each run    {shape.describe()}")
    print(f"at a time   {count}")
    unsealed = Sandbox.why_not(kind)
    print(f"sandbox     {kind}: "
          + (f"NOT AVAILABLE ({unsealed})" if unsealed else "ready"))
    print(f"cgroups     {why or 'ready'}")
    print(f"round       runs share {window / 3600:g} hours; no run gets "
          f"more than {_clock(cap)}")
    if count:
        for n in sorted({1, count, 50, 100, 200, MAX_ENTRIES}):
            print(f"  {n:>4} entries -> every run gets "
                  f"{_clock(run_seconds(n, count, window, cap))}")
    ok = count >= 1 and not why and not unsealed
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
