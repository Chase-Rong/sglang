"""Follow one remote joint-budget matrix run and refresh the comparison workbook."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import shlex
import stat as stat_module
import sys
import tempfile
import time

import paramiko

try:
    from ssh_auth import connect_host
except ImportError:  # script directory not on sys.path
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from ssh_auth import connect_host

REPO = Path(__file__).resolve().parents[2]
ARMS = ("offload64", "baseline64")
COORDINATOR = "baseline_coordinator"
COORDINATOR_STATUS = COORDINATOR + "/status.json"
RECEIPTS = ("matrix/summary.json", "matrix/final.json")
ARTIFACTS = ("coordinator.log", "coordinator_owner.json")
EXCLUDED_DIRECTORIES = {"tmp", "profile", "source"}
TERMINAL_COORDINATOR = {"completed", "baseline_failed", "baseline_start_failed", "blocked"}
ABSENT_ERRORS = (2, 3, 20)
RECONNECT_ATTEMPTS = 3
EXPECTED_CASES = 8


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_export_workbook():
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from export_fsa_matrix_xlsx import export_workbook
    return export_workbook


class Follower:
    def __init__(self, arguments, password):
        root = arguments.root
        if not root.startswith("/home/j00628475/") or ".." in root.split("/"):
            raise ValueError("Remote root must stay under /home/j00628475")
        self.root = root.rstrip("/")
        self.name = self.root.rsplit("/", 1)[-1]
        if self.name in ("", ".", ".."):
            raise ValueError("Remote root must name a directory under /home/j00628475")
        self.host = arguments.host
        self.container = arguments.container
        self.workbook = Path(arguments.workbook).resolve()
        self.interval = max(1, arguments.interval)
        self.timeout_hours = arguments.timeout_hours
        self.password = password
        self.base = REPO / "results" / "joint_budget" / self.host / self.name
        self.status_path = self.base / "follower_status.json"
        self.client = None
        self.stats = {}
        self.synced = {}
        self.runner_alive = {}
        self.container_first = None
        self.last_snapshot = None
        self.status = {
            "host": self.host, "root": self.root, "container": self.container,
            "base": str(self.base), "started_at": timestamp(), "updated_at": None,
            "interval_s": self.interval, "timeout_hours": self.timeout_hours,
            "state": "starting", "success": False, "last_sync": None,
            "workbook": {"requested": str(self.workbook), "path": None}, "export_error": None,
            "coordinator": {}, "offload64": {}, "baseline64": {},
            "container_started_at": None, "downloads": {"files": 0, "bytes": 0}, "last_cycle": {},
            "terminal": None,
        }

    def connect(self):
        self.close()
        client = connect_host(self.host, password=self.password or None)
        self.client = client

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    def secure(self, operation):
        for attempt in range(RECONNECT_ATTEMPTS + 1):
            try:
                if self.client is None:
                    self.connect()
                return operation()
            except (paramiko.SSHException, EOFError, OSError) as error:
                self.close()
                if attempt >= RECONNECT_ATTEMPTS:
                    raise
                print(f"[{timestamp()}] transport {type(error).__name__}; reconnect {attempt + 1}/{RECONNECT_ATTEMPTS}",
                      flush=True)
                time.sleep(min(5 * (attempt + 1), self.interval))

    def command(self, command, check=True):
        _, output, errors = self.client.exec_command(shlex.join(command), timeout=30)
        stdout = output.read().decode(errors="replace")
        stderr = errors.read().decode(errors="replace")
        code = output.channel.recv_exit_status()
        if check and code:
            raise RuntimeError(f"remote exit {code}: {stderr.strip()[:200]}")
        return code, stdout, stderr

    def remote_stat(self, transfer, path):
        try:
            return transfer.stat(path)
        except IOError as error:
            if getattr(error, "errno", None) in ABSENT_ERRORS:
                return None
            raise

    def remote_listdir(self, transfer, path):
        try:
            return transfer.listdir_attr(path)
        except IOError as error:
            if getattr(error, "errno", None) in ABSENT_ERRORS or "No such file" in str(error):
                return None
            raise

    def read_json(self, transfer, path):
        try:
            with transfer.open(path) as source:
                return json.loads(source.read().decode(errors="replace"))
        except ValueError:
            return None
        except IOError as error:
            if getattr(error, "errno", None) in ABSENT_ERRORS:
                return None
            raise

    def container_info(self):
        code, stdout, _ = self.command(
            ["docker", "inspect", "--format", "{{.State.StartedAt}}|{{.State.Status}}", self.container], check=False)
        if code:
            return None
        fields = stdout.strip().split("|")
        if len(fields) != 2:
            return None
        return {"started_at": fields[0], "status": fields[1]}

    def probe_runner(self, pid):
        try:
            identifier = int(pid)
        except (TypeError, ValueError):
            return {"exists": None, "error": "unreadable pid"}
        code = ("import json,pathlib;"
                f"f=pathlib.Path('/proc/{identifier}/stat');"
                "print(json.dumps({'exists':f.exists(),"
                "'start_ticks':f.read_text().split(')',1)[1].split()[19] if f.exists() else None,"
                "'state':f.read_text().split(')',1)[1].split()[0] if f.exists() else None}))")
        status, stdout, stderr = self.command(["docker", "exec", self.container, "python3", "-c", code], check=False)
        if status:
            return {"exists": None, "error": stderr.strip()[:200]}
        try:
            return json.loads(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"exists": None}

    def poll_remote(self):
        snapshot = {"stats": {}, "summaries": {}, "finals": {}, "progress": {}, "runners": {},
                    "runner_states": {}, "coordinator": None, "container": None, "changed": []}
        receipts = [f"{self.root}/{arm}/{suffix}" for arm in ARMS for suffix in RECEIPTS]
        receipts.append(f"{self.root}/{COORDINATOR_STATUS}")
        with self.client.open_sftp() as transfer:
            for path in receipts:
                information = self.remote_stat(transfer, path)
                snapshot["stats"][path] = (None if information is None
                                           else [int(information.st_size), int(information.st_mtime)])
            for arm in ARMS:
                prefix = f"{self.root}/{arm}"
                snapshot["summaries"][arm] = self.read_json(transfer, f"{prefix}/matrix/summary.json")
                snapshot["finals"][arm] = self.read_json(transfer, f"{prefix}/matrix/final.json")
                snapshot["progress"][arm] = self.read_json(transfer, f"{prefix}/matrix/progress.json")
                snapshot["runners"][arm] = self.read_json(transfer, f"{prefix}/matrix_runner.json")
            snapshot["coordinator"] = self.read_json(transfer, f"{self.root}/{COORDINATOR_STATUS}")
        snapshot["changed"] = [path for path, value in snapshot["stats"].items() if value != self.stats.get(path)]
        self.stats = snapshot["stats"]
        snapshot["container"] = self.container_info()
        self.inspect_runners(snapshot)
        return snapshot

    def inspect_runners(self, snapshot):
        for arm in ARMS:
            record = snapshot["runners"].get(arm)
            if snapshot["finals"].get(arm) is not None:
                snapshot["runner_states"][arm] = {"status": "finished"}
                continue
            if not isinstance(record, dict) or "pid" not in record:
                snapshot["runner_states"][arm] = {"status": "absent"}
                continue
            probe = self.probe_runner(record["pid"])
            if probe.get("exists") is None:
                snapshot["runner_states"][arm] = {"status": "unknown", "probe": probe}
                continue
            alive = (bool(probe.get("exists")) and str(probe.get("start_ticks")) == str(record.get("start_ticks"))
                     and probe.get("state") != "Z")
            if not alive:
                with self.client.open_sftp() as transfer:
                    final = self.read_json(transfer, f"{self.root}/{arm}/matrix/final.json")
                if final is not None:
                    snapshot["finals"][arm] = final
                    snapshot["runner_states"][arm] = {"status": "finished"}
                    snapshot["changed"].append(f"{self.root}/{arm}/matrix/final.json")
                    continue
            if alive:
                self.runner_alive[arm] = True
            snapshot["runner_states"][arm] = {
                "status": "alive" if alive else "gone",
                "observed_alive": self.runner_alive.get(arm, False),
                "record": {"pid": record.get("pid"), "start_ticks": record.get("start_ticks")},
                "probe": probe,
            }

    def targets(self, changed):
        selected = []
        for path in changed:
            relative = path[len(self.root) + 1:]
            head = relative.split("/")[0]
            if relative == COORDINATOR_STATUS:
                selected.append((self.root + "/" + COORDINATOR, self.base / COORDINATOR))
                selected.extend((self.root + "/" + name, self.base / name) for name in ARTIFACTS)
            elif head in ARMS:
                selected.append((self.root + "/" + head, self.base / head))
        unique = []
        seen = set()
        for remote, local in selected:
            if remote not in seen:
                seen.add(remote)
                unique.append((remote, local))
        return unique

    def final_targets(self):
        selected = [(self.root + "/" + arm, self.base / arm) for arm in ARMS]
        selected.append((self.root + "/" + COORDINATOR, self.base / COORDINATOR))
        selected.extend((self.root + "/" + name, self.base / name) for name in ARTIFACTS)
        return selected

    def collect(self, targets):
        counters = {"changed": bool(targets), "files": 0, "bytes": 0, "skipped": 0, "errors": 0}
        if not targets:
            return counters
        self.base.mkdir(parents=True, exist_ok=True)
        with self.client.open_sftp() as transfer:
            for remote, local in targets:
                information = self.remote_stat(transfer, remote)
                if information is None:
                    continue
                if stat_module.S_ISDIR(information.st_mode):
                    self.download_tree(transfer, remote, local, counters)
                elif stat_module.S_ISREG(information.st_mode):
                    self.download_file(transfer, remote, local, information, counters)
        return counters

    def download_tree(self, transfer, remote, local, counters):
        entries = self.remote_listdir(transfer, remote)
        if entries is None:
            return
        local.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            source = remote + "/" + entry.filename
            target = local / entry.filename
            if stat_module.S_ISDIR(entry.st_mode):
                if entry.filename not in EXCLUDED_DIRECTORIES:
                    self.download_tree(transfer, source, target, counters)
            elif stat_module.S_ISREG(entry.st_mode) and not entry.filename.endswith((".tmp", ".part")):
                self.download_file(transfer, source, target, entry, counters)

    def download_file(self, transfer, remote, local, information, counters):
        key = [int(information.st_size), int(information.st_mtime)]
        if self.synced.get(remote) == key and local.is_file():
            counters["skipped"] += 1
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(suffix=".part", dir=local.parent, delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            with transfer.open(remote) as source, temporary.open("wb") as output:
                remaining = int(information.st_size)
                while remaining:
                    block = source.read(min(remaining, 1 << 20))
                    if not block:
                        raise OSError("Remote file shortened during collection")
                    output.write(block)
                    remaining -= len(block)
        except (FileNotFoundError, IOError) as error:
            temporary.unlink(missing_ok=True)
            if isinstance(error, FileNotFoundError) or getattr(error, "errno", None) in ABSENT_ERRORS:
                counters["errors"] += 1
                print(f"[{timestamp()}] skip vanished {remote}", flush=True)
                return
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(local)
        self.synced[remote] = key
        counters["files"] += 1
        counters["bytes"] += int(information.st_size)

    def attempt_export(self):
        try:
            self.verify_pairs()
            report = load_export_workbook()(self.workbook, self.base / "baseline64/matrix", self.base / "offload64/matrix")
        except Exception as error:
            self.status["export_error"] = repr(error)[:300]
            return None
        self.status["export_error"] = None
        self.status["workbook"] = {
            "requested": str(self.workbook), "path": report["workbook"], "backup": report["backup"],
            "backup_created": report["backup_created"], "generated_at": report["generated_at"],
            "sheets": report["sheets"],
        }
        return report

    def verify_pairs(self):
        roots = [self.base / arm / "matrix" for arm in ARMS]
        if not all((root / "weight_metadata_sha256.json").is_file() for root in roots):
            return

        def read(path):
            return json.loads(path.read_text(encoding="utf-8"))

        if read(roots[0] / "weight_metadata_sha256.json") != read(roots[1] / "weight_metadata_sha256.json"):
            raise ValueError("Paired weight metadata hashes differ")
        configs = [read(root / "server_info.json") for root in roots]
        for field in ("context_length", "max_running_requests", "max_total_tokens", "tp_size",
                      "page_size", "kv_cache_dtype", "quantization", "disable_radix_cache",
                      "speculative_algorithm", "chunked_prefill_size", "mem_fraction_static"):
            if configs[0].get(field) != configs[1].get(field):
                raise ValueError(f"Paired server configuration differs: {field}")
        paired = []
        for case in roots[0].iterdir():
            if not case.is_dir() or not case.name.startswith(("single_", "saturation_")):
                continue
            other = roots[1] / case.name
            if not all((path / "summary.json").is_file() for path in (case, other)):
                continue
            if any(read(path / "summary.json").get("status") != "ok" for path in (case, other)):
                continue
            if read(case / "input_hashes.json") != read(other / "input_hashes.json"):
                raise ValueError(f"Paired input hashes differ: {case.name}")
            for request in case.glob("request_*"):
                source = read(request / "input_metadata.json")
                target = read(other / request.name / "input_metadata.json")
                if source != target:
                    raise ValueError(f"Paired request differs: {case.name}/{request.name}")
            paired.append(case.name)
        self.status["pair_validation"] = {"at": timestamp(), "matched_cases": sorted(paired)}

    def describe_arm(self, arm, snapshot):
        final = snapshot["finals"].get(arm)
        summary = snapshot["summaries"].get(arm) if isinstance(snapshot["summaries"].get(arm), dict) else {}
        progress = snapshot["progress"].get(arm) if isinstance(snapshot["progress"].get(arm), dict) else {}
        cases = []
        for payload in (summary, final):
            if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
                cases = payload["cases"]
        counts = Counter(case.get("status") for case in cases if isinstance(case, dict))
        cleanup = final.get("cleanup") if isinstance(final, dict) else None
        meta = final if isinstance(final, dict) else summary
        return {
            "status": meta.get("status"),
            "phase": progress.get("phase"),
            "input_tokens": progress.get("input_tokens"),
            "cases": len(cases),
            "expected_cases": EXPECTED_CASES,
            "ok": counts.get("ok", 0),
            "no_data": counts.get("no_data", 0),
            "final_received": final is not None,
            "cleanup_error": meta.get("cleanup_error"),
            "cleanup_remaining": cleanup.get("remaining_owned_pids") if isinstance(cleanup, dict) else None,
        }

    def describe(self, snapshot, downloads):
        self.status["state"] = "running"
        for arm in ARMS:
            self.status[arm] = self.describe_arm(arm, snapshot)
        coordinator = snapshot["coordinator"] if isinstance(snapshot["coordinator"], dict) else {}
        self.status["coordinator"] = {
            "status": coordinator.get("status"), "reason": coordinator.get("reason"),
            "runner_pid": coordinator.get("runner_pid"), "exit_code": coordinator.get("exit_code"),
            "observed": bool(coordinator),
        }
        observed = (snapshot["container"] or {}).get("started_at")
        if observed:
            self.status["container_started_at"] = observed
        self.status["downloads"] = {"files": self.status["downloads"]["files"] + downloads["files"],
                                    "bytes": self.status["downloads"]["bytes"] + downloads["bytes"]}
        self.status["last_cycle"] = {"at": timestamp(), "changed": downloads["changed"], "files": downloads["files"],
                                     "skipped": downloads["skipped"], "errors": downloads["errors"]}

    def success(self, snapshot):
        for arm in ARMS:
            final = snapshot["finals"].get(arm)
            if not isinstance(final, dict) or final.get("status") != "completed":
                return False, f"{arm}_not_completed"
            if len(final.get("cases", [])) != EXPECTED_CASES:
                return False, f"{arm}_cases_incomplete"
            if final.get("cleanup_error"):
                return False, f"{arm}_cleanup_error"
            cleanup = final.get("cleanup")
            if not isinstance(cleanup, dict) or cleanup.get("remaining_owned_pids") != []:
                return False, f"{arm}_cleanup_unconfirmed"
        if not isinstance(snapshot["coordinator"], dict) or snapshot["coordinator"].get("status") != "completed":
            return False, "coordinator_not_completed"
        return True, None

    def finish(self, state, reason, details):
        terminal = {"state": state, "reason": reason, "at": timestamp(), "details": details or {}}
        self.status["state"] = state
        self.status["success"] = state == "completed"
        self.status["terminal"] = terminal
        return terminal

    def terminal_state(self, snapshot):
        container = snapshot["container"]
        if container:
            if self.container_first is None:
                self.container_first = container
            elif container["started_at"] != self.container_first["started_at"]:
                return self.finish("interrupted", "container_restarted",
                                   {"first": self.container_first, "current": container})
            elif self.container_first["status"] == "running" and container["status"] != "running":
                return self.finish("interrupted", "container_not_running", {"current": container})
        for arm in ARMS:
            state = snapshot["runner_states"].get(arm, {})
            if state.get("status") == "gone" and state.get("observed_alive"):
                return self.finish("interrupted", f"{arm}_runner_vanished",
                                   {"record": state["record"], "probe": state["probe"],
                                    "container_first": self.container_first, "container": container})
        coordinator = snapshot["coordinator"] if isinstance(snapshot["coordinator"], dict) else {}
        if coordinator.get("status") in TERMINAL_COORDINATOR:
            complete, reason = self.success(snapshot)
            if complete:
                return self.finish("completed", "coordinator_completed", {"coordinator": coordinator})
            return self.finish("incomplete", f"terminated_by_coordinator_{coordinator.get('status')}",
                               {"coordinator": coordinator, "reason": reason})
        return None

    def cycle(self):
        snapshot = self.secure(self.poll_remote)
        self.last_snapshot = snapshot
        downloads = self.secure(lambda: self.collect(self.targets(snapshot["changed"])))
        if downloads["changed"]:
            self.status["last_sync"] = timestamp()
        if downloads["changed"] or self.status.get("export_error"):
            self.attempt_export()
        self.describe(snapshot, downloads)
        terminal = self.terminal_state(snapshot)
        self.write_status()
        arms = " ".join(f"{arm}:{self.status[arm].get('status') or '-'}"
                        f"({self.status[arm].get('ok', 0)}/{self.status[arm].get('cases', 0)}"
                        f"|{self.status[arm].get('phase') or '-'})" for arm in ARMS)
        print(f"[{timestamp()}] coord={self.status['coordinator'].get('status') or '-'} {arms} "
              f"new={downloads['files']} skipped={downloads['skipped']} failed={downloads['errors']} "
              f"workbook={self.status['workbook']['path'] or '-'}", flush=True)
        return terminal

    def terminal_tail(self, terminal):
        try:
            self.secure(lambda: self.collect(self.final_targets()))
        except Exception as error:
            terminal["details"]["artifact_collection_error"] = repr(error)[:300]
        self.attempt_export()
        if self.status.get("export_error") or terminal["details"].get("artifact_collection_error"):
            terminal["state"] = "incomplete"
            terminal["reason"] = "final_artifacts_or_export_incomplete"
            self.status.update(state="incomplete", success=False)

    def write_status(self):
        self.status["updated_at"] = timestamp()
        write_json_atomic(self.status_path, self.status)

    def run(self):
        deadline = time.monotonic() + self.timeout_hours * 3600
        self.status["pid"] = os.getpid()
        self.status["state"] = "running"
        self.write_status()
        while True:
            terminal = self.cycle()
            if terminal is None and time.monotonic() >= deadline:
                terminal = self.finish("timeout", "timeout_hours_exceeded",
                                       {"timeout_hours": self.timeout_hours,
                                        "coordinator": self.status["coordinator"]})
            if terminal is not None:
                self.terminal_tail(terminal)
                self.write_status()
                print(f"[{timestamp()}] TERMINAL state={terminal['state']} reason={terminal['reason']} "
                      f"success={self.status['success']}", flush=True)
                return 0 if terminal["state"] == "completed" else 1
            time.sleep(self.interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Follow a remote joint-budget matrix run and refresh the workbook")
    parser.add_argument("--host", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--container", default="jzx_sglang_5.3_flash")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=24)
    arguments = parser.parse_args(argv)
    if not arguments.workbook.is_file():
        raise SystemExit(f"Workbook not found: {arguments.workbook}")
    password = None  # key auth is preferred; connect_host prompts only if no key works
    follower = Follower(arguments, password)
    print(f"[{timestamp()}] following {arguments.host}:{follower.root} -> {follower.base}", flush=True)
    try:
        return follower.run()
    except KeyboardInterrupt:
        follower.status["state"] = "stopped"
        follower.status["terminal"] = {"state": "stopped", "reason": "keyboard_interrupt", "at": timestamp(),
                                       "details": {}}
        follower.write_status()
        print(f"[{timestamp()}] follower stopped by user; remote run untouched", flush=True)
        return 1
    except Exception as error:
        follower.status["state"] = "failed"
        follower.status["terminal"] = {"state": "failed", "reason": type(error).__name__, "at": timestamp(),
                                       "details": {"message": str(error)[:400]}}
        follower.write_status()
        print(f"[{timestamp()}] follower failed: {type(error).__name__}: {str(error)[:300]}", flush=True)
        return 1
    finally:
        follower.close()


if __name__ == "__main__":
    sys.exit(main())
