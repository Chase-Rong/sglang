"""Shared SSH connection for the A5 NPU remote helpers.

Prefers local key authentication (non-interactive) and falls back to an
interactive password prompt only when no key is accepted.

Rationale: `remote_joint_budget.py`, `follow_fsa_matrix.py` and
`remote_fsa_native_capacity.py` used to hard-code
`look_for_keys=False, allow_agent=False` plus an unconditional `getpass()`
prompt, which forced a manual password on every invocation even where an
authorized key is already installed. `--password` still wins when given.
"""

import getpass
import os

import paramiko

DEFAULT_KEY_CANDIDATES = (
    os.path.expanduser(r"~\.ssh\id_ed25519"),
    os.path.expanduser("~/.ssh/id_ed25519"),
    os.path.expanduser("~/.ssh/id_rsa"),
)


def _load_private_key(path):
    """Load whatever key type the file happens to be."""
    for loader in (
        paramiko.Ed25519Key,
        paramiko.ECDSAKey,
        paramiko.RSAKey,
    ):
        try:
            return loader.from_private_key_file(path)
        except Exception:
            continue
    return None


def connect_host(host, username="root", password=None, timeout=15, key_file=None):
    """Return a connected SSHClient for `host`.

    Order: explicit `password` argument, then every usable local key, then an
    interactive password prompt as the last resort.
    """
    def _new():
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return client

    if password:
        client = _new()
        client.connect(host, username=username, password=password,
                       look_for_keys=False, allow_agent=False, timeout=timeout)
        return client

    candidates = (key_file,) if key_file else DEFAULT_KEY_CANDIDATES
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        key = _load_private_key(path)
        if key is None:
            continue
        try:
            client = _new()
            client.connect(host, username=username, pkey=key, timeout=timeout,
                           allow_agent=False, look_for_keys=False)
            return client
        except Exception:
            continue

    client = _new()
    client.connect(host, username=username,
                   password=getpass.getpass(f"SSH password for {username}@{host}: "),
                   look_for_keys=False, allow_agent=False, timeout=timeout)
    return client
