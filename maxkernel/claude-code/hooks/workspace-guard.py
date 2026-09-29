#!/usr/bin/env python3
"""PreToolUse guard for Claude Code.

Three jobs:
  1. Confine filesystem access to the project root (this checkout).
  2. Block every route to Pallas/Mosaic *reference implementations* --
     jax.experimental.pallas source, jax/_src/pallas, pallas/ops, and web
     lookups of the same -- so kernels get written, not copied.
  3. Seal a run's `ref/` directory once the reference has been reconciled, so
     the planner and implementer work from the distilled ideas ledger rather
     than from raw CUDA.

Deliberately NOT blocked: the word "pallas" in workspace files, in code the
agent writes, or in `from jax.experimental import pallas as pl` imports.
The block is on reading the reference *sources*, not on using the library.

Escape hatch for maintenance (user-only): start Claude Code with
    MAXKERNEL_GUARD_OFF=1 claude
The agent cannot set this itself -- hooks inherit Claude Code's process
environment, not the environment of the shell the Bash tool runs in.
"""

import json
import os
import re
import sys

HOME = "{{HOME}}"
CONFIG = "{{CLAUDE_DIR}}"

# --- what the agent may touch -------------------------------------------------

# The workspace. This is the whole allowance.
ALLOWED_ROOTS = ("{{MAXKERNEL_ROOT}}",)

# A workspace of "/" or the home directory itself would allow everything.
# Refuse loudly rather than silently protecting nothing -- a mis-rendered
# template is otherwise indistinguishable from a working install.
WORKSPACE_DEGENERATE = ALLOWED_ROOTS[0].rstrip("/") in ("", HOME.rstrip("/"))

# The config directory is sealed except for these, so that the skill, the
# subagent prompts and this guard itself remain editable. Everything else
# under it -- transcripts, tool-result dumps, sessions, credentials, memory --
# stays unreachable. Checked before DENIED_SUBPATHS, so it overrides them.
ALLOWED_EXCEPTIONS = (
    f"{CONFIG}/skills",
    f"{CONFIG}/agents",
    f"{CONFIG}/hooks",
)

# Runnable, but its library source is off limits (that is where pallas lives).
ALLOWED_EXEC_PREFIXES = (f"{HOME}/{{VENV_NAME}}/bin/",)

# Binaries, libs, device nodes -- needed to run anything at all.
# Note: /tmp and /var/tmp are deliberately absent.
SYSTEM_PREFIXES = (
    "/usr/bin/",
    "/usr/sbin/",
    "/usr/local/bin/",
    "/usr/local/sbin/",
    "/bin/",
    "/sbin/",
    "/dev/",
    "/proc/",
    "/sys/",
)

# Belt and braces: denied even if a root above were widened to cover them.
# (`memory/../<session>.jsonl` cannot slip through -- normpath collapses `..`
# before these are checked, so it resolves outside the memory exception.)
DENIED_SUBPATHS = (
    f"{CONFIG}/projects",          # conversation transcripts + tool-result dumps
    f"{CONFIG}/history.jsonl",
    f"{CONFIG}/shell-snapshots",
    f"{CONFIG}/sessions",
    f"{CONFIG}/session-env",
    f"{CONFIG}/todos",
    f"{CONFIG}/backups",
    f"{CONFIG}/.credentials.json",
)

# --- pallas reference-implementation patterns ---------------------------------

# Filesystem paths to the reference sources. Always denied, any tool, any root.
PALLAS_PATH = re.compile(
    r"""jax/experimental/pallas
      | jax/experimental/mosaic
      | jax/_src/pallas
      | jax/_src/tpu
      | /pallas/ops
      | /pallas/
      | (?:site|dist)-packages/[^\s'";|]*pallas
      | (?:site|dist)-packages/[^\s'";|]*mosaic
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Introspection tricks that read source without naming a path.
SOURCE_PEEK = re.compile(
    r"""inspect\.get(?:source|file|sourcefile|sourcelines)
      | importlib[^\s]*get_source
      | \bpkgutil\b
      | pip\s+(?:show|download)
      | \.__file__
      | linecache\.getlines
    """,
    re.IGNORECASE | re.VERBOSE,
)

MENTIONS_PALLAS = re.compile(r"\bpallas\b|\bmosaic\b", re.IGNORECASE)

# Absolute paths, anchored to this machine's real top-level dirs so that
# things like `sed s/foo/bar/` are not mistaken for filesystem paths.
ABS_PATH = re.compile(
    r"(?<![A-Za-z0-9_.\-])(?:~|\$HOME|\$\{HOME\})?"
    r"/(?:home|root|usr|bin|sbin|lib|lib32|lib64|libx32|etc|opt|var|tmp|proc"
    r"|sys|dev|run|snap|srv|mnt|media|boot)"
    # The directory name must END here. Without this boundary the alternation
    # matches a prefix of a longer name: `opt` matches the first three letters
    # of `optimized`, the optional tail below matches empty, and the guard
    # extracts the token `/opt` -- denying <run_dir>/iter<n>/optimized.py, the
    # file this loop touches more than any other. Same collision for /tmpfile,
    # /vary, /runtime, /homedir, /libexec, /etcetera.
    r"(?![A-Za-z0-9_.\-])"
    r"(?:/[^\s'\";|&$()<>]*)?"
)

# Home-anchored paths in any form: ~/.ssh/id_rsa, $HOME/{{VENV_NAME}}/lib, ...
HOME_PATH = re.compile(
    r"(?<![A-Za-z0-9_.\-])(?:~|\$HOME|\$\{HOME\})/[^\s'\";|&$()<>]*"
)

# Relative paths that climb out of the current directory.
REL_ESCAPE = re.compile(r"(?<![A-Za-z0-9_.\-])\.\./[^\s'\";|&$()<>]*")

# `cd` / `pushd` forms that leave the workspace without naming a path:
# a bare `cd` (goes home), `cd ..`, `cd ~`, `cd -`, `cd $HOME`.
CD_ESCAPE = re.compile(
    r"(?:^|[;&|]\s*)(?:cd|pushd)\s*(?:$|[;&|])"
    r"|(?:^|[;&|]\s*)(?:cd|pushd)\s+(?:\.\.|~|-|\$HOME|\$\{HOME\})(?:\s|;|&|\||$)"
)

# Trailing punctuation that belongs to surrounding prose, not to the path
# (e.g. "is {{CLAUDE_DIR}} needed?" must not be read as a path named `.claude?`).
TRAILING_PUNCT = ".,;:!?'\")]}>`*"


def clean(token: str) -> str:
    return token.rstrip(TRAILING_PUNCT)


def normalize(path: str) -> str:
    path = clean(path)
    for prefix in ("${HOME}", "$HOME", "~"):
        if path.startswith(prefix):
            path = HOME + path[len(prefix):]
            break
    return os.path.normpath(path)


def under(path: str, roots) -> bool:
    for root in roots:
        root = root.rstrip("/")
        if path == root or path.startswith(root + "/"):
            return True
    return False


def path_allowed(path: str) -> bool:
    p = normalize(path)
    if under(p, ALLOWED_EXCEPTIONS):   # checked first: overrides the denies below
        return True
    if under(p, DENIED_SUBPATHS):
        return False
    if under(p, ALLOWED_EXEC_PREFIXES):
        return True
    if under(p, ALLOWED_ROOTS):
        return True
    if under(p, SYSTEM_PREFIXES):
        return True
    return False


# --- the reference seal ------------------------------------------------------

# A run that was given a reference kernel keeps everything derived from it
# under `<run_dir>/ref/`. The reference analyzer and the reconciler need to
# read that directory; nobody after them does. Once reconciliation finishes,
# the worker drops a `.sealed` marker in it and this guard refuses further
# reads.
#
# The seal is a path-level rule on purpose. A PreToolUse payload carries
# `tool_name`, `tool_input` and `cwd` -- it does not say which subagent is
# calling -- so a rule phrased as "the implementer may not read ref/" could
# not actually be enforced here. Phrased as "nobody may read a sealed ref/",
# it can be, and it expresses the same intent: the raw reference is available
# exactly while it is being distilled, and never afterwards.
REF_SEAL_NAME = ".sealed"

# A bare relative mention of the reference directory: `ref/x.cu`, `./ref/x.cu`,
# `iter3/../ref/x.cu`. Anchored on a word boundary so `myref/` does not match.
REF_RELATIVE = re.compile(r"(?:(?<=^)|(?<=[\s=\"'(:]))\.{0,2}/?ref/[\w./-]*")


def ref_seal_violation(path: str) -> bool:
    """True if `path` reaches into a `ref/` directory that has been sealed."""
    norm = normalize(path)
    parts = norm.split(os.sep)
    for i, part in enumerate(parts):
        if part != "ref":
            continue
        ref_dir = os.sep.join(parts[: i + 1])
        # Only ever applies inside a run directory under the workspace.
        if not under(ref_dir, ALLOWED_ROOTS):
            continue
        if os.path.exists(os.path.join(ref_dir, REF_SEAL_NAME)):
            # Reading the seal itself is how the worker checks idempotency.
            if os.path.basename(norm) == REF_SEAL_NAME:
                return False
            return True
    return False


def deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


OUTSIDE = (
    "Blocked: {path} is outside the project root. The only readable root "
    f"is {ALLOWED_ROOTS[0]}. ({HOME}/{{VENV_NAME}}/bin/ is runnable but its "
    "library source is not readable; conversation transcripts, tool-result "
    "dumps and scratch dirs are not readable at all.)"
)

OUTSIDE_CWD = (
    "Blocked: this session's working directory is {cwd}, which is outside the "
    "workspace. Shell commands resolve bare relative paths against it, so they "
    "could reach outside the workspace without ever naming a path this guard "
    "can see. Restart Claude Code from {root} and this lifts."
)

REF_SEALED = (
    "Blocked: this run's ref/ directory is sealed.\n\n"
    "It holds reference kernels the user supplied for the PLANNER to mine. "
    "They have already been reconciled against the primary source and "
    "distilled into <run_dir>/ideas_ledger.json, where each idea carries a "
    "portability class, a TPU translation and a trust level. The plan cites "
    "the ones it adopted; those citations are the supported way to use the "
    "reference.\n\n"
    "Reading the raw source at this point is how a TPU kernel ends up "
    "transliterating SIMT machinery -- warp shuffles, __syncthreads, "
    "per-thread register blocking -- that has no counterpart here and becomes "
    "scalar busywork in Pallas.\n\n"
    "If the plan does not say enough to implement from, report that instead."
)

REFERENCE = (
    "Blocked: that reaches a Pallas/Mosaic reference implementation. "
    "Kernels in this workspace must be written from the docs and first "
    "principles, not copied from jax.experimental.pallas sources. "
    "Using `pallas` in code you write is fine -- reading its source is not."
)


def main() -> None:
    if os.environ.get("MAXKERNEL_GUARD_OFF") == "1":
        sys.exit(0)

    if WORKSPACE_DEGENERATE:
        deny(
            "Blocked: the workspace resolves to %s, which would allow "
            "everything. Re-run install.py so the hook points at a real "
            "workspace directory." % ALLOWED_ROOTS[0]
        )

    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # unparseable input: stay out of the way

    tool = payload.get("tool_name", "")
    args = payload.get("tool_input") or {}
    cwd = payload.get("cwd") or HOME

    # Bare relative paths in a shell command are resolved by the shell against
    # its working directory, which this hook never sees change. Requiring that
    # directory to sit inside the workspace makes every such path safe by
    # construction -- only `..` can then climb out, and REL_ESCAPE / CD_ESCAPE
    # catch that.
    if tool == "Bash" and not under(os.path.normpath(cwd), ALLOWED_ROOTS):
        deny(OUTSIDE_CWD.format(cwd=cwd, root=ALLOWED_ROOTS[0]))

    # 1. Explicit path fields -- exact check, no guessing.
    for field in ("file_path", "path", "notebook_path"):
        value = args.get(field)
        if isinstance(value, str) and value:
            candidate = (
                value if os.path.isabs(normalize(value)) else os.path.join(cwd, value)
            )
            if PALLAS_PATH.search(normalize(candidate)):
                deny(REFERENCE)
            if ref_seal_violation(candidate):
                deny(REF_SEALED)
            if not path_allowed(candidate):
                deny(OUTSIDE.format(path=value))

    # 2. Web lookups of the reference implementations.
    if tool in ("WebSearch", "WebFetch"):
        blob = " ".join(
            str(args.get(k, "")) for k in ("query", "url", "prompt", "allowed_domains")
        )
        if MENTIONS_PALLAS.search(blob):
            deny(REFERENCE)
        return

    # 3. Free-text fields: shell commands, glob/grep patterns, subagent prompts.
    blob = " ".join(
        str(args.get(k, ""))
        for k in ("command", "pattern", "glob", "prompt", "description")
        if args.get(k)
    )
    if not blob:
        return

    if PALLAS_PATH.search(blob):
        deny(REFERENCE)

    if CD_ESCAPE.search(blob):
        deny(OUTSIDE_CWD.format(cwd="(changed by cd)", root=ALLOWED_ROOTS[0]))

    if MENTIONS_PALLAS.search(blob) and SOURCE_PEEK.search(blob):
        deny(REFERENCE)

    # {{VENV_NAME}} is runnable but not readable.
    if re.search(r"{{VENV_NAME}}/(?!bin/)", blob):
        deny(OUTSIDE.format(path="{{VENV_NAME}} library source"))

    for pattern in (HOME_PATH, ABS_PATH):
        for match in pattern.finditer(blob):
            token = clean(match.group(0))
            if not token or token in ("~", "$HOME"):
                continue
            if ref_seal_violation(token):
                deny(REF_SEALED)
            if not path_allowed(token):
                deny(OUTSIDE.format(path=token))

    for match in REL_ESCAPE.finditer(blob):
        resolved = os.path.normpath(os.path.join(cwd, clean(match.group(0))))
        if PALLAS_PATH.search(resolved):
            deny(REFERENCE)
        if ref_seal_violation(resolved):
            deny(REF_SEALED)
        if not path_allowed(resolved):
            deny(OUTSIDE.format(path=match.group(0)))

    # A relative `ref/...` that never escapes the run directory matches none of
    # the patterns above, because none of them fire on a plain relative path.
    # `cat ref/flash.cu` from inside a run directory is exactly the read the
    # seal exists to stop, so resolve bare `ref/` mentions against cwd too.
    for match in REF_RELATIVE.finditer(blob):
        resolved = os.path.normpath(os.path.join(cwd, clean(match.group(0))))
        if ref_seal_violation(resolved):
            deny(REF_SEALED)


if __name__ == "__main__":
    main()
