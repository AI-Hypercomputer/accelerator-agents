#!/usr/bin/env python3
"""Test suite for workspace-guard.py.

Usage:  python3 {{CLAUDE_DIR}}/hooks/guard-tests.py [path-to-guard.py]

Run it after editing the guard. Because the fixtures below contain the very
paths the guard blocks, run it with the guard disabled for the session:
    MAXKERNEL_GUARD_OFF=1 python3 {{CLAUDE_DIR}}/hooks/guard-tests.py
"""

import json
import os
import subprocess
import sys

GUARD = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "{{CLAUDE_DIR}}/hooks/workspace-guard.py"
)

HOME = "{{HOME}}"
WS = "{{MAXKERNEL_ROOT}}"
CFG = "{{CLAUDE_DIR}}"
PROJ = f"{CFG}/projects/" + HOME.replace("/", "-")
VENV = f"{HOME}/{{VENV_NAME}}"
PALLAS = f"{VENV}/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu"

ETC_PASSWD = '/etc/passwd'
USR_LIB = '/usr/lib/python3/x.py'
USR_BIN = '/usr/bin/python3'
USR_BIN_DIR = '/usr/bin'
NULL_DEV = '/dev/null'

BLOCK, ALLOW = "BLOCK", "ALLOW"


def case(expect, label, tool, tool_input, cwd=HOME):
    return (expect, label, {"cwd": cwd, "tool_name": tool, "tool_input": tool_input})


CASES = [
    # --- conversation logs / tool-result dumps ---
    case(BLOCK, "read transcript jsonl", "Read", {"file_path": f"{PROJ}/6a0730e5.jsonl"}),
    case(BLOCK, "cat tool-results dump", "Bash",
         {"command": f"cat {PROJ}/1cc53fbc-ebac/tool-results/b0vrzblc9.txt"}),
    case(BLOCK, "grep all transcripts", "Bash", {"command": "grep -rl BlockSpec {{CLAUDE_DIR}}/projects/"}),
    case(BLOCK, "subagent jsonl", "Read", {"file_path": f"{PROJ}/1cc53fbc-ebac/subagents/agent-a1b.jsonl"}),
    case(BLOCK, "memory/.. escape", "Read", {"file_path": f"{PROJ}/memory/../6a0730e5.jsonl"}),
    case(BLOCK, "history.jsonl", "Read", {"file_path": f"{CFG}/history.jsonl"}),
    case(BLOCK, "shell snapshots", "Bash", {"command": "cat {{CLAUDE_DIR}}/shell-snapshots/snap.sh"}),
    case(BLOCK, "credentials", "Read", {"file_path": f"{CFG}/.credentials.json"}),
    case(BLOCK, "backups", "Read", {"file_path": f"{CFG}/backups/settings.json.pre-guard"}),

    # --- scratch space ---
    case(BLOCK, "read /tmp", "Read", {"file_path": "/tmp/claude-693950567/x/notes.md"}),
    case(BLOCK, "write /tmp", "Write", {"file_path": "/tmp/foo.py", "content": "x"}),
    case(BLOCK, "cat /var/tmp", "Bash", {"command": "cat /var/tmp/x"}),

    # --- rest of the config dir ---
    case(ALLOW, "read agent def", "Read", {"file_path": f"{CFG}/agents/maxkernel-worker.md"}),
    case(ALLOW, "read SKILL.md", "Read", {"file_path": f"{CFG}/skills/maxkernel/SKILL.md"}),
    case(ALLOW, "read own guard", "Read", {"file_path": f"{CFG}/hooks/workspace-guard.py"}),
    case(ALLOW, "edit own guard", "Edit",
         {"file_path": f"{CFG}/hooks/workspace-guard.py", "old_string": "a", "new_string": "b"}),
    # the carve-outs must not open the rest of the config directory
    case(BLOCK, "skills/.. escape", "Read", {"file_path": f"{CFG}/skills/../settings.json"}),
    case(BLOCK, "hooks/.. into projects", "Read",
         {"file_path": f"{CFG}/hooks/../projects/x.jsonl"}),
    case(BLOCK, "edit settings.json", "Edit",
         {"file_path": f"{CFG}/settings.json", "old_string": "a", "new_string": "b"}),

    # --- memory carve-out ---
    case(BLOCK, "write memory file", "Write", {"file_path": f"{PROJ}/memory/foo.md", "content": "x"}),
    case(BLOCK, "read MEMORY.md", "Read", {"file_path": f"{PROJ}/memory/MEMORY.md"}),

    # --- prose false-positive fix ---
    # `{{CLAUDE_DIR}}` is outside the workspace now, so this blocks either way --
    # kept to pin that the config dir is unreachable even in prose.
    case(BLOCK, "tilde in a question", "Bash",
         {"command": 'echo "does it need {{CLAUDE_DIR}}?"'}, cwd=WS),
    # This one genuinely exercises trailing-punctuation stripping: without it
    # the path reads as `.../memory.` which falls outside the carve-out.
    case(BLOCK, "memory now sealed too", "Bash",
         {"command": f'echo "writing to {PROJ}/memory."'}, cwd=WS),
    # Trailing-punctuation stripping, on a path that IS allowed:
    case(ALLOW, "allowed path, trailing dot", "Bash",
         {"command": f'echo "see {WS}/tools/retrieval.py."'}, cwd=WS),
    case(ALLOW, "allowed path, backticks", "Bash",
         {"command": f'echo "see `{WS}/tools`"'}, cwd=WS),
    case(ALLOW, "allowed path, glob star", "Bash",
         {"command": f'echo "matching {WS}/workspace/**"'}, cwd=WS),
    case(ALLOW, "allowed path, trailing paren", "Bash",
         {"command": f'echo "see {WS}/tools/retrieval.py)"'}, cwd=WS),

    # --- pallas reference implementations ---
    case(BLOCK, "cat pallas ops", "Bash", {"command": f"cat {PALLAS}/flash_attention.py"}),
    case(BLOCK, "grep venv lib", "Bash", {"command": "grep -rn BlockSpec ~/{{VENV_NAME}}/lib/"}),
    case(BLOCK, "find pallas", "Bash", {"command": 'find / -path "*/pallas/ops/*" -name "*.py"'}),
    case(BLOCK, "inspect.getsource", "Bash",
         {"command": 'python3 -c "import inspect,jax.experimental.pallas as pl; print(inspect.getsource(pl))"'}),
    case(BLOCK, "pallas __file__", "Bash",
         {"command": 'python -c "from jax.experimental import pallas; print(pallas.__file__)"'}),
    case(BLOCK, "rel escape venv", "Bash",
         {"command": f"cat ../{{VENV_NAME}}/lib/python3.11/site-packages/jax/experimental/pallas/ops/tpu/paged.py"},
         cwd=WS),
    case(BLOCK, "Glob into venv", "Glob", {"path": VENV, "pattern": "**/pallas/**"}),
    case(BLOCK, "WebSearch pallas", "WebSearch", {"query": "jax pallas tpu flash attention kernel"}),
    case(BLOCK, "WebFetch jax github", "WebFetch",
         {"url": "https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/fa.py",
          "prompt": "show kernel"}),
    case(BLOCK, "subagent asked to read", "Agent",
         {"prompt": f"read {PALLAS}/flash_attention.py and summarize the tiling"}),
    case(BLOCK, "mosaic source", "Bash",
         {"command": "less /usr/lib/python3/dist-packages/jax/experimental/mosaic/core.py"}),

    # --- other outside-workspace ---
    case(BLOCK, "ssh key", "Bash", {"command": "cat ~/.ssh/id_rsa"}),
    case(BLOCK, "bash_history", "Read", {"file_path": f"{HOME}/.bash_history"}),
    case(BLOCK, "other project dir", "Bash", {"command": f"ls -la {HOME}/maxkernel_deploy"}),

    # --- normal MaxKernel work must still pass ---
    case(ALLOW, "workspace md", "Read", {"file_path": f"{WS}/reference/pallas_xla_interaction.md"}),
    case(ALLOW, "workspace kernel", "Read", {"file_path": f"{WS}/workspace/run_1/kernel.py"}),
    case(ALLOW, "write pallas import", "Write",
         {"file_path": f"{WS}/workspace/k.py",
          "content": "from jax.experimental import pallas as pl\nimport jax.experimental.pallas.tpu as pltpu\n"}),
    case(ALLOW, "edit kernel", "Edit",
         {"file_path": f"{WS}/workspace/run_1/kernel.py",
          "old_string": "pl.BlockSpec(a)", "new_string": "pl.BlockSpec(b)"}),
    case(ALLOW, "venv python run", "Bash",
         {"command": "~/{{VENV_NAME}}/bin/python tools/assemble_test_run.py --run workspace/run_1"}, cwd=WS),
    case(ALLOW, "source activate", "Bash",
         {"command": f"source {VENV}/bin/activate && python kernel.py"}, cwd=WS),
    case(ALLOW, "grep workspace", "Bash", {"command": f"grep -rn pallas {WS}/subagents/"}, cwd=WS),
    case(ALLOW, "sed expr not a path", "Bash",
         {"command": "sed -i s/block_q/BLOCK_Q/g workspace/run_1/kernel.py"}, cwd=WS),
    case(ALLOW, "pip install", "Bash",
         {"command": "~/{{VENV_NAME}}/bin/python -m pip install -r requirements.txt"}, cwd=WS),
    case(ALLOW, "curl localhost", "Bash",
         {"command": "curl -s http://localhost:8000/run -d @payload.json"}, cwd=WS),
    case(ALLOW, "websearch tpu (no pallas)", "WebSearch",
         {"query": "TPU MXU systolic array dimensions v5e"}),
    case(ALLOW, "python -c uses pallas", "Bash",
         {"command": '~/{{VENV_NAME}}/bin/python -c "from jax.experimental import pallas as pl; print(pl.BlockSpec)"'},
         cwd=WS),
    case(ALLOW, "git status", "Bash", {"command": f"git -C {WS} status --short"}, cwd=WS),
    case(ALLOW, "mktemp, no literal path", "Bash",
         {"command": 'python3 -c "import tempfile; print(tempfile.mkdtemp())"'}, cwd=WS),
    case(ALLOW, "no path fields at all", "TodoWrite", {"todos": []}),

    # --- working-directory gate: shell cannot run from outside the workspace ---
    case(BLOCK, "bash from home cwd", "Bash", {"command": "ls"}, cwd=HOME),
    case(BLOCK, "bash from home, rel read", "Bash", {"command": "cat .bashrc"}, cwd=HOME),
    case(BLOCK, "bash from home, rel search", "Bash", {"command": "grep -rn secret ."}, cwd=HOME),
    case(BLOCK, "bash from venv cwd", "Bash", {"command": "ls"}, cwd=VENV),
    case(ALLOW, "bash from workspace cwd", "Bash", {"command": "ls"}, cwd=WS),
    case(ALLOW, "bash from workspace subdir", "Bash", {"command": "ls"}, cwd=f"{WS}/workspace/run_1"),
    # non-shell tools resolve their own path fields, so no cwd gate is needed
    case(ALLOW, "read tool from home cwd", "Read", {"file_path": f"{WS}/README.md"}, cwd=HOME),

    # --- cd escapes that never name a path ---
    case(BLOCK, "bare cd", "Bash", {"command": "cd"}, cwd=WS),
    case(BLOCK, "cd dotdot", "Bash", {"command": "cd .. && cat .bashrc"}, cwd=WS),
    case(BLOCK, "cd tilde", "Bash", {"command": "cd ~"}, cwd=WS),
    case(BLOCK, "cd dash", "Bash", {"command": "cd - ; ls"}, cwd=WS),
    case(BLOCK, "cd HOME var", "Bash", {"command": "cd $HOME"}, cwd=WS),
    case(BLOCK, "pushd dotdot", "Bash", {"command": "pushd .."}, cwd=WS),
    case(ALLOW, "cd into workspace subdir", "Bash", {"command": "cd workspace/run_1 && ls"}, cwd=WS),

    # --- system paths: executables run, but the tree is not browsable ---
    case(BLOCK, "view system config file", "Bash", {"command": f"cat {ETC_PASSWD}"}, cwd=WS),
    case(BLOCK, "read system lib source", "Bash", {"command": f"head -50 {USR_LIB}"}, cwd=WS),
    case(ALLOW, "run system binary", "Bash", {"command": f"{USR_BIN} kernel.py"}, cwd=WS),
    case(ALLOW, "redirect to null device", "Bash", {"command": "python3 kernel.py 2>" + NULL_DEV}, cwd=WS),
    # normpath drops the trailing slash; under() must still match these
    case(ALLOW, "PATH export, no trailing slash", "Bash",
         {"command": f"export PATH={USR_BIN_DIR}:$PATH && python3 kernel.py"}, cwd=WS),

    # --- directory-name prefix collisions --------------------------------
    # ABS_PATH recognises a token as a path by matching a real top-level
    # directory name. That name must end at a boundary: without one, `opt`
    # matches the first three letters of `optimized` and the guard extracts
    # `/opt`, denying the loop's most-touched artifact. Both directions are
    # pinned here -- the collisions must pass, the real directories must not.
    case(ALLOW, "iter kernel, template form", "Bash",
         {"command": "test -s <run_dir>/iter<n>/optimized.py"}, cwd=WS),
    case(ALLOW, "iter kernel, brace-expanded", "Bash",
         {"command": "cat ${run_dir}/optimized.py"}, cwd=WS),
    case(ALLOW, "iter kernel, concrete path", "Bash",
         {"command": f"test -s {WS}/workspace/run_1/iter3/optimized.py"}, cwd=WS),
    case(ALLOW, "iter kernel in subagent prompt", "Agent",
         {"prompt": "Implement the kernel at <run_dir>/iter<n>/optimized.py"}, cwd=WS),
    case(ALLOW, "iter kernel as write target", "Write",
         {"file_path": f"{WS}/workspace/run_1/iter3/optimized.py", "content": "x"}),
    case(ALLOW, "apply_best_config rewrites iter kernel", "Bash",
         {"command": f"python3 tools/apply_best_config.py spec.json results.json "
                     f"{WS}/workspace/run_1/iter3/optimized.py"}, cwd=WS),
    # ABS_PATH is a recogniser, not an exhaustive denier: it already ignores
    # roots it does not know (/data, /srvdata). A name that merely starts like
    # a known root joins that set rather than being mis-read as the root.
    case(ALLOW, "tmp-prefixed root file is not /tmp", "Bash",
         {"command": "cat /tmpfile-notes.txt"}, cwd=WS),

    # ... and the real directories those names collide with still deny
    case(BLOCK, "third-party install tree", "Bash",
         {"command": "cat /opt/google/creds.json"}, cwd=WS),
    case(BLOCK, "third-party install tree, bare", "Bash",
         {"command": "ls /opt"}, cwd=WS),
    case(BLOCK, "system log tree", "Bash",
         {"command": "cat /var/log/syslog"}, cwd=WS),
    case(BLOCK, "runtime state tree", "Bash",
         {"command": "cat /run/secrets/token"}, cwd=WS),
    case(BLOCK, "another user's home", "Bash",
         {"command": "ls /home/someone-else/.ssh"}, cwd=WS),
]


def main():
    env = dict(os.environ)
    env.pop("MAXKERNEL_GUARD_OFF", None)  # never let the escape hatch skew results

    failures = 0
    for expect, label, payload in CASES:
        out = subprocess.run(
            [sys.executable, GUARD], input=json.dumps(payload),
            capture_output=True, text=True, env=env,
        ).stdout
        got = BLOCK if '"deny"' in out else ALLOW
        mark = "ok  " if got == expect else "FAIL"
        if got != expect:
            failures += 1
        print(f"  {mark} {got:<5} {label}")

    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
