# Coworker Agent Framework

The **Coworker Agent Framework** provides a domain-neutral, harness-agnostic
foundation for defining, compiling, installing, and executing collaborative
multi-agent packages across target agent harnesses (such as **Claude Code**
and **Codex**).

The framework acts as a pure compiler and runtime orchestrator: it enforces
deterministic schema contracts, validates hierarchical delegation graphs, and
guarantees isolated execution namespaces and artifact provenance without
embedding domain-specific prompts or agent logic.

---

## Package Lifecycle Workflow

The canonical workflow comprises four core stages: **Verification**,
**Translation**, **Installation**, and **Uninstallation**.

```
┌─────────────────┐       ┌────────────────────┐       ┌─────────────────────┐
│ Canonical       │ verify│ Harness-Specific   │install│ Active Project /    │
│ Package Source  ├──────►│ Distribution       ├──────►│ Workspace           │
│ (package.json)  │trans- │ (dist/<harness>/)  │       │ (.coworker/<pkg>/)  │
└─────────────────┘  late └────────────────────┘       └─────────────────────┘
```

### 1. Package Verification
Validates manifest schema, instructions integrity, tool permissions,
compatibility targets, and delegation topologies (detecting circular
delegations, depth exceeding two levels, and unreachable subagents):

```bash
python3 framework/coworker.py verify path/to/package
```

### 2. Harness Translation
Compiles a canonical package into a standalone distribution tailored to a
specific agent harness (`claude-code` or `codex`). Generates native manifests,
agent system prompts, skill contracts, and bundles the portable runtime
snapshot:

```bash
# Compile for Claude Code
python3 framework/coworker.py translate path/to/package \
    --harness claude-code \
    --output dist/claude-code/<package-name>

# Compile for Codex
python3 framework/coworker.py translate path/to/package \
    --harness codex \
    --output dist/codex/<package-name>
```

### 3. Installation
Installs a compiled distribution into a destination workspace. Pre-validates
file collisions, collects and validates environment answers, executes
non-destructive JSON/TOML configuration merges, and records an atomic ownership
ledger (`ownership.json`):

```bash
# Interactive installation
python3 framework/coworker.py install dist/codex/<package-name> \
    --destination /path/to/project

# Non-interactive / Automated installation
python3 framework/coworker.py install dist/claude-code/<package-name> \
    --destination /path/to/project \
    --answers path/to/answers.json \
    --non-interactive

# Upgrade an existing installation
python3 framework/coworker.py install dist/claude-code/<package-name> \
    --destination /path/to/project \
    --upgrade
```

### 4. Uninstallation
Cleanly removes all managed files and rolls back modified JSON/TOML
configuration entries while preserving user-created artifacts and unmodified
settings:

```bash
python3 framework/coworker.py uninstall /path/to/project \
    --package <package-name>
```

---

## Embedded Runtime Architecture

Every generated distribution bundles a self-contained runtime snapshot under
`runtime/` (`coworker.py` and `utils.py`). Installed agent entrypoints invoke
this local runtime to manage execution sessions and artifact lifecycles without
relying on external framework installations.

### Run Namespace Isolation
Before initiating subagent delegation, the entrypoint workflow allocates a
collision-resistant, timestamped run directory under
`.coworker/<package-name>/runs/<run-id>/`:

```bash
python3 .coworker/<package-name>/runtime/coworker.py start-run \
    --workspace . \
    --package <package-name>
```

Output:
```json
{
   "created_at" : "2026-08-14T05:00:00Z",
   "environment_revision" : 1,
   "framework_version" : "0.0.1",
   "package" : "cli-ui",
   "run_id" : "20260814T050000Z-7a3b9f12c4d5"
}
```

### Artifact Provenance & Descriptors
Durable outputs are materialized exclusively within
`.coworker/<package-name>/runs/<run-id>/artifacts/`. Agents exchange structured
descriptors containing canonical URIs, content hashes, and schema metadata
rather than embedding raw payloads across message boundaries:

```bash
python3 .coworker/<package-name>/runtime/coworker.py describe-artifact \
    --workspace . \
    --package <package-name> \
    --run-id <run-id> \
    --file ui-spec.json \
    --schema schemas/ui-specification.json \
    --media-type application/json
```

Output:
```json
{
   "environment_revision" : 1,
   "media_type" : "application/json",
   "run_id" : "20260814T050000Z-7a3b9f12c4d5",
   "schema" : "schemas/ui-specification.json",
   "sha256" : "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
   "uri" : "workspace://cli-ui/runs/20260814T050000Z-7a3b9f12c4d5/artifacts/ui-spec.json"
}
```

### Contract & Payload Validation
Validates input and output instances against package schemas using the
deterministic JSON Schema validator:

```bash
python3 .coworker/<package-name>/runtime/coworker.py validate \
    --schema schemas/analysis-request.json \
    instance.json
```

---

## Core Guarantees & Constraints

1. **Domain-Neutral Core**: All domain intelligence, prompts, schemas,
   environment questions, and tool definitions reside strictly within the
   package manifest (`package.json`).
2. **Safe Path Traversal Protection**: All paths are strictly bounded within
   root boundaries, rejecting traversal elements (`..`) and symlink escapes.
3. **Deterministic Verification**: Strict validation of delegation trees (no
   cycles, max depth <= 2, complete reachability) and semantic version ranges.
4. **Hermetic Runtime Snapshot**: Installed projects execute independently from
   source trees via embedded `runtime/` bundles.
