# CVXPY Codex Skills

This repository packages `convexify-with-cvxpy` as a Codex plugin.

## Layout

```text
.agents/plugins/marketplace.json
plugins/convexify-with-cvxpy/.codex-plugin/plugin.json
plugins/convexify-with-cvxpy/skills/convexify-with-cvxpy/SKILL.md
```

The plugin manifest follows the Codex plugin shape: `.codex-plugin/plugin.json`
at the plugin root, skill folders under `skills/`, and marketplace discovery
through `.agents/plugins/marketplace.json`.

## Local Skill Install

For a direct skill-only install, copy:

```text
plugins/convexify-with-cvxpy/skills/convexify-with-cvxpy
```

to:

```text
~/.codex/skills/convexify-with-cvxpy
```

## Validation

Validate the skill:

```bash
uv run --no-project --with PyYAML python \
  ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  plugins/convexify-with-cvxpy/skills/convexify-with-cvxpy
```

Run the bundled audit script smoke test:

```bash
uv run --no-project --with cvxpy python \
  plugins/convexify-with-cvxpy/skills/convexify-with-cvxpy/scripts/cvxpy_convex_audit.py \
  --self-test
```
