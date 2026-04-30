# Codex Skills

This repository is a reusable Codex plugin catalog for skills.
`convexify-with-cvxpy` is the first plugin in the catalog; future skills can be
added as additional plugin folders under `plugins/`.

## Layout

```text
.agents/plugins/marketplace.json
plugins/convexify-with-cvxpy/.codex-plugin/plugin.json
plugins/convexify-with-cvxpy/skills/convexify-with-cvxpy/SKILL.md
```

Each plugin follows the Codex plugin shape:

```text
plugins/<plugin-name>/.codex-plugin/plugin.json
plugins/<plugin-name>/skills/<skill-name>/SKILL.md
```

Marketplace discovery is handled through `.agents/plugins/marketplace.json`.

## Adding Another Skill

Create a new plugin folder under `plugins/`, add its skill folder under that
plugin's `skills/` directory, then append the plugin to
`.agents/plugins/marketplace.json`.

For the standard scaffold:

```bash
python3 ~/.codex/skills/.system/plugin-creator/scripts/create_basic_plugin.py \
  <plugin-name> --path plugins --with-skills --with-assets \
  --with-marketplace --category Coding
```

Then fill:

```text
plugins/<plugin-name>/.codex-plugin/plugin.json
```

The marketplace entry should use `source.path` set to `./plugins/<plugin-name>`.

## Local Skill Install

For a direct skill-only install of the current CVXPY skill, copy:

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
