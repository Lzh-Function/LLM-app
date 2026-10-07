# /// script
# requires-python = ">=3.12"
# dependencies = ["tomlkit>=0.13,<1"]
# ///
"""Install an optional Qwen profile and migrate the former global Qwen defaults."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import tomlkit

SOURCE = Path(__file__).resolve().parent
MODEL = "qwen3.8-flash-next-sc117-abliterated-iq3_xxs"
LOCAL_KEYS = (
    "model",
    "model_provider",
    "model_catalog_json",
    "model_context_window",
    "model_auto_compact_token_limit",
    "model_reasoning_effort",
    "show_raw_agent_reasoning",
    "web_search",
)


def previous_gpt_settings(root):
    """Recover the settings before our global Qwen installation, if available."""
    for path in sorted((root / "config-backups").glob("flash-next-*/config.toml")):
        settings = tomlkit.parse(path.read_text())
        if settings.get("model_provider", "openai") == "openai" and settings.get(
            "model", ""
        ).startswith("gpt-"):
            return settings
    return tomlkit.document()


def write_atomic(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content)
    temporary.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o600)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-home", type=Path, help="Install into this configuration directory"
    )
    args = parser.parse_args()
    root = (
        (
            args.config_home
            or Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        )
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.toml"
    old = tomlkit.parse(path.read_text()) if path.is_file() else tomlkit.document()
    profile_path = root / "qwen.config.toml"
    catalog_path = root / "flash-next-models.json"
    template = (
        (SOURCE / "qwen.config.toml")
        .read_text()
        .replace(
            '"@CODEX_HOME@/flash-next-models.json"',
            json.dumps(str(root / "flash-next-models.json")),
        )
    )
    migrate = old.get("model") == MODEL or old.get("model_provider") == "strata"
    previous = previous_gpt_settings(root) if migrate else None
    backup = (
        root
        / "config-backups"
        / datetime.now(timezone.utc).strftime("flash-next-%Y%m%d-%H%M%S-%f")
    )
    backup.mkdir(parents=True)
    legacy = root / "qwen-local.config.toml"
    for original in (path, profile_path, catalog_path, legacy):
        if original.is_file():
            shutil.copy2(original, backup / original.name)
    if migrate:
        # Restore GPT defaults without reverting unrelated edits since installation.
        for key in LOCAL_KEYS:
            old.pop(key, None)
            if key in previous:
                old[key] = previous[key]
        for key in ("apps", "multi_agent"):
            if key in previous.get("features", {}) and "features" not in old:
                old["features"] = tomlkit.table()
            features = old.get("features", {})
            original = previous.get("features", {})
            if key in original:
                features[key] = original[key]
            elif features.get(key) is False:
                features.pop(key, None)
        if "features" in old and not old["features"]:
            del old["features"]
    # The local provider/catalog belong solely to the optional profile.
    providers = old.get("model_providers", {})
    providers.pop("strata", None)
    if "model_providers" in old and not providers:
        del old["model_providers"]
    if old.get("model_catalog_json") == str(catalog_path):
        del old["model_catalog_json"]
    if "sandbox_workspace_write" not in old:
        old["sandbox_workspace_write"] = tomlkit.table()
    old["sandbox_workspace_write"]["network_access"] = True
    old.setdefault("sandbox_mode", "workspace-write")
    catalog = json.loads((SOURCE / "models.json").read_text())
    catalog["models"][0]["base_instructions"] = (SOURCE / "instructions.md").read_text()
    base = tomlkit.dumps(old)
    if migrate:
        base = (
            "\n".join(
                line
                for line in base.splitlines()
                if not line.startswith(("# Local Qwen", "# Keep the local"))
            )
            + "\n"
        )
    write_atomic(catalog_path, json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    write_atomic(profile_path, template)
    write_atomic(path, base)
    if legacy.is_file():
        legacy.unlink()
    print(
        f"Default: {old.get('model', 'Codex built-in default')} ({old.get('model_provider', 'openai')})"
    )
    print(f"Optional Qwen: codex -p qwen; profile: {profile_path}")
    print("Qwen context: 131072; auto compact: 98304; sandbox network: enabled")
    print(f"Previous configuration: {backup}")


if __name__ == "__main__":
    main()
