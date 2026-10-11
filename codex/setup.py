# /// script
# requires-python = ">=3.12"
# dependencies = ["tomlkit>=0.13,<1"]
# ///
"""Install local Flash Next, Qwen3.5-9B and Bonsai profiles; preserve GPT defaults."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tomlkit

SOURCE = Path(__file__).resolve().parent
MODEL = "qwen3.8-flash-next-sc117-abliterated-iq3_xxs"
PROFILES = (("flash-next", "models.json", "flash-next-models.json"),
            ("qwen", "models.json", "flash-next-models.json"),
            ("qwen-9b", "qwen-9b-models.json", "qwen-9b-models.json"),
            ("bonsai", "bonsai-models.json", "bonsai-models.json"))
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


def standard_instructions(model):
    """Use the installed CLI's bundled GPT instructions without rewriting them."""
    result = subprocess.run(
        ["codex", "debug", "models", "--bundled"],
        check=True, capture_output=True, text=True,
    )
    models = json.loads(result.stdout)["models"]
    standard = next((entry for entry in models if entry["slug"] == model), None)
    if standard is None:
        standard = next(entry for entry in models if entry["slug"] == "gpt-6.1-sol")
    return {key: standard[key] for key in ("base_instructions", "model_messages")}


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
    catalog_path = root / "flash-next-models.json"
    migrate = old.get("model") == MODEL or old.get("model_provider") == "strata"
    previous = previous_gpt_settings(root) if migrate else None
    instructions = standard_instructions((previous if migrate else old).get("model", "gpt-6.1-sol"))
    subprocess.run(
        ["uv", "sync", "--project", str(SOURCE / "search-mcp"), "--locked"],
        check=True,
    )
    backup = (
        root
        / "config-backups"
        / datetime.now(timezone.utc).strftime("flash-next-%Y%m%d-%H%M%S-%f")
    )
    backup.mkdir(parents=True)
    legacy = root / "qwen-local.config.toml"
    originals = [path, legacy]
    for name, _, filename in PROFILES:
        originals.extend((root / f"{name}.config.toml", root / filename))
    for original in originals:
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
    for name, source_catalog, filename in PROFILES:
        profile_path = root / f"{name}.config.toml"
        template = (SOURCE / f"{name}.config.toml").read_text().replace(
            f'"@CODEX_HOME@/{filename}"', json.dumps(str(root / filename))
        ).replace("@ROOT@", str(SOURCE.parent))
        catalog = json.loads((SOURCE / source_catalog).read_text())
        catalog["models"][0].update(instructions)
        # Keep manually added project trust, providers and MCP servers.
        previous_profile = profile_path
        if name == "flash-next" and not previous_profile.is_file():
            # Carry project trust and user MCP settings across the renamed profile.
            previous_profile = root / "qwen.config.toml"
        profile = tomlkit.parse(previous_profile.read_text()) if previous_profile.is_file() else tomlkit.document()
        for key, value in tomlkit.parse(template).items():
            if isinstance(value, dict) and key in profile:
                if key in ("model_providers", "mcp_servers"):
                    for child, settings in value.items():
                        profile[key].setdefault(child, tomlkit.table()).update(settings)
                else:
                    profile[key].update(value)
            else:
                profile[key] = value
        write_atomic(root / filename, json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
        write_atomic(profile_path, tomlkit.dumps(profile))
        print(f"Optional {name}: codex -p {name}; profile: {profile_path}")
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
    write_atomic(path, base)
    if legacy.is_file():
        legacy.unlink()
    print(
        f"Default: {old.get('model', 'Codex built-in default')} ({old.get('model_provider', 'openai')})"
    )
    print("All local profiles: installed Codex bundled GPT instructions; local_web_search MCP")
    print("qwen is a compatibility alias for flash-next; qwen-9b selects Qwen3.5-9B")
    for name in ("flash-next", "qwen-9b", "bonsai"):
        profile = tomlkit.parse((root / f"{name}.config.toml").read_text())
        print(f"{name}: context {profile['model_context_window']}; auto compact "
              f"{profile['model_auto_compact_token_limit']}; reasoning {profile['model_reasoning_effort']}")
    print(f"Previous configuration: {backup}")


if __name__ == "__main__":
    main()
