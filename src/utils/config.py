"""
Configuration Management for the NCKH Image Captioning system.

Provides YAML-based configuration loading, nested attribute access,
config merging, CLI argument overrides, and config serialization.

Example usage:
    >>> config = load_config("configs/base_config.yaml")
    >>> print(config.model.d_model)  # 512
    >>> print(config.training.stage1.lr)  # 0.0001
    >>> config.save("outputs/experiment_config.yaml")
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ConfigNode:
    """A configuration node that supports nested dot-notation attribute access.

    Wraps a dictionary so that nested keys can be accessed as attributes,
    e.g. ``config.model.d_model`` instead of ``config['model']['d_model']``.

    Attributes are stored internally in a plain ``dict`` and can be iterated,
    converted back to a dict, or pretty-printed.

    Args:
        data: A dictionary of configuration key-value pairs.
        name: Optional human-readable name for this node (used in repr).
    """

    _RESERVED = frozenset({
        "_data", "_name", "to_dict", "get", "keys", "values", "items",
        "update", "merge", "pretty", "save", "freeze", "unfreeze",
        "_frozen",
    })

    def __init__(self, data: Optional[Dict[str, Any]] = None, name: str = "root") -> None:
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_frozen", False)
        if data is not None:
            for key, value in data.items():
                self._set_item(key, value)

    # --- Internal helpers ---------------------------------------------------

    def _set_item(self, key: str, value: Any) -> None:
        """Recursively wrap nested dicts as ConfigNode objects."""
        if isinstance(value, dict):
            value = ConfigNode(value, name=f"{self._name}.{key}")
        elif isinstance(value, list):
            value = [
                ConfigNode(v, name=f"{self._name}.{key}[{i}]")
                if isinstance(v, dict) else v
                for i, v in enumerate(value)
            ]
        self._data[key] = value

    # --- Attribute access ---------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(
                f"ConfigNode '{self._name}' has no attribute '{key}'. "
                f"Available keys: {list(self._data.keys())}"
            )

    def __setattr__(self, key: str, value: Any) -> None:
        if self._frozen:
            raise RuntimeError(
                f"Cannot set '{key}' on frozen ConfigNode '{self._name}'."
            )
        self._set_item(key, value)

    def __delattr__(self, key: str) -> None:
        if self._frozen:
            raise RuntimeError(
                f"Cannot delete '{key}' on frozen ConfigNode '{self._name}'."
            )
        if key in self._data:
            del self._data[key]
        else:
            raise AttributeError(
                f"ConfigNode '{self._name}' has no attribute '{key}'."
            )

    # --- Dict-like interface ------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with an optional default.

        Args:
            key: The configuration key.
            default: Value to return if key is not found.

        Returns:
            The configuration value or *default*.
        """
        return self._data.get(key, default)

    def keys(self):
        """Return the configuration keys."""
        return self._data.keys()

    def values(self):
        """Return the configuration values."""
        return self._data.values()

    def items(self):
        """Return the configuration key-value pairs."""
        return self._data.items()

    # --- Conversion ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert the ConfigNode back to a plain dictionary.

        Returns:
            A nested dictionary representation of the configuration.
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    v.to_dict() if isinstance(v, ConfigNode) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    # --- Merging ------------------------------------------------------------

    def update(self, other: Union["ConfigNode", Dict[str, Any]]) -> None:
        """Update this config with values from *other* (shallow merge).

        Args:
            other: Another ConfigNode or plain dict whose values overwrite
                   matching keys in this node.
        """
        if isinstance(other, ConfigNode):
            other = other.to_dict()
        for key, value in other.items():
            self._set_item(key, value)

    def merge(self, other: Union["ConfigNode", Dict[str, Any]]) -> None:
        """Deep-merge *other* into this config.

        Nested dicts are merged recursively; non-dict values are overwritten.

        Args:
            other: Another ConfigNode or dict to merge.
        """
        if isinstance(other, ConfigNode):
            other = other.to_dict()
        _deep_merge(self._data, other, node_name=self._name)

    # --- Freeze / unfreeze --------------------------------------------------

    def freeze(self) -> None:
        """Freeze this node and all children, preventing further mutation."""
        object.__setattr__(self, "_frozen", True)
        for value in self._data.values():
            if isinstance(value, ConfigNode):
                value.freeze()

    def unfreeze(self) -> None:
        """Unfreeze this node and all children, allowing mutation."""
        object.__setattr__(self, "_frozen", False)
        for value in self._data.values():
            if isinstance(value, ConfigNode):
                value.unfreeze()

    # --- Serialization ------------------------------------------------------

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the configuration to a YAML file.

        Args:
            filepath: Destination file path. Parent directories are created
                      automatically.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    # --- Pretty print -------------------------------------------------------

    def pretty(self, indent: int = 0) -> str:
        """Return a human-readable, indented string of the configuration.

        Args:
            indent: Current indentation level (used internally for recursion).

        Returns:
            A formatted string representation.
        """
        lines: List[str] = []
        prefix = "  " * indent
        for key, value in self._data.items():
            if isinstance(value, ConfigNode):
                lines.append(f"{prefix}{key}:")
                lines.append(value.pretty(indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, ConfigNode):
                        lines.append(f"{prefix}  -")
                        lines.append(item.pretty(indent + 2))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    # --- Repr / Str ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"ConfigNode(name='{self._name}', keys={list(self._data.keys())})"

    def __str__(self) -> str:
        return self.pretty()


# =============================================================================
# Module-level helper functions
# =============================================================================


def _deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
    node_name: str = "root",
) -> None:
    """Recursively merge *override* into *base* in-place.

    Args:
        base: The base dictionary to merge into.
        override: The override dictionary whose values take precedence.
        node_name: Name for logging / debugging.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], ConfigNode)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key]._data, value, node_name=f"{node_name}.{key}")
        elif (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value, node_name=f"{node_name}.{key}")
        else:
            if isinstance(value, dict):
                base[key] = ConfigNode(value, name=f"{node_name}.{key}")
            else:
                base[key] = value


def load_config(filepath: Union[str, Path]) -> ConfigNode:
    """Load a YAML configuration file and return a ConfigNode.

    Args:
        filepath: Path to the YAML configuration file.

    Returns:
        A ConfigNode wrapping the loaded configuration.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return ConfigNode(raw, name=filepath.stem)


def merge_configs(
    base: Union[ConfigNode, Dict[str, Any]],
    *overrides: Union[ConfigNode, Dict[str, Any], str, Path],
) -> ConfigNode:
    """Merge a base configuration with one or more overrides.

    Each override can be a ConfigNode, a dict, or a path to a YAML file.
    Overrides are applied left-to-right; later values win.

    Args:
        base: The base configuration (ConfigNode, dict, or YAML path).
        *overrides: Additional configurations to merge on top.

    Returns:
        A new ConfigNode containing the merged result.

    Example:
        >>> merged = merge_configs(
        ...     "configs/base_config.yaml",
        ...     "configs/fusion_cross_attention.yaml",
        ...     {"training": {"stage1": {"lr": 5e-5}}},
        ... )
    """
    # Resolve base
    if isinstance(base, (str, Path)):
        base = load_config(base)
    elif isinstance(base, dict):
        base = ConfigNode(base)

    # Deep copy so the original is not mutated
    merged_dict = copy.deepcopy(base.to_dict())
    merged = ConfigNode(merged_dict, name="merged")

    for override in overrides:
        if isinstance(override, (str, Path)):
            override = load_config(override)
        if isinstance(override, ConfigNode):
            override = override.to_dict()
        merged.merge(override)

    return merged


class ConfigManager:
    """High-level configuration manager with CLI override support.

    Loads a base YAML config, optionally merges experiment-specific overrides,
    and applies any command-line ``--set`` arguments.

    Args:
        base_config_path: Path to the base YAML config.
        experiment_config_path: Optional path to an experiment-specific override.

    Example:
        >>> manager = ConfigManager("configs/base_config.yaml")
        >>> config = manager.parse()          # parses sys.argv
        >>> print(config.model.d_model)
        512

    CLI override syntax::

        python train.py --config configs/base_config.yaml \\
            --set model.d_model=256 training.stage1.lr=5e-5
    """

    def __init__(
        self,
        base_config_path: Optional[Union[str, Path]] = None,
        experiment_config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.base_config_path = base_config_path
        self.experiment_config_path = experiment_config_path

    # --- CLI argument parsing -----------------------------------------------

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        """Build the CLI argument parser.

        Returns:
            An :class:`argparse.ArgumentParser` with ``--config``,
            ``--experiment``, and ``--set`` arguments.
        """
        parser = argparse.ArgumentParser(
            description="NCKH Image Captioning - Configuration",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--config",
            type=str,
            default="configs/base_config.yaml",
            help="Path to the base configuration YAML file.",
        )
        parser.add_argument(
            "--experiment",
            type=str,
            default=None,
            help="Path to an experiment-specific override YAML file.",
        )
        parser.add_argument(
            "--set",
            nargs="*",
            default=[],
            dest="overrides",
            help=(
                "Override config values via key=value pairs "
                "(e.g. --set model.d_model=256 training.stage1.lr=5e-5)."
            ),
        )
        return parser

    def parse(self, args: Optional[List[str]] = None) -> ConfigNode:
        """Parse CLI arguments and return the final merged config.

        Args:
            args: Explicit argument list. If ``None``, ``sys.argv[1:]`` is used.

        Returns:
            A fully merged ConfigNode ready for use.
        """
        parser = self.build_arg_parser()
        parsed, _ = parser.parse_known_args(args)

        # Determine config paths
        base_path = self.base_config_path or parsed.config
        exp_path = self.experiment_config_path or parsed.experiment

        # Load and merge
        config = load_config(base_path)
        if exp_path is not None:
            config = merge_configs(config, exp_path)

        # Apply CLI overrides
        if parsed.overrides:
            cli_dict = self._parse_overrides(parsed.overrides)
            config.merge(cli_dict)

        return config

    @staticmethod
    def _parse_overrides(overrides: List[str]) -> Dict[str, Any]:
        """Convert a list of ``key=value`` strings into a nested dict.

        Supports dotted keys (e.g. ``model.d_model=256``).  Values are
        automatically cast to ``int``, ``float``, ``bool``, or ``None`` when
        possible; otherwise they remain as strings.

        Args:
            overrides: List of ``key=value`` strings.

        Returns:
            A nested dictionary suitable for merging.
        """
        result: Dict[str, Any] = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(
                    f"Invalid override format: '{item}'. Expected key=value."
                )
            key, value_str = item.split("=", 1)
            value = _auto_cast(value_str)

            # Build nested dict from dotted key
            keys = key.split(".")
            d = result
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        return result


def _auto_cast(value: str) -> Any:
    """Attempt to cast a string value to a Python literal.

    Tries, in order: ``None``, ``bool``, ``int``, ``float``, JSON list/dict.
    Falls back to the original string.

    Args:
        value: The raw string value from a CLI override.

    Returns:
        The cast Python value.
    """
    # None
    if value.lower() in ("none", "null", "~"):
        return None

    # Bool
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # Int
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # JSON list / dict
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    return value
