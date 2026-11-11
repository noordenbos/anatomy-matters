"""Read the shared classifier palette config; never assign colors by position.

Reference colors, aliases, provenance, and explicit temporary fallbacks live in
``config/classifier_colors.json``. Returned dictionaries are independent copies.
Unknown classifier labels fail clearly rather than becoming missing-data gray.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import warnings

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "classifier_colors.json"


class PaletteFallbackWarning(UserWarning):
    """A known study label needs an explicitly documented local color."""


@lru_cache(maxsize=1)
def _config():
    config = json.loads(CONFIG_PATH.read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported classifier palette config version")
    seen = set()
    for key, spec in config["classifiers"].items():
        for binding in [key, *spec["bindings"]]:
            if binding in seen:
                raise ValueError(f"Duplicate classifier binding: {binding}")
            seen.add(binding)
        colors = {**spec["reference_colors"], **spec.get("local_fallback", {}).get("colors", {})}
        if not all(re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for c in colors.values()):
            raise ValueError(f"Invalid hex color in {key}")
        if not set(spec["label_aliases"].values()).issubset(colors):
            raise ValueError(f"Alias without a color in {key}")
    return config


def classifier_key(name: str) -> str | None:
    """Resolve a classifier ID, metadata column, or existing figure heading."""
    for key, spec in _config()["classifiers"].items():
        if name == key or name in spec["bindings"]:
            return key
    return None


def _spec(name):
    key = classifier_key(name)
    if key is None:
        raise KeyError(f"No classifier palette configured for {name!r}")
    return _config()["classifiers"][key]


def _label(value):
    return str(value).strip()


def is_missing_label(value) -> bool:
    return _label(value).casefold() in {s.casefold() for s in _config()["missing"]["tokens"]}


def missing_color() -> str:
    return _config()["missing"]["color"]


def reference_palette(name: str) -> dict[str, str]:
    """Original source labels and colors, without local exceptions or aliases."""
    return dict(_spec(name)["reference_colors"])


def classifier_palette(name: str, values=None) -> dict[str, str]:
    """Named colors, optionally validated/subsetted against observed labels.

    Pass observed values when plotting. This handles missing tokens, warns for
    unresolved legacy labels, and rejects unconfigured labels. Category order
    cannot change the assigned color. Aliases only affect color lookup, not data.
    """
    spec = _spec(name)
    fallback = spec.get("local_fallback", {})
    palette = {**spec["reference_colors"], **fallback.get("colors", {})}
    for alias, target in spec["label_aliases"].items():
        palette[alias] = palette[target]
    palette.update(dict.fromkeys(_config()["missing"]["tokens"], missing_color()))
    if values is None:
        return palette
    labels = list(dict.fromkeys(_label(v) for v in values))
    unknown = [v for v in labels if v not in palette and not is_missing_label(v)]
    if unknown:
        raise ValueError(f"Unconfigured labels for {name}: {unknown}. Update {CONFIG_PATH.name} explicitly.")
    fallback_labels = [v for v in labels if spec["label_aliases"].get(v, v) in fallback.get("colors", {})]
    if fallback_labels and fallback.get("warn"):
        warnings.warn(f"{spec['name']}: {fallback['reason']} Labels: {fallback_labels}", PaletteFallbackWarning, stacklevel=2)
    return {v: missing_color() if is_missing_label(v) else palette[v] for v in labels}


def classifier_color(name: str, value) -> str:
    return classifier_palette(name, [value])[_label(value)]


def classifier_bindings() -> dict[str, str]:
    """All recognized classifier identifiers, including historical headings."""
    return {binding: key for key, spec in _config()["classifiers"].items() for binding in [key, *spec["bindings"]]}


def classifier_reference_label(name: str, value) -> str:
    """Resolve a stored code to its configured published subtype name."""
    label = _label(value)
    return _spec(name)["label_aliases"].get(label, label)
