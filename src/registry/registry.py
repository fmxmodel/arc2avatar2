"""
Arc2Avatar — Plugin Registry System (Module V / Directive 78)
===============================================================
Exposes register(kind, name, cls) and get(kind, name) -> type
for exactly five registrable kinds: datasets, models, losses, renderers, optimizers.

Config files reference plugins by string name, resolved through this registry
at startup — never by direct import path in config (Directive 78).
"""

from typing import Any, Callable, Dict, Optional, Type


_REGISTRY: Dict[str, Dict[str, Type]] = {
    "datasets": {},
    "models": {},
    "losses": {},
    "renderers": {},
    "optimizers": {},
}

_VALID_KINDS = set(_REGISTRY.keys())


def register(kind: str, name: str, cls: Type) -> None:
    """Register a plugin class.

    Inputs:    kind (one of: datasets, models, losses, renderers, optimizers),
               name (string identifier, e.g., "arcface_v1"),
               cls (the class to register).
    Outputs:   None.
    Exceptions: raises ValueError if kind is not one of the valid kinds.
    Side effects: adds class to internal registry.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"Invalid registry kind: '{kind}'. "
            f"Valid kinds: {sorted(_VALID_KINDS)}"
        )
    if not isinstance(name, str) or not name:
        raise ValueError(f"Registry name must be a non-empty string, got: {name}")

    _REGISTRY[kind][name] = cls
    print(f"[REGISTRY] Registered {kind}/{name}: {cls.__module__}.{cls.__qualname__}")


def get(kind: str, name: str) -> Type:
    """Retrieve a registered plugin class by kind and name.

    Inputs:    kind (string), name (string).
    Outputs:   registered class.
    Exceptions: raises KeyError if kind or name not found.
    Side effects: none.
    """
    if kind not in _VALID_KINDS:
        raise KeyError(
            f"Unknown registry kind: '{kind}'. Valid kinds: {sorted(_VALID_KINDS)}"
        )
    if name not in _REGISTRY[kind]:
        raise KeyError(
            f"'{name}' not registered in kind '{kind}'. "
            f"Registered: {list(_REGISTRY[kind].keys())}"
        )
    return _REGISTRY[kind][name]


def list_registered(kind: Optional[str] = None) -> Dict[str, list]:
    """List all registered plugins, optionally filtered by kind.

    Inputs:    optional kind filter.
    Outputs:   dict of kind -> list of registered names.
    Side effects: none.
    """
    if kind:
        if kind not in _VALID_KINDS:
            return {}
        return {kind: list(_REGISTRY[kind].keys())}
    return {k: list(v.keys()) for k, v in _REGISTRY.items()}


def resolve_from_config(config: Any, kind: str, name_field: str) -> Type:
    """Utility: resolve a class from a config field.

    Convenience for: get(kind, getattr(config, name_field))

    Inputs:    config object, kind string, config field name holding the plugin name.
    Outputs:   registered class.
    """
    name = getattr(config, name_field, None)
    if not name:
        raise ValueError(
            f"Config field '{name_field}' is empty or missing "
            f"for registry kind '{kind}'"
        )
    return get(kind, name)
