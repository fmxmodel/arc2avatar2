"""EXT subsystem — Extensibility (Module V)
Directives 77-79.
"""

class IdentityEncoder:
    """Abstract base class for identity encoders (Directive 77)."""
    def encode(self, image) -> object:
        raise NotImplementedError


class Renderer:
    """Abstract base class for renderers."""
    def render(self, state, camera) -> object:
        raise NotImplementedError


class MeshModel:
    """Abstract base class for mesh models."""
    pass


class AnimationDriver:
    """Abstract base class for animation drivers."""
    def apply(self, state, coeffs) -> object:
        raise NotImplementedError
