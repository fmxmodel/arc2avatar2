# Arc2Avatar — Extension Points
> Directive 79: Documented contracts for each of the four ABCs.

## IdentityEncoder
**Abstract method:** `encode(image: torch.Tensor) -> IdentityEmbedding`

**Input:** `image` tensor `[3, H, W]` float32, range [0, 1].
**Output:** `IdentityEmbedding` with `vector [512]` float32 L2-normalized.

**Invariants to preserve:**
- I3: Output vector must be 512-d and L2-normalized.

**Default implementation:** ArcFace-based encoder.

**Worked example (dummy):**
```python
from src.registry.registry import register
from src.contracts.schemas import IdentityEmbedding
from src.ext.api import IdentityEncoder

class DummyEncoder(IdentityEncoder):
    def encode(self, image):
        return IdentityEmbedding(
            vector=torch.zeros(512),
            source_image_hash="dummy"
        )

register("models", "dummy_encoder", DummyEncoder)
```

## Renderer
**Abstract method:** `render(state: GaussianState, camera: CameraSample) -> RenderResult`

**Input:** `GaussianState`, `CameraSample`.
**Output:** `RenderResult` with `image [3, H, W]` float32.

**Invariants to preserve:**
- I2: vertex_id length match is verified at GaussianState construction.

## MeshModel
**Abstract interface:** Must expose fixed, versioned vertex topology compatible with Invariant I1.

## AnimationDriver
**Abstract method:** `apply(state: GaussianState, coeffs: torch.Tensor) -> GaussianState`

**Input:** `GaussianState`, blendshape coefficient tensor.
**Output:** Deformed `GaussianState`.
