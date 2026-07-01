# Arc2Avatar — API Changelog
> Directive 45: Breaking api.py signature changes are logged here.

## [Unreleased]

### Added
- Initial system scaffold (prebuild phase).
- All 22 subsystem `api.py` files with 4-section docstrings.
- Data contracts in `src/contracts/schemas.py` (GaussianState, FlameMesh, IdentityEmbedding, CameraSample, RenderResult, OptimizerState, ExpressionState, RunManifest).
- Configuration system with 4-layer merge (Directives 36-39).
- Checkpoint lifecycle manager (Directives 46-48).
- GPU resource manager (Directives 49-51).
- Training framework factory (Directives 52-55).
- Error hierarchy with typed exceptions and recovery policy (Directives 62-64).
- Logging framework (Directives 56-58).
- Plugin registry system (Directives 77-79).
- Pipeline scheduler with 4 invocation modes (Directives 73-76).
- 22 subsystem charter files (Directive 0.2).
- Dependency graph and invariants documentation (Directives 0.3, 0.4).
