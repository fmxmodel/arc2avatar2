"""ANIM subsystem — Animation Rig & Expression Refinement (Module G)
Directives 24-27.
"""

def apply_blendshapes(gaussian_state, flame_mesh, expression_state) -> object:
    """
    Inputs:    GaussianState, FlameMesh, ExpressionState.
    Outputs:   deformed GaussianState (new means and rotated covariances).
    Exceptions: none (pure deformation function).
    Side effects: none.
    """
    pass


def detect_mouth_opening(expression_state, config) -> bool:
    """
    Inputs:    ExpressionState, AnimationConfig.
    Outputs:   True if expression requires refinement.
    Exceptions: none.
    Side effects: none.
    """
    pass


def run_refinement(gaussian_state, identity, prior_model, config) -> object:
    """
    Inputs:    GaussianState, IdentityEmbedding, frozen prior, AnimationConfig.
    Outputs:   refined ExpressionState with mouth-interior patch.
    Exceptions: raises DivergenceError if refinement diverges.
    Side effects: saves expr_<name>_refined.pt per expression.
    """
    pass


def check_identity_preservation(rendered_refined, original_embedding, config) -> float:
    """
    Inputs:    rendered refined expression image, original IdentityEmbedding, AnimationConfig.
    Outputs:   cosine similarity score.
    Exceptions: raises ValidationError if similarity below threshold.
    Side effects: none.
    """
    pass
