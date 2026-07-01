"""CONFIG subsystem — Configuration (Module J)
Directives 36-39.
"""

def resolve_config(experiment=None, cli_args=None):
    """
    Inputs:    optional experiment name, optional CLI override dict.
    Outputs:   fully-resolved PipelineConfig.
    Exceptions: none.
    Side effects: reads YAML files from configs/ directory.
    """
    pass


def validate_config(cfg) -> list:
    """
    Inputs:    PipelineConfig.
    Outputs:   list of ValidationError (empty if valid).
    Exceptions: none.
    Side effects: none.
    """
    pass


def load_and_validate_config(experiment=None, cli_args=None):
    """
    Inputs:    optional experiment name, optional CLI override dict.
    Outputs:   validated PipelineConfig.
    Exceptions: sys.exit(1) on validation failure (Directive 39).
    Side effects: prints all errors, exits non-zero if invalid.
    """
    pass
