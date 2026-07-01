"""STATE subsystem — State Management (Module M)
Directives 46-48.
"""

def get_current_state() -> str:
    """
    Inputs:    None.
    Outputs:   current FSM state name.
    Exceptions: none.
    Side effects: none (reads run_state.json from disk).
    """
    pass


def transition(target_state: str) -> None:
    """
    Inputs:    target state name.
    Outputs:   None.
    Exceptions: raises InvalidStateTransitionError if transition not in legal table.
    Side effects: persists new state to run_state.json immediately.
    """
    pass


def save_checkpoint(state, path: str, schema_version: int) -> str:
    """
    Inputs:    state object, path, schema version.
    Outputs:   content hash of saved file.
    Exceptions: raises IOError on write failure.
    Side effects: writes checkpoint atomically (temp file then rename).
    """
    pass


def validate_checkpoint(path: str, expected_version: int) -> bool:
    """
    Inputs:    path, expected schema version.
    Outputs:   True if valid, False otherwise.
    Exceptions: none.
    Side effects: none.
    """
    pass


def load_checkpoint(path: str, expected_version: int) -> object:
    """
    Inputs:    path, expected schema version.
    Outputs:   loaded state object.
    Exceptions: raises ValueError on validation failure (Invariant I4).
    Side effects: calls validate_checkpoint internally.
    """
    pass


def replace_checkpoint(old_path: str, new_state, schema_version: int) -> None:
    """
    Inputs:    old path, new state, schema version.
    Outputs:   None.
    Exceptions: none.
    Side effects: archives old checkpoint before writing new one.
    """
    pass


def prune_archive(older_than_days: int) -> int:
    """
    Inputs:    age threshold in days.
    Outputs:   number of pruned files.
    Exceptions: none.
    Side effects: deletes archived checkpoints older than N days.
    """
    pass
