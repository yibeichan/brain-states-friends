"""Environment-driven path resolution.

Centralizes ``os.getenv`` lookups so that no machine-specific absolute paths
live in tracked code. Configure the variables in a local ``.env`` file at the
project root (see ``.env.example`` for the full list).
"""

import os

from dotenv import load_dotenv

# Idempotent: searches from the cwd upward for a .env file.
load_dotenv()


def require_env(name: str) -> str:
    """Return the value of environment variable ``name``, or raise if unset.

    Fails fast with an actionable message rather than silently falling back to
    a hard-coded path, so a misconfigured environment is caught immediately.
    """
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable {name!r} is not set. "
            f"Add it to your .env file (see .env.example)."
        )
    return value
