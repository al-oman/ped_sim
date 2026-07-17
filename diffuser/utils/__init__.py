from .training import EMA, Trainer


def device_arg(default="cpu"):
    """Pull an optional --device=cuda|mps|cpu flag off the command line, so
    the same scripts run on the HPC unchanged. Removes the flag from
    sys.argv so positional arguments keep working."""
    import sys
    for a in sys.argv:
        if a.startswith("--device="):
            sys.argv.remove(a)
            return a.split("=", 1)[1]
    return default
