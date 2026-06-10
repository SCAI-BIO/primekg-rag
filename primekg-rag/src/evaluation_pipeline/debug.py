from .config import DEBUG_GENE, DEBUG_LOG_PATH


_debug_fh = None


def debug_write(msg: str):
    if not DEBUG_GENE:
        return

    global _debug_fh

    if _debug_fh is None:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _debug_fh = open(DEBUG_LOG_PATH, "w", encoding="utf-8")

    _debug_fh.write(msg + "\n")
    _debug_fh.flush()

