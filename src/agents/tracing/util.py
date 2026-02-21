from .setup import get_trace_provider


def time_iso() -> str:
    return get_trace_provider().time_iso()


def gen_trace_id() -> str:
    return get_trace_provider().gen_trace_id()


def gen_span_id() -> str:
    """Generate a new span ID."""
    return get_trace_provider().gen_span_id()


def gen_group_id() -> str:
    """Generate a new group ID."""
    return get_trace_provider().gen_group_id()
