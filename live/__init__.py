try:
    from live.indicators import vwap_stack_features
except Exception:  # pragma: no cover - optional dependency hygiene
    def vwap_stack_features(*args, **kwargs):
        raise ImportError("live.indicators dependencies are unavailable")
