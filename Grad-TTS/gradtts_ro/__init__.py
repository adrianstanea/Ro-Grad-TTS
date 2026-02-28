def __getattr__(name):
    if name == "GradTTSRomanian":
        from gradtts_ro.pipeline import GradTTSRomanian
        return GradTTSRomanian
    raise AttributeError(f"module 'gradtts_ro' has no attribute {name}")

__all__ = ["GradTTSRomanian"]
