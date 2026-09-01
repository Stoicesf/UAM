"""SECDO experiments package. Import submodules directly to avoid heavy import chains."""

__all__ = ["run_experiment"]


def __getattr__(name: str):
    if name == "run_experiment":
        from secdo.experiments.runner import run_experiment

        return run_experiment
    raise AttributeError(name)
