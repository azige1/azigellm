def initialize_worker() -> None:
    from . import losses  # noqa: F401  确保在 Ray worker 内注册 sampled_reverse_kl
    from .patches import install_runtime_patches

    install_runtime_patches()
