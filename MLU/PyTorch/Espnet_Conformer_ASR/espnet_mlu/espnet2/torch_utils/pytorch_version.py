import torch
import torch_mlu


def pytorch_cudnn_version() -> str:
    message = (
        f"pytorch.version={torch.__version__}, "
        f"mlu.available={torch.mlu.is_available()}, "
    )

    if torch.backends.cudnn.enabled:
        message += (
            f"cudnn.version={torch.backends.cudnn.version()}, "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark}, "
            f"cudnn.deterministic={torch.backends.cudnn.deterministic}"
        )
    return message
