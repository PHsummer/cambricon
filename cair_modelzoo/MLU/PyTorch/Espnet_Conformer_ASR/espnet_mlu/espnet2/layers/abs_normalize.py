from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch_mlu


class AbsNormalize(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # return output, output_lengths
        raise NotImplementedError
