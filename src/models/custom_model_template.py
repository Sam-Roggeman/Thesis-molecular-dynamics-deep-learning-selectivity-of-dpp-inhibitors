from abc import abstractmethod, ABC

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str


class AbstractNNModel(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs):
        pass
    @abstractmethod
    def input_shape(self):
        pass
    def count_parameters(self):
         # Count total number of trainable parameters in the model
         return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self):
       # Create a dummy input tensor with the same shape as the model's expected input
       input_size = self.input_shape() # This method should be implemented by the subclass to return the correct input shape
       inputs = torch.randn(1, *input_size)  # Batch size of 1
       # move inputs to the same device as the model
       inputs = inputs.to(next(self.parameters()).device)
       flops = FlopCountAnalysis(self, inputs)
       return flops

    def get_vram_usage(self):
         """
         :return: VRAM usage of the model parameters in bytes
         """
         # Calculate VRAM usage for the model parameters
         param_vram = 0
         for param in self.parameters():
             param_vram += param.numel() * param.element_size()

         # Calculate VRAM usage for the input tensor
         input_size = self.input_shape()
         inputs = torch.randn(1, *input_size)
         input_vram_batch = inputs.numel() * inputs.element_size()

         return param_vram, input_vram_batch
    def print_info(self):
        print(f"Number of parameters: {self.count_parameters()}")
        flops = self.get_flops()
        print(f"MegaFLOPs: {flops.total() / 1e6}")
        print(f"Model VRAM usage: {self.get_vram_usage() / (1024 ** 2):.2f} MB")
        print(f"\nFLOPs breakdown: \n{flop_count_str(flops)}")
