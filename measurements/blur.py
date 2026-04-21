from .registry import register_operator
from .base import Operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from .motionblur.motionblur import Kernel
import yaml


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
def _build_reflect_index(h: int, w: int, pad: int, device) -> torch.Tensor:
    base = torch.arange(h * w, device=device, dtype=torch.float32).reshape(1, 1, h, w)
    return F.pad(base, (pad, pad, pad, pad), mode="reflect").to(torch.long).squeeze(0).squeeze(0)


def _reflection_unpad_adjoint(x_pad: torch.Tensor, out_hw: tuple[int, int], pad: int) -> torch.Tensor:
    if pad == 0:
        return x_pad

    h, w = out_hw
    index = _build_reflect_index(h=h, w=w, pad=pad, device=x_pad.device)
    flat_index = index.reshape(1, -1).expand(x_pad.shape[0] * x_pad.shape[1], -1)

    x_flat = x_pad.reshape(x_pad.shape[0] * x_pad.shape[1], -1)
    out_flat = torch.zeros((x_flat.shape[0], h * w), device=x_pad.device, dtype=x_pad.dtype)
    out_flat.scatter_add_(1, flat_index, x_flat)
    return out_flat.reshape(x_pad.shape[0], x_pad.shape[1], h, w)


def _depthwise_kernel(kernel_2d: torch.Tensor, channels: int, device, dtype) -> torch.Tensor:
    return (
        kernel_2d.to(device=device, dtype=dtype)
        .reshape(1, 1, *kernel_2d.shape)
        .expand(channels, 1, -1, -1)
        .contiguous()
    )


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None, initialize=True):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1,
                      padding=0, bias=False, groups=3)
        )

        self.k = None
        if initialize:
            self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size),
                       intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


class _LinearBlurOperator(Operator):
    measurement_model = "linear_mse"

    def adjoint(self, p: torch.Tensor, x_like: torch.Tensor | None = None) -> torch.Tensor:
        kernel = getattr(self, "kernel", None)
        if kernel is None:
            raise RuntimeError("Blur operator adjoint requires a realized kernel.")

        channels = int(p.shape[1])
        weight = _depthwise_kernel(kernel, channels=channels, device=p.device, dtype=p.dtype)
        x_pad = F.conv_transpose2d(p, weight, stride=1, padding=0, groups=channels)
        pad = int(weight.shape[-1] // 2)
        return _reflection_unpad_adjoint(x_pad, out_hw=(p.shape[-2], p.shape[-1]), pad=pad)


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='gaussian_blur')
class GaussianBlur(_LinearBlurOperator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel().to(dtype=torch.float32)
        self.conv.update_weights(self.kernel)
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='motion_blur')
class MotionBlur(_LinearBlurOperator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device,
                               initialize=False).to(device)

        self.kernel_object = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        self.kernel = torch.tensor(self.kernel_object.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(self.kernel)
        self.conv.requires_grad_(False)

    def __call__(self, data):
        # A^T * A
        return self.conv(data)

# Adapted from: https://github.com/zhangbingliang2019/DAPS/blob/main/forward_operator/__init__.py
# Original author: bingliang
@register_operator(name='nonlinear_blur')
class NonlinearBlur(Operator):
    def __init__(self, opt_yml_path, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.blur_model.requires_grad_(False)

        np.random.seed(0)
        kernel_np = np.random.randn(1, 512, 2, 2) * 1.2
        random_kernel = (torch.from_numpy(kernel_np)).float().to(self.device)
        self.random_kernel = random_kernel

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        from .bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        return blur_model

    def call_old(self, data):
        data = (data + 1.0) / 2.0  
        blurred = []
        for i in range(data.shape[0]):
            single_blurred = self.blur_model.adaptKernel(
                data[i:i + 1], kernel=self.random_kernel)
            blurred.append(single_blurred)
        blurred = torch.cat(blurred, dim=0)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred

    def __call__(self, data):
        data = (data + 1.0) / 2.0  

        random_kernel = self.random_kernel.repeat(data.shape[0], 1, 1, 1)
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred
