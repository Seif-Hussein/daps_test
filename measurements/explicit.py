import math

import torch
import torch.nn.functional as F
from scipy.special import lambertw

from .blur import GaussianBlur, MotionBlur, NonlinearBlur
from .compressionquantization import CompressionQuantization
from .downsample import DownSampling
from .hdr import HighDynamicRange
from .inpainting import Inpainting
from .phaseretrieval import PhaseRetrieval
from .registry import register_operator
from .transmission_ct import TransmissionCT


class ExplicitOperatorMixin:
    is_explicit_operator = True
    measurement_model = "generic"

    def describe_operator(self):
        return {
            "name": getattr(self, "name", self.__class__.__name__),
            "measurement_model": getattr(self, "measurement_model", "generic"),
        }


class ExplicitLinearOperatorMixin(ExplicitOperatorMixin):
    measurement_model = "linear_mse"

    def adjoint(self, p: torch.Tensor, x_like: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("Explicit linear operators must implement adjoint().")


def _broadcast_param(value, target: torch.Tensor) -> torch.Tensor:
    vec = torch.as_tensor(value, device=target.device, dtype=target.dtype).reshape(-1)
    if vec.numel() == 1:
        vec = vec.expand(target.shape[0])
    elif vec.numel() != target.shape[0]:
        raise ValueError(f"Expected 1 or {target.shape[0]} values, got {vec.numel()}.")
    return vec.reshape((-1,) + (1,) * (target.dim() - 1))


def _apply_resizer_adjoint(resizer, y: torch.Tensor, output_shape: list[int]) -> torch.Tensor:
    x = y
    ops = list(zip(resizer.sorted_dims, resizer.field_of_view, resizer.weights))
    for dim, fov, w in reversed(ops):
        x = torch.transpose(x, dim, 0)
        out_len = int(output_shape[dim])

        x_flat = x.reshape(x.shape[0], -1)
        out_flat = torch.zeros((out_len, x_flat.shape[1]), device=x.device, dtype=x.dtype)

        fov_2d = fov.to(device=x.device, dtype=torch.long).reshape(fov.shape[0], fov.shape[1])
        w_2d = w.to(device=x.device, dtype=x.dtype).reshape(w.shape[0], w.shape[1])
        for tap in range(fov_2d.shape[0]):
            contrib = x_flat * w_2d[tap].unsqueeze(-1)
            out_flat.index_add_(0, fov_2d[tap], contrib)

        x = out_flat.reshape((out_len,) + x.shape[1:])
        x = torch.transpose(x, dim, 0)
    return x


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
    return kernel_2d.to(device=device, dtype=dtype).reshape(1, 1, *kernel_2d.shape).expand(channels, 1, -1, -1).contiguous()


@register_operator(name="down_sampling_explicit")
class DownSamplingExplicit(DownSampling, ExplicitLinearOperatorMixin):
    def __init__(self, resolution=256, scale_factor=4, device="cuda", sigma=0.05, channels=3):
        self.resolution = int(resolution)
        self.scale_factor = float(scale_factor)
        self.channels = int(channels)
        self.device = device
        super().__init__(resolution=resolution, scale_factor=scale_factor, device=device, sigma=sigma, channels=channels)

    def adjoint(self, p: torch.Tensor, x_like: torch.Tensor | None = None) -> torch.Tensor:
        if x_like is not None:
            output_shape = list(x_like.shape)
        else:
            output_shape = [p.shape[0], self.channels, self.resolution, self.resolution]
        return _apply_resizer_adjoint(self.down_sample, p, output_shape)


@register_operator(name="inpainting_explicit")
class InpaintingExplicit(Inpainting, ExplicitLinearOperatorMixin):
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, device="cuda",
                 sigma=0.05):
        self.resolution = int(resolution)
        self.device = device
        super().__init__(
            mask_type=mask_type,
            mask_len_range=mask_len_range,
            mask_prob_range=mask_prob_range,
            resolution=resolution,
            device=device,
            sigma=sigma,
        )

    def _ensure_mask(self, x_like: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            self.mask = self.mask_gen(torch.zeros_like(x_like))
            self.mask = self.mask[0:1, 0:1, :, :]
        return self.mask.to(device=x_like.device, dtype=x_like.dtype)

    def __call__(self, x):
        mask = self._ensure_mask(x)
        return x * mask

    def adjoint(self, p: torch.Tensor, x_like: torch.Tensor | None = None) -> torch.Tensor:
        template = x_like if x_like is not None else p
        mask = self._ensure_mask(template)
        return p * mask


class _ExplicitBlurBase(ExplicitLinearOperatorMixin):
    def _blur_adjoint(self, p: torch.Tensor) -> torch.Tensor:
        kernel = getattr(self, "kernel", None)
        if kernel is None and hasattr(self, "conv") and hasattr(self.conv, "get_kernel"):
            kernel = self.conv.get_kernel()
        if hasattr(kernel, "kernelMatrix"):
            kernel = torch.as_tensor(kernel.kernelMatrix)
        if kernel is None:
            raise RuntimeError("Explicit blur operator requires a realized kernel.")

        channels = int(p.shape[1])
        weight = _depthwise_kernel(kernel, channels=channels, device=p.device, dtype=p.dtype)
        x_pad = F.conv_transpose2d(p, weight, stride=1, padding=0, groups=channels)
        pad = int(weight.shape[-1] // 2)
        return _reflection_unpad_adjoint(x_pad, out_hw=(p.shape[-2], p.shape[-1]), pad=pad)

    def adjoint(self, p: torch.Tensor, x_like: torch.Tensor | None = None) -> torch.Tensor:
        return self._blur_adjoint(p)


@register_operator(name="gaussian_blur_explicit")
class GaussianBlurExplicit(GaussianBlur, _ExplicitBlurBase):
    def __init__(self, kernel_size, intensity, device="cuda", sigma=0.05):
        self.kernel_size = int(kernel_size)
        self.intensity = float(intensity)
        self.device = device
        super().__init__(kernel_size=kernel_size, intensity=intensity, device=device, sigma=sigma)


@register_operator(name="motion_blur_explicit")
class MotionBlurExplicit(MotionBlur, _ExplicitBlurBase):
    def __init__(self, kernel_size, intensity, device="cuda", sigma=0.05):
        self.kernel_size = int(kernel_size)
        self.intensity = float(intensity)
        self.device = device
        super().__init__(kernel_size=kernel_size, intensity=intensity, device=device, sigma=sigma)


@register_operator(name="phase_retrieval_explicit")
class PhaseRetrievalExplicit(PhaseRetrieval, ExplicitOperatorMixin):
    measurement_model = "phase_retrieval"

    def primal_adjoint(self, p: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        return 0.5 * self.adjoint_complex(p, out_hw=out_hw)

    @staticmethod
    def dual_prox(w: torch.Tensor, y_amp: torch.Tensor, sigma_dual: float, sigma_n: float,
                  eps: float = 1.0e-12) -> torch.Tensor:
        sigma_dual_view = _broadcast_param(sigma_dual, w)
        a = sigma_dual_view * (sigma_n ** 2)
        r0 = w.abs()
        r_star = (a * r0 + y_amp) / (a + 1.0)
        return w * (r_star / (r0 + eps))


@register_operator(name="transmission_ct_explicit")
class TransmissionCTExplicit(TransmissionCT, ExplicitOperatorMixin):
    measurement_model = "transmission_ct"

    @staticmethod
    def _lambertw_principal(x: torch.Tensor) -> torch.Tensor:
        x_cpu = x.detach().to(device="cpu", dtype=torch.float64).numpy()
        w_cpu = lambertw(x_cpu, k=0).real
        return torch.from_numpy(w_cpu).to(device=x.device, dtype=x.dtype)

    def dual_prox(self, w: torch.Tensor, y_counts: torch.Tensor, sigma_dual: float,
                  eps: float = 1.0e-12) -> torch.Tensor:
        sigma = _broadcast_param(sigma_dual, w)
        eta = torch.as_tensor(float(getattr(self, "eta", 1.0)), device=w.device, dtype=w.dtype)
        i0 = self.incident_counts(w).to(device=w.device, dtype=w.dtype)
        arg = (eta * i0 / sigma.clamp_min(eps)) * torch.exp(
            (-w + (eta * y_counts) / sigma).clamp(min=-80.0, max=80.0)
        )
        arg = arg.clamp_min(0.0)
        return w - (eta * y_counts) / sigma + self._lambertw_principal(arg)


@register_operator(name="high_dynamic_range_explicit")
class HighDynamicRangeExplicit(HighDynamicRange, ExplicitOperatorMixin):
    measurement_model = "pointwise_clipped_gaussian"


@register_operator(name="compression_quantization_explicit")
class CompressionQuantizationExplicit(CompressionQuantization, ExplicitOperatorMixin):
    measurement_model = "quantized_linear"

    def __init__(self, compression_factor, nbits, device="cuda", sigma=0.05):
        super().__init__(compression_factor=compression_factor, nbits=nbits, device=device, sigma=sigma)
        self.input_shape = None

    def __call__(self, x):
        if self.input_shape is None:
            self.input_shape = tuple(x.shape[-3:])
        return super().__call__(x)

    def adjoint_unquantized(self, p: torch.Tensor) -> torch.Tensor:
        if self.A is None or self.input_shape is None:
            raise RuntimeError("Compression matrix A is not initialized. Call the operator once before adjoint_unquantized().")
        x = self.A.transpose(-1, -2) @ p.contiguous().view(*p.shape[:-2], -1, 1)
        return x.view(*p.shape[:-2], *self.input_shape)


@register_operator(name="nonlinear_blur_explicit")
class NonlinearBlurExplicit(NonlinearBlur, ExplicitOperatorMixin):
    measurement_model = "nonlinear_blur"
