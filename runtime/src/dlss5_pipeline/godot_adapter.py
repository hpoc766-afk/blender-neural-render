"""Godot HDR input adapter for the recovered static model.

Tone mapping and fixed model shape live here, outside the transport. Motion and
reverse-Z depth remain available in the source packet; this model ignores them.
"""
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .color import linear_to_srgb
from .model import ModelRunner, load_model


class GodotStaticAdapter:
    def __init__(self, checkpoint: Path, exposure: float = 1.0, cuda_graph: bool = True,
                 strength: float = 1.0, hdr_ceiling: float = 64.0):
        if not np.isfinite([strength, hdr_ceiling, exposure]).all() or not 0 <= strength <= 1 or hdr_ceiling <= 0:
            raise ValueError('Strength in [0, 1] and positive HDR ceiling required')
        self.strength = strength
        self.hdr_ceiling = hdr_ceiling
        if exposure <= 0 or not torch.cuda.is_available():
            raise ValueError('Positive exposure and CUDA PyTorch required')
        self.exposure = exposure
        self.model, self.metadata = load_model(checkpoint, torch.device('cuda:0'), 'fp16')
        self.runner = ModelRunner(self.model, use_cuda_graph=cuda_graph)
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]

    @staticmethod
    def spatial_transform(height: int, width: int) -> dict:
        """Model-owned, aspect-preserving 256px letterbox contract."""
        if height < 1 or width < 1:
            raise ValueError('Positive native image dimensions required')
        scale = min(256 / height, 256 / width)
        h, w = max(1, round(height * scale)), max(1, round(width * scale))
        top, left = (256 - h) // 2, (256 - w) // 2
        return {'profile': 'static_square_v1' if (height, width) == (256, 256) else 'static_letterbox_v1',
                'native_size_hw': [height, width], 'model_size_hw': [256, 256],
                'content_box_xywh': [left, top, w, h], 'padding': 'replicate',
                'resize': 'bilinear_align_corners_false_antialias_true',
                'restore_resize': 'bilinear_align_corners_false_antialias_false', 'scale': scale}

    @classmethod
    def native_image(cls, value: np.ndarray, height: int, width: int) -> np.ndarray:
        """Remove model padding before returning colors to the native view."""
        if value.shape == (height, width, 3):
            return np.ascontiguousarray(value, dtype=np.float32)
        if value.shape != (256, 256, 3):
            raise ValueError('Expected native RGB or the declared 256x256 model image')
        left, top, w, h = cls.spatial_transform(height, width)['content_box_xywh']
        cropped = np.ascontiguousarray(value[top:top+h, left:left+w])
        tensor = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0)
        return F.interpolate(tensor, size=(height, width), mode='bilinear',
                             align_corners=False)[0].permute(1, 2, 0).contiguous().numpy()

    def restore_hdr(self, frame: np.ndarray, channels: list[str], source: np.ndarray,
                    output: np.ndarray) -> tuple[np.ndarray, dict]:
        """Invert this adapter's mapping as a residual anchored to original HDR.

        Identical model input/output produces unchanged HDR despite FP16 input
        quantization. The inverse is bounded near 1; this is experimental, not
        the recovered native HDR contract.
        """
        indices = [channels.index(name) for name in ('linear_hdr_r', 'linear_hdr_g', 'linear_hdr_b')]
        original = np.ascontiguousarray(frame[..., indices])
        if not np.isfinite(original).all() or not np.isfinite(source).all() or not np.isfinite(output).all():
            raise ValueError('Finite native and model colors required')
        source = self.native_image(source, *original.shape[:2])
        output = self.native_image(output, *original.shape[:2])
        limit = self.hdr_ceiling * self.exposure / (1 + self.hdr_ceiling * self.exposure)
        def inverse(value):
            bounded = np.clip(value, 0, limit)
            return bounded / (1 - bounded) / self.exposure
        # Signed scene-linear values are valid; clipping the restored base breaks
        # both identity and strength-zero contracts. Only the model inverse is bounded.
        hdr = original if self.strength == 0 else (
            original + self.strength * (inverse(output) - inverse(source)))
        if not np.isfinite(hdr).all():
            raise RuntimeError('Nonfinite restored HDR')
        return np.ascontiguousarray(hdr, dtype='<f4'), {
            'hdr_restore': 'bounded inverse-Reinhard residual anchored to original HDR',
            'strength': self.strength, 'hdr_inverse_ceiling': self.hdr_ceiling,
            'inverse_clamped_channels': int(np.count_nonzero((output < 0) | (output > limit))),
            'restored_hdr_min': float(hdr.min()),
            'restored_hdr_max': float(hdr.max()),
            'spatial_transform': self.spatial_transform(*original.shape[:2]),
            'hdr_mae': float(np.abs(hdr - original).mean())}

    @torch.inference_mode()
    def infer(self, frame: np.ndarray, channels: list[str]):
        started = time.perf_counter()
        indices = [channels.index(name) for name in ('linear_hdr_r', 'linear_hdr_g', 'linear_hdr_b')]
        rgb = np.ascontiguousarray(frame[..., indices])
        if not np.isfinite(rgb).all():
            raise ValueError('Nonfinite HDR input')
        a, b, c = self.events
        a.record()
        x = torch.from_numpy(rgb).to('cuda:0').permute(2, 0, 1).unsqueeze(0)
        x = x.clamp_min(0) * self.exposure
        x = x / (1 + x)  # Smooth Reinhard mapping; preserve HDR source separately.
        transform = self.spatial_transform(*rgb.shape[:2])
        left, top, width, height = transform['content_box_xywh']
        if x.shape[-2:] != (height, width):
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False, antialias=True)
        x = F.pad(x, (left, 256-width-left, top, 256-height-top), mode='replicate')
        x = x.to(dtype=self.runner.dtype).contiguous()
        b.record()
        y = self.runner(x)
        c.record()
        # Copies finish before the captured output buffer is reused.
        source = x[0].permute(1, 2, 0).float().contiguous().cpu().numpy()
        output = y[0].permute(1, 2, 0).float().contiguous().cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError('Nonfinite model output')
        display = []
        for linear in (source, output):
            encoded = linear_to_srgb(torch.from_numpy(linear))
            display.append(encoded.mul(255).round().to(torch.uint8).numpy())
        stats = {'model_gpu_ms': b.elapsed_time(c), 'preprocess_gpu_ms': a.elapsed_time(b),
                 'adapter_wall_ms': (time.perf_counter() - started) * 1000,
                 'linear_mae': float(np.abs(output - source).mean()),
                 'output_range': [float(output.min()), float(output.max())],
                 'hdr_max': float(rgb.max()), 'width': source.shape[1], 'height': source.shape[0],
                 'negative_native_channels': int(np.count_nonzero(rgb < 0)),
                 'spatial_transform': transform}
        return source, output, display, stats
