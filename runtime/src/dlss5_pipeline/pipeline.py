"""GPU tensor pipeline around the recovered fixed 256x256 image model.

Engine interop is deliberately outside this class. Pass an already-owned CUDA
NCHW tensor and receive a CUDA tensor; there are no CPU image round trips here.
Native DLSS5 temporal resource bindings have NOT been reconstructed.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch import Tensor
from torch.nn import functional as F

from .color import srgb_to_linear, linear_to_srgb
from .model import ROOT, ModelRunner, load_model


@dataclass(frozen=True)
class PipelineConfig:
    checkpoint: Path = ROOT / "assets" / "dlss5_static.pt"
    device: str = "cuda:0"
    precision: str = "fp16"
    cuda_graph: bool = False
    strength: float = 1.0
    history_weight: float = 0.0
    history_rejection: float = 0.04

    def __post_init__(self):
        if not 0 <= self.strength <= 1 or not 0 <= self.history_weight < 1:
            raise ValueError("strength must be [0,1], history_weight must be [0,1)")
        if self.history_rejection <= 0:
            raise ValueError("history_rejection must be positive")


@dataclass
class FrameInput:
    color: Tensor  # [1,3,H,W], normalized linear-LDR RGB or sRGB, already on device
    frame_id: int
    color_space: str = "linear"
    roi: tuple[int, int, int, int] | None = None  # x,y,width,height; exactly 256x256
    motion: Tensor | None = None  # [1,2,H,W], current->previous pixels, x right/y down
    history_valid: Tensor | None = None  # [1,1,H,W], 0 rejects history/occlusion
    control_mask: Tensor | None = None  # [1,1,H,W], 0 preserves source
    reset_history: bool = False


@dataclass
class FrameOutput:
    color: Tensor
    frame_id: int
    roi: tuple[int, int, int, int]
    history_applied: bool
    gpu_ms: float | None
    host_ms: float


class TorchRenderPipeline:
    """Single-consumer pipeline; caller supplies stream/resource synchronization.

    Only the selected 256x256 ROI is processed; pixels outside it are copied.
    Optional external history reprojection is experimental, not native DLSS5.
    CUDA graph capture covers the model, not dynamic ROI/history/compositing.
    """
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA PyTorch is required; no CPU fallback")
            torch.cuda.set_device(self.device)
        model, self.metadata = load_model(Path(config.checkpoint), self.device, config.precision)
        self.runner = ModelRunner(model, use_cuda_graph=config.cuda_graph)
        self.dtype = self.runner.dtype
        self._history_source = self._history_output = None
        self._history_key = None
        self._last_frame_id = None
        y, x = torch.meshgrid(torch.arange(256, device=self.device), torch.arange(256, device=self.device), indexing="ij")
        self._grid = torch.stack((x, y), dim=-1).float()[None]

    def reset(self) -> None:
        self._history_source = self._history_output = None
        self._history_key = self._last_frame_id = None

    def reload_checkpoint(self, checkpoint: str | Path) -> None:
        """Call between frames; a failed load leaves the old model intact."""
        model, metadata = load_model(Path(checkpoint), self.device, self.config.precision)
        runner = ModelRunner(model, use_cuda_graph=self.config.cuda_graph)
        self.runner, self.metadata, self.dtype = runner, metadata, runner.dtype
        self.reset()

    def _validate(self, frame: FrameInput) -> tuple[int, int, int, int]:
        x = frame.color
        if x.ndim != 4 or x.shape[:2] != (1, 3) or not x.is_floating_point():
            raise ValueError("color must be floating-point [1,3,H,W]")
        # torch.device('cuda') has no index; normalize through an actual model parameter.
        if x.device != self.runner.device:
            raise ValueError(f"Input must be on {self.runner.device}; received {x.device}")
        if frame.color_space not in {"linear", "srgb"}:
            raise ValueError("color_space must be linear or srgb; HDR requires engine tonemapping")
        h, w = x.shape[-2:]
        roi = frame.roi or ((w - 256) // 2, (h - 256) // 2, 256, 256)
        if any(type(v) is not int for v in roi) or len(roi) != 4:
            raise ValueError("roi must contain four integers")
        left, top, rw, rh = roi
        if (rw, rh) != (256, 256) or left < 0 or top < 0 or left + rw > w or top + rh > h:
            raise ValueError("ROI must be an in-bounds 256x256 region")
        for name, channels in (("motion", 2), ("history_valid", 1), ("control_mask", 1)):
            value = getattr(frame, name)
            if value is not None and (value.shape != (1, channels, h, w) or value.device != x.device or not value.is_floating_point()):
                raise ValueError(f"{name} must be floating-point [1,{channels},H,W] on the color device")
        return roi

    def _reproject(self, source: Tensor, output: Tensor, motion: Tensor, valid: Tensor | None) -> Tensor:
        # align_corners=False: normalized centers are (pixel + .5) / size * 2 - 1.
        coords = self._grid + motion.permute(0, 2, 3, 1).float()
        finite = torch.isfinite(coords).all(-1)
        in_bounds = finite & (coords[..., 0] >= 0) & (coords[..., 0] <= 255) & (coords[..., 1] >= 0) & (coords[..., 1] <= 255)
        grid = (torch.nan_to_num(coords, nan=-1024., posinf=-1024., neginf=-1024.) + 0.5) / 256 * 2 - 1
        history_source = F.grid_sample(self._history_source.float(), grid, align_corners=False)
        history_output = F.grid_sample(self._history_output.float(), grid, align_corners=False)
        delta = (history_source - source.float()).abs().mean(1, keepdim=True)
        weight = in_bounds[:, None].float() * (1 - delta / self.config.history_rejection).clamp(0, 1)
        if valid is not None:
            weight = weight * torch.nan_to_num(valid.float(), nan=0.).clamp(0, 1)
        weight = weight * self.config.history_weight
        return output.float() * (1 - weight) + history_output * weight

    @torch.inference_mode()
    def render(self, frame: FrameInput, *, measure: bool = True) -> FrameOutput:
        started = time.perf_counter()
        left, top, rw, rh = roi = self._validate(frame)
        key = (tuple(frame.color.shape), roi, frame.color_space)
        # A dropped/nonconsecutive frame invalidates one-frame motion vectors.
        if frame.reset_history or key != self._history_key or (self._last_frame_id is not None and frame.frame_id != self._last_frame_id + 1):
            self.reset()
        begin = end = None
        if measure and self.device.type == "cuda":
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        crop = frame.color[..., top:top + rh, left:left + rw]
        source = srgb_to_linear(crop.float()) if frame.color_space == "srgb" else crop.float()
        prediction = self.runner(source.to(self.dtype).contiguous()).float()
        history_applied = self.config.history_weight > 0 and self._history_output is not None and frame.motion is not None
        if history_applied:
            motion = frame.motion[..., top:top + rh, left:left + rw]
            valid = frame.history_valid[..., top:top + rh, left:left + rw] if frame.history_valid is not None else None
            prediction = self._reproject(source, prediction, motion, valid)
        mask = self.config.strength
        if frame.control_mask is not None:
            mask = frame.control_mask[..., top:top + rh, left:left + rw].clamp(0, 1) * mask
        prediction = source + mask * (prediction - source)
        # Store pre-mask neural output? Store composited output to keep control regions stable.
        if self.config.history_weight > 0:
            self._history_source = source.detach().clone()
            self._history_output = prediction.detach().clone()
        self._history_key, self._last_frame_id = key, frame.frame_id
        display = linear_to_srgb(prediction) if frame.color_space == "srgb" else prediction.clamp(0, 1)
        result = frame.color.clone()
        result[..., top:top + rh, left:left + rw] = display.to(result.dtype)
        gpu_ms = None
        if end is not None:
            end.record()
            end.synchronize()
            gpu_ms = begin.elapsed_time(end)
        return FrameOutput(result, frame.frame_id, roi, history_applied, gpu_ms, (time.perf_counter() - started) * 1000)
