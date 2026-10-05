"""Load the actual recovered network; never fall back to random/identity weights."""
from __future__ import annotations
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = ROOT / "vendor" / "dlss5-onnx" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))
from dlss5.graph import DLSS5Graph
from dlss5.static import DLSS5StaticImage

FORMAT = "dlss5_recovered_static_pytorch_v1"


def load_model(checkpoint: Path, device: torch.device, precision: str) -> tuple[nn.Module, dict]:
    if precision not in {"fp16", "fp32", "reference"}:
        raise ValueError("precision must be fp16, fp32 or reference")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint format")
    graph = DLSS5Graph().enable_fp8_emulation()
    model = DLSS5StaticImage(graph, portable_reductions=precision == "reference")
    model.load_state_dict(data["state_dict"], strict=True)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    return model, data["metadata"]


class ModelRunner:
    """Optional fixed-shape CUDA graph removes per-frame Python launch overhead.

    All instances are single-consumer; do not submit concurrent streams/threads.
    Captured outputs are borrowed until the next call, so the pipeline copies them.
    """
    def __init__(self, model: nn.Module, use_cuda_graph: bool = False):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.graph = None
        self.capture_error = None
        if use_cuda_graph:
            if self.device.type != "cuda":
                raise ValueError("CUDA graph requires a CUDA device")
            self._capture()

    @torch.inference_mode()
    def _capture(self) -> None:
        self.static_input = torch.zeros((1, 3, 256, 256), device=self.device, dtype=self.dtype)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model(self.static_input)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        # No silent fallback: callers need to know whether capture actually works.
        with torch.cuda.graph(graph):
            self.static_output = self.model(self.static_input)
        self.graph = graph

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            return self.model(rgb)
        self.static_input.copy_(rgb)
        self.graph.replay()
        return self.static_output
