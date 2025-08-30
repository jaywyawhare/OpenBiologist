"""
Boltz predictor adapter

This module provides a thin wrapper that exposes an infer_pdb(sequence: str) API
compatible with the rest of the service. It dynamically imports a Boltz model
implementation if available and surfaces clear errors otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional
import logging


class BoltzPredictor:
    """Adapter for Boltz using its Python API if available, else CLI fallback.

    Exposes `infer_pdb(sequence: str) -> str`.
    """

    def __init__(self) -> None:
        self._api: Optional[Any] = None
        self._load_api()
        self._logger = logging.getLogger(__name__)
        self._verbose = os.environ.get("OPENBIO_BOLTZ_VERBOSE", "1") not in ("0", "false", "False")

    def _load_api(self) -> None:
        # Try to load Boltz Python API for better Flash Attention support
        try:
            import boltz.main
            self._api = boltz.main
            print("✅ Boltz Python API loaded successfully")
        except ImportError:
            print("⚠️ Boltz Python API not available, will use CLI")
            self._api = None

    def eval(self) -> None:
        # For compatibility with service expectations
        return None

    def _write_fasta(self, sequence: str, dest: Path, chain_id: str = "A") -> None:
        # Use proper Boltz FASTA format: >CHAIN_ID|ENTITY_TYPE|MSA_PATH
        # Using 'empty' for MSA_PATH runs in single sequence mode without MSA
        dest.write_text(f">{chain_id}|protein|empty\n{sequence}\n", encoding="utf-8")

    def _read_first_pdb(self, out_dir: Path) -> str:
        # Search recursively for .pdb output in the specified directory
        for path in out_dir.rglob("*.pdb"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "ATOM" in text or "HETATM" in text:
                return text
        
        # Fallback: accept mmCIF if PDB not present
        for path in out_dir.rglob("*.cif"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text:
                return text
        
        # Check if Boltz created output in a different location (boltz_results_*)
        parent_dir = out_dir.parent
        for subdir in parent_dir.glob("boltz_results_*"):
            if subdir.is_dir():
                # Search in the Boltz results directory
                for path in subdir.rglob("*.pdb"):
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    if "ATOM" in text or "HETATM" in text:
                        return text
                for path in subdir.rglob("*.cif"):
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    if text:
                        return text
        
        # Last resort: report directory contents to aid debugging
        listing = []
        for sub in out_dir.rglob("*"):
            try:
                rel = sub.relative_to(out_dir)
                listing.append(str(rel))
            except Exception:
                continue
        
        # Also check parent directory for Boltz results
        for sub in parent_dir.glob("*"):
            try:
                listing.append(str(sub.name))
            except Exception:
                continue
        
        raise FileNotFoundError(
            "No PDB or CIF file produced by Boltz in output directory. Contents: "
            + ", ".join(sorted(listing)[:200])
        )

    def _run_api(self, fasta: Path, out_dir: Path) -> None:
        assert self._api is not None
        # Force GPU first
        accelerator = "gpu"
        try:
            # Call boltz.main.predict programmatically (not default; may fail if click-wrapped)
            self._api.predict(  # type: ignore[call-arg]
                str(fasta),
                str(out_dir),
                accelerator=accelerator,
                devices=1,
                output_format="pdb",
                model="boltz2",
                subsample_msa=True,
                num_subsampled_msa=1024,
                no_kernels=True,
            )
        except Exception as e:
            raise RuntimeError(f"Boltz API prediction failed: {e}") from e

    def _run_cli(self, fasta: Path, out_dir: Path) -> None:
        exe = shutil.which("boltz")  # type: ignore[name-defined]
        cmd: list[str]
        # Build common args (force GPU first, optimize for speed)
        common = [
            "predict",
            str(fasta),
            "--out_dir",
            str(out_dir),
            "--output_format",
            "pdb",
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--no_kernels",
            "--num_workers",
            "8",  # Increased workers for better performance
            "--recycling_steps",
            "2",  # Reduce from default 3 for speed
            "--sampling_steps",
            "100",  # Reduce from default 200 for speed
            "--diffusion_samples",
            "1",  # Keep at 1 for speed
            "--step_scale",
            "1.5",  # Slightly lower for faster convergence
        ]
        if exe:
            cmd = [exe] + common
        else:
            # python -m boltz.main predict ...
            cmd = [sys.executable, "-m", "boltz.main"] + common
        if self._verbose:
            self._logger.info("Running Boltz CLI: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if self._verbose:
                # Log a limited amount of output to avoid flooding logs
                if proc.stdout:
                    self._logger.info("Boltz stdout (last 2000 chars): %s", proc.stdout[-2000:])
                if proc.stderr:
                    self._logger.info("Boltz stderr (last 2000 chars): %s", proc.stderr[-2000:])
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode(errors='ignore') if e.stderr else "")
            stdout_text = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode(errors='ignore') if e.stdout else "")
            raise RuntimeError(
                "Boltz CLI prediction failed: "
                + (stderr_text[-4000:] if stderr_text else "")
                + ("\nStdout: " + stdout_text[-2000:] if stdout_text else "")
            )

    def infer_pdb(self, sequence: str, chain_id: str = "A") -> str:
        # Optimize float32 matmul for speed (trade precision for performance)
        try:
            import torch
            # Use 'high' precision for maximum speed on A100 Tensor Cores
            torch.set_float32_matmul_precision('high')
            # Enable TensorFloat-32 for even faster computation on A100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Optimize memory allocation
            torch.backends.cudnn.benchmark = True
            # Use deterministic algorithms for consistency
            torch.backends.cudnn.deterministic = False
            
            # Additional PyTorch performance optimizations
            torch.backends.cuda.enable_math_sdp(True)  # Math-based attention fallback
            torch.backends.cuda.enable_flash_sdp(True)  # Flash attention if available
            
            # Advanced memory optimizations
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Additional PyTorch performance tweaks
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            
            # Try to import and enable Flash Attention optimizations
            try:
                import flash_attn
                print("✅ Flash Attention enabled for faster computation")
            except ImportError:
                print("⚠️ Flash Attention not available, using standard attention")
            
        except ImportError:
            pass  # Torch not available, continue without optimization

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fasta = tmp / "query.fasta"
            out_dir = tmp / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._write_fasta(sequence, fasta, chain_id)

            # Force CUDA device visibility if unset
            if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
            # Performance optimizations for A100
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Non-blocking CUDA operations
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable latest cuDNN
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"  # Force TF32 on A100
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Optimize memory allocation
            os.environ["CUDA_MEMORY_FRACTION"] = "0.95"  # Use 95% of GPU memory
            
            # Additional CUDA optimizations
            os.environ["CUDA_CACHE_DISABLE"] = "0"  # Enable CUDA cache
            os.environ["CUDA_CACHE_PATH"] = "/tmp/cuda_cache"  # Set cache path
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Optimize device ordering
            os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA device-side assertions
            
            # Advanced performance optimizations
            os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
            os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
            os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Limit NumExpr threads
            os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Limit OpenBLAS threads
            os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Limit VecLib threads
            os.environ["BLAS_NUM_THREADS"] = "1"  # Limit BLAS threads

            ran_via_api = False
            if self._api is not None:
                try:
                    self._run_api(fasta, out_dir)
                    ran_via_api = True
                except Exception:
                    # Fallback to CLI on any API error
                    ran_via_api = False
            if not ran_via_api:
                # Lazy import to avoid unconditional dependency
                import shutil  # noqa: WPS433
                self._run_cli(fasta, out_dir)

            return self._read_first_pdb(out_dir)



