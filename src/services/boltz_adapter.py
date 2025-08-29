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
        # Prefer CLI for stability; API path is click-wrapped and brittle.
        self._api = None

    def eval(self) -> None:
        # For compatibility with service expectations
        return None

    def _write_fasta(self, sequence: str, dest: Path) -> None:
        dest.write_text(f">query\n{sequence}\n", encoding="utf-8")

    def _read_first_pdb(self, out_dir: Path) -> str:
        # Search recursively for .pdb output
        for path in out_dir.rglob("*.pdb"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "ATOM" in text or "HETATM" in text:
                return text
        # Fallback: accept mmCIF if PDB not present
        for path in out_dir.rglob("*.cif"):
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
        # Build common args (force GPU first, disable kernels for speed)
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

    def infer_pdb(self, sequence: str) -> str:
        # Optimize float32 matmul for speed (trade precision for performance)
        try:
            import torch
            torch.set_float32_matmul_precision('high')
        except ImportError:
            pass  # Torch not available, continue without optimization

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fasta = tmp / "query.fasta"
            out_dir = tmp / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._write_fasta(sequence, fasta)

            # Force CUDA device visibility if unset
            if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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



