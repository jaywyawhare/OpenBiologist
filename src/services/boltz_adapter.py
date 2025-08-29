"""
Boltz predictor adapter

This module provides a thin wrapper that exposes an infer_pdb(sequence: str) API
compatible with the rest of the service. It dynamically imports a Boltz model
implementation if available and surfaces clear errors otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


class BoltzPredictor:
    """Adapter for Boltz using its Python API if available, else CLI fallback.

    Exposes `infer_pdb(sequence: str) -> str`.
    """

    def __init__(self) -> None:
        self._api: Optional[Any] = None
        self._load_api()

    def _load_api(self) -> None:
        # Try programmatic API via boltz.main.predict
        try:
            import boltz.main as boltz_main  # type: ignore

            if hasattr(boltz_main, "predict"):
                self._api = boltz_main
                return
        except Exception:
            # Will fall back to CLI
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
        raise FileNotFoundError("No PDB file produced by Boltz in output directory")

    def _run_api(self, fasta: Path, out_dir: Path) -> None:
        assert self._api is not None
        # Choose accelerator based on CUDA presence
        accelerator = "gpu" if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None) else "cpu"
        try:
            # Call boltz.main.predict programmatically
            self._api.predict(
                data=str(fasta),
                out_dir=str(out_dir),
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
        if exe:
            cmd = [
                exe,
                "predict",
                str(fasta),
                "--out_dir",
                str(out_dir),
                "--output_format",
                "pdb",
            ]
        else:
            # python -m boltz.main predict ...
            cmd = [
                sys.executable,
                "-m",
                "boltz.main",
                "predict",
                str(fasta),
                "--out_dir",
                str(out_dir),
                "--output_format",
                "pdb",
            ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Boltz CLI prediction failed: {e.stderr.decode(errors='ignore')}")

    def infer_pdb(self, sequence: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fasta = tmp / "query.fasta"
            out_dir = tmp / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._write_fasta(sequence, fasta)

            if self._api is not None:
                self._run_api(fasta, out_dir)
            else:
                # Lazy import to avoid unconditional dependency
                import shutil  # noqa: WPS433

                self._run_cli(fasta, out_dir)

            return self._read_first_pdb(out_dir)



