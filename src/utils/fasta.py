"""
FASTA utilities: verify and fix protein sequences/FASTA text.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def sanitize_sequence(raw: str) -> Tuple[str, List[str]]:
    """Clean a raw sequence string: uppercase, strip whitespace, remove invalid chars.

    Returns a tuple of (clean_sequence, warnings).
    """
    warnings: List[str] = []
    if not raw:
        return "", ["Empty input sequence"]
    seq = "".join(ch for ch in raw.upper() if ch.isalpha())
    invalid = sorted(set(ch for ch in seq if ch not in STANDARD_AA))
    if invalid:
        warnings.append(f"Removed invalid amino acids: {''.join(invalid)}")
        seq = "".join(ch for ch in seq if ch in STANDARD_AA)
    return seq, warnings


def parse_and_fix_fasta(text: str) -> Tuple[str, str, List[str]]:
    """Parse input as FASTA or raw sequence and produce a valid FASTA.

    Returns (header, fixed_sequence, warnings).
    """
    warnings: List[str] = []
    header = "A"  # Default chain ID for raw sequences
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return header, "", ["Empty input"]

    if lines[0].startswith(">"):
        # Parse FASTA header: >CHAIN_ID|ENTITY_TYPE|MSA_PATH
        header_line = lines[0][1:].strip()
        if "|" in header_line:
            parts = header_line.split("|")
            if len(parts) >= 2:
                chain_id = parts[0].strip()
                entity_type = parts[1].strip()
                if entity_type == "protein":
                    header = chain_id
                    if len(parts) > 2 and parts[2].strip() != "empty":
                        warnings.append(f"MSA path '{parts[2]}' ignored (running in single sequence mode)")
                else:
                    warnings.append(f"Entity type '{entity_type}' not yet supported, treating as protein")
                    header = chain_id
            else:
                header = header_line
        else:
            header = header_line
        seq_raw = "".join(lines[1:])
    else:
        # No header - treat as raw sequence
        seq_raw = "".join(lines)
        warnings.append("No FASTA header found, treating as raw protein sequence")

    seq, ws = sanitize_sequence(seq_raw)
    warnings.extend(ws)
    if len(seq) == 0:
        warnings.append("Sequence became empty after sanitization")
    return header, seq, warnings


def format_fasta(header: str, sequence: str, width: int = 60) -> str:
    """Format a sequence into FASTA string with fixed line width."""
    lines = [f">{header}"]
    for i in range(0, len(sequence), width):
        lines.append(sequence[i : i + width])
    return "\n".join(lines) + "\n"

def format_boltz_fasta(chain_id: str, sequence: str, entity_type: str = "protein", msa_path: str = "empty", width: int = 60) -> str:
    """Format a sequence into Boltz FASTA format: >CHAIN_ID|ENTITY_TYPE|MSA_PATH"""
    header = f"{chain_id}|{entity_type}|{msa_path}"
    lines = [f">{header}"]
    for i in range(0, len(sequence), width):
        lines.append(sequence[i : i + width])
    return "\n".join(lines) + "\n"


def validate_sequence_minmax(sequence: str, min_len: int = 10, max_len: int = 2000) -> Dict[str, object]:
    seq, warnings = sanitize_sequence(sequence)
    if not seq:
        return {"valid": False, "error": "Empty or invalid sequence", "warnings": warnings}
    if len(seq) < min_len:
        return {"valid": False, "error": f"Sequence too short (minimum {min_len})", "warnings": warnings}
    if len(seq) > max_len:
        return {"valid": False, "error": f"Sequence too long (maximum {max_len})", "warnings": warnings}
    return {"valid": True, "sequence": seq, "warnings": warnings}


