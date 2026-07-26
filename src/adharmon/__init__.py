"""Harmonize multi-source Alzheimer's disease sample metadata on Azure.

Ingests sample metadata from several AD transcriptomics studies with mutually
incompatible annotation conventions, maps them onto one canonical schema,
validates quality at the ingestion boundary, and lands the result in an ADLS
Gen2 medallion layout with a lineage manifest per run.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
