"""Utilities for extracting text from PDF transcripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

logger = logging.getLogger(__name__)


class PDFIngestor:
    """Extract raw text from PDF files with fallback strategies."""

    def __init__(self, prefer_pdfplumber: bool = True):
        self.prefer_pdfplumber = prefer_pdfplumber

    def extract_text(self, path: Path) -> List[str]:
        if self.prefer_pdfplumber and pdfplumber is not None:
            return self._extract_with_pdfplumber(path)
        if PdfReader is not None:
            return self._extract_with_pypdf(path)
        if pdfplumber is not None:
            return self._extract_with_pdfplumber(path)
        raise RuntimeError("No PDF extraction backend available. Please install pdfplumber or pypdf.")

    def _extract_with_pdfplumber(self, path: Path) -> List[str]:
        pages: List[str] = []
        if pdfplumber is None:  # pragma: no cover - sanity check
            raise RuntimeError("pdfplumber is not installed")
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                pages.append(text)
        logger.debug("Extracted %s pages from %s using pdfplumber", len(pages), path)
        return pages

    def _extract_with_pypdf(self, path: Path) -> List[str]:
        if PdfReader is None:  # pragma: no cover - sanity check
            raise RuntimeError("pypdf is not installed")
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        logger.debug("Extracted %s pages from %s using pypdf", len(pages), path)
        return pages

    def batch_extract(self, paths: Iterable[Path]) -> List[List[str]]:
        return [self.extract_text(path) for path in paths]
