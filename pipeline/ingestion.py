"""Utilities for extracting text from PDF transcripts."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional

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

    def __init__(
        self,
        prefer_pdfplumber: bool = True,
        max_workers: Optional[int] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.prefer_pdfplumber = prefer_pdfplumber
        self.max_workers = max_workers
        self.cache_dir = cache_dir

    def extract_text(self, path: Path) -> List[str]:
        if self.cache_dir is not None:
            cached = self._load_from_cache(path)
            if cached is not None:
                logger.debug("Loaded %s pages from cache for %s", len(cached), path)
                return cached
        pages = self._do_extract(path)
        if self.cache_dir is not None:
            self._save_to_cache(path, pages)
        return pages

    def _do_extract(self, path: Path) -> List[str]:
        if self.prefer_pdfplumber and pdfplumber is not None:
            return self._extract_with_pdfplumber(path)
        if PdfReader is not None:
            return self._extract_with_pypdf(path)
        if pdfplumber is not None:
            return self._extract_with_pdfplumber(path)
        raise RuntimeError("No PDF extraction backend available. Please install pdfplumber or pypdf.")

    def _cache_path(self, path: Path) -> Path:
        stat = path.stat()
        key = f"{path.stem}_{stat.st_mtime_ns}_{stat.st_size}"
        return self.cache_dir / f"{key}.json"  # type: ignore[operator]

    def _load_from_cache(self, path: Path) -> Optional[List[str]]:
        try:
            cache_file = self._cache_path(path)
        except OSError:
            return None
        if not cache_file.exists():
            return None
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_to_cache(self, path: Path, pages: List[str]) -> None:
        assert self.cache_dir is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_path(path)
        try:
            cache_file.write_text(json.dumps(pages), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write PDF text cache for %s: %s", path, exc)

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
        paths_list = list(paths)
        if len(paths_list) <= 1 or self.max_workers == 1:
            return [self.extract_text(p) for p in paths_list]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.extract_text, paths_list))
