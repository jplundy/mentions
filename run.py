"""Command line interface for the historical transcript data pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

from pipeline import (
    DatasetPublisher,
    PDFIngestor,
    PipelineConfig,
    Segment,
    Segmenter,
    SpeakerFilter,
    TranscriptInventory,
    TranscriptPreprocessor,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("mentions.pipeline")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the historical transcript dataset")
    parser.add_argument("config", type=Path, help="Path to pipeline configuration YAML/JSON file")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Optional dataset version identifier (default: UTC timestamp)",
    )
    parser.add_argument(
        "--backend",
        choices=["pdfplumber", "pypdf"],
        default="pdfplumber",
        help="Preferred PDF extraction backend",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    config = PipelineConfig.from_file(args.config)
    config.ensure_directories()

    inventory = TranscriptInventory.from_config(config)
    ingestor = PDFIngestor(prefer_pdfplumber=args.backend == "pdfplumber")
    preprocessor = TranscriptPreprocessor()
    segmenter = Segmenter(config.segmentation)
    speaker_filter = None
    if config.speaker_filter:
        speaker_filter = SpeakerFilter.from_terms(
            include=config.speaker_filter.include,
            exclude=config.speaker_filter.exclude,
        )

    publisher = DatasetPublisher(
        inventory=inventory,
        output_dir=config.output_directory,
        target_words=config.iter_target_words(),
        speaker_filter=speaker_filter,
    )

    segmented_records: Dict[str, List[Segment]] = {}
    for record in inventory:
        logger.info("Processing %s", record.event_id)
        pages = ingestor.extract_text(record.pdf_path)
        cleaned_text = preprocessor.preprocess(pages)
        segments = segmenter.segment(record.event_id, cleaned_text)
        segmented_records[record.event_id] = segments

    rows = publisher.build_rows(segmented_records)
    validation = publisher.validate(rows)
    dataset_path = publisher.publish(rows, version=args.version, validation=validation)
    logger.info("Validation summary: %s", validation.as_dict())
    return dataset_path


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
