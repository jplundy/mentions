from datetime import datetime
from pathlib import Path

import pytest

from pipeline.dataset import DatasetPublisher
from pipeline.inventory import TranscriptInventory, TranscriptRecord
from pipeline.segmentation import Segment
from pipeline.speakers import SpeakerFilter


@pytest.fixture
def transcript_record():
    return TranscriptRecord(
        event_id="EVT1",
        event_type="fomc_press_conference",
        event_date=datetime(2023, 1, 1),
        speakers=["Jerome Powell", "Reporter"],
        pdf_path=Path("dummy.pdf"),
    )


def test_speaker_filter_basic_matching():
    flt = SpeakerFilter.from_terms(include=["powell"])
    assert flt.allows("CHAIR POWELL")
    assert not flt.allows("REPORTER 1")
    assert not flt.allows(None)


def test_speaker_filter_with_exclude():
    flt = SpeakerFilter.from_terms(include=[], exclude=["reporter"])
    assert flt.allows("CHAIR POWELL")
    assert not flt.allows("FEDERAL RESERVE REPORTER")


def test_dataset_publisher_applies_filter(transcript_record):
    inventory = TranscriptInventory([transcript_record])
    segments = [
        Segment(
            event_id="EVT1",
            segment_id="EVT1_S0001",
            text="Welcome everyone",
            speaker="CHAIR POWELL",
            start_char=0,
            end_char=17,
        ),
        Segment(
            event_id="EVT1",
            segment_id="EVT1_S0002",
            text="Thank you, Chair Powell",
            speaker="REPORTER",
            start_char=18,
            end_char=40,
        ),
    ]

    publisher = DatasetPublisher(
        inventory=inventory,
        output_dir=Path("/tmp"),
        target_words=["welcome"],
        speaker_filter=SpeakerFilter.from_terms(include=["powell"]),
    )

    rows = publisher._segment_rows(transcript_record, segments)
    assert len(rows) == 1
    assert rows[0]["speaker"] == "CHAIR POWELL"
    assert rows[0]["target__welcome"] is True
