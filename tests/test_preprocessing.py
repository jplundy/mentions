from pipeline.preprocessing import TranscriptPreprocessor
from dataclasses import dataclass

from pipeline.segmentation import Segmenter


@dataclass
class _DummySegmentationConfig:
    mode: str = "speaker_turn"
    window_size: int | None = None
    stride: int | None = None


def test_preprocess_normalizes_title_case_labels():
    preprocessor = TranscriptPreprocessor()
    pages = ["Mr. Powell: Good afternoon everyone\nReporter: Thank you, Chair Powell"]
    processed = preprocessor.preprocess(pages)
    assert "MR POWELL: Good afternoon everyone" in processed
    assert "REPORTER: Thank you, Chair Powell" in processed


def test_segmenter_handles_title_case_labels():
    preprocessor = TranscriptPreprocessor()
    pages = [
        "Mr. Powell: Good afternoon everyone\nReporter: Thank you, Chair Powell\nMr. Powell: You're welcome",
    ]
    processed = preprocessor.preprocess(pages)
    segmenter = Segmenter(_DummySegmentationConfig(mode="speaker_turn"))
    segments = segmenter.segment("EVT1", processed)
    speakers = [segment.speaker for segment in segments]
    assert speakers[0] == "MR POWELL"
    assert speakers[1] == "REPORTER"
    assert speakers[2] == "MR POWELL"


def test_preprocess_converts_period_speaker_labels_to_colons():
    preprocessor = TranscriptPreprocessor()
    pages = [
        "Chair Powell. Good afternoon everyone\nMr. Powell. Thank you",
    ]
    processed = preprocessor.preprocess(pages)
    assert "CHAIR POWELL:" in processed
    assert "MR POWELL:" in processed


def test_segmenter_detects_labels_after_period_normalization():
    preprocessor = TranscriptPreprocessor()
    pages = [
        "Chair Powell. Good afternoon everyone\nChris Rugaber. Thanks",
    ]
    processed = preprocessor.preprocess(pages)
    segmenter = Segmenter(_DummySegmentationConfig(mode="speaker_turn"))
    segments = segmenter.segment("EVT2", processed)
    speakers = [segment.speaker for segment in segments]
    assert speakers == ["CHAIR POWELL", "CHRIS RUGABER"]
