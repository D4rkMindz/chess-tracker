"""Vision package: from photos to an Observation (occupancy and piece colour) per photo."""
from chess_tracker.vision.align import AlignmentError
from chess_tracker.vision.pipeline import calibrate, observe

__all__ = ["AlignmentError", "calibrate", "observe"]
