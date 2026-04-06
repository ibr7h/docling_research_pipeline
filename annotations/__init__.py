"""Annotation layer for scientific PDF workflows."""

from annotations.hooks import annotation_context_for_prompt, attach_annotations_to_results
from annotations.models import Annotation
from annotations.store import AnnotationStore

__all__ = [
    "Annotation",
    "AnnotationStore",
    "attach_annotations_to_results",
    "annotation_context_for_prompt",
]
