from typing import Literal, Optional
from pydantic import BaseModel


class QASPERExample(BaseModel):
    """
    A single processed QASPER question-answer pair with quality flags.

    One QASPERExample corresponds to one question against one paper. Where
    multiple annotators have answered the same question, a single answer is
    selected by type priority (free_form > extractive > yes_no) from the
    agreeing majority. Quality flags are retained so that downstream consumers
    can apply their own filtering policy without reprocessing the dataset.
    """
    
    qasper_id: str
    question_id: str
    question_text: str
    
    # None when is_unanswerable is True
    answer_type: Optional[Literal["extractive", "free_form", "yes_no"]] = None
    
    raw_answer: str
    abstract: str
    evidence: list[str]
    has_float_evidence: bool
    is_unanswerable: bool
    is_cleanable: bool
    annotator_count: int
    annotators_agree_answersability: bool
    split: Literal["train", "validation", "test"]

    nlp_background: Optional[str] = None
    topic_background: Optional[str] = None
    paper_read: Optional[bool] = None
    search_query: Optional[str] = None


class QASData(BaseModel):
    question_id: str
    question_text: str
    
    # None when is_unanswerable is True
    answer_type: Optional[Literal["extractive", "free_form", "yes_no"]] = None

    raw_answer: str
    evidence: list[str]
    has_float_evidence: bool
    is_unanswerable: bool
    is_cleanable: bool
    annotator_count: int
    annotators_agree_answersability: bool

    nlp_background: Optional[str] = None
    topic_background: Optional[str] = None
    paper_read: Optional[bool] = None
    search_query: Optional[str] = None
