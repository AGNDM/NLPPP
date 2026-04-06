from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from typing import Literal


class PaperMetadata(BaseModel):
    """
    Descriptive metadata for a single ACL Anthology paper.

    Populated during metadata fetch and treated as immutable thereafter.
    """
    anthology_id: str
    arxiv_id: Optional[str] = None 
    """arxiv_id obtained from Semantic Scholar."""
    title: str
    authors: List[str]
    abstract: Optional[str] = None
    citation_count: Optional[int] = None
    """Total citation count from Semantic Scholar."""
    pdf_url: Optional[str] = None
    """
    URL of the paper PDF on aclanthology.org. 
    None for papers where no PDF is available."""


class StagingStatus(str, Enum):
    """Pipeline processing state for a single paper."""
    pending = "pending"
    downloaded = "downloaded"
    converted = "converted"
    failed = "failed"


class StagingRecord(BaseModel):
    """PIpeline operational state for a single paper."""
    anthology_id: str
    tier: Optional[Literal["1", "2", "3"]] = None
    """
    Determines depth of processing:
        1: abstract only (no staging required)
        2: abstract + introduction + conclusion
        3: full section-aware chunks
    None if tier has not yet been assigned.
    """
    tier_source: Optional[Literal["auto", "manual"]] = None
    staging_status: StagingStatus = StagingStatus.pending
    local_pdf_path: Optional[str] = None
    local_md_path: Optional[str] = None
    error: Optional[str] = None  # populated on failure
    notes: Optional[str] = None


class Chunk(BaseModel):
    """Placeholder TBD"""
    chunk_id: str
    anthology_id: str
    tier: Literal["1", "2", "3"]
    section: Optional[str] = None
    chunk_index: int
    text: str
    token_count: int
