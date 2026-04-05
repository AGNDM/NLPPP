from pydantic import BaseModel
from typing import List, Optional

class PaperMetadata(BaseModel):
    anthology_id: str
    title: str
    authors: List[str]
    abstract: Optional[str] = None
    citation_count: int = 0
    pdf_url: str
    local_pdf_path: Optional[str] = None
    
    
class ProcessedDocument(BaseModel):
    metadata: PaperMetadata
    markdown_content: str  
    # chunks: List[str]    