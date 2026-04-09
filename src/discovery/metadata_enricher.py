from typing import Optional
from semanticscholar import SemanticScholar
from models import PaperMetadata


class MetadataEnricher:
    """Handles external API calls to enhance the base ACL metadata."""
    
    def __init__(self, s2_api_key: Optional[str] = None):
        self.s2_client = SemanticScholar(api_key=s2_api_key)

    def add_citation_metrics(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        """Batch-fetches S2 metrics and updates the models."""
        pass

    def scrape_missing_abstracts(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        """Performs BeautifulSoup scraping for papers without abstracts."""
        pass
