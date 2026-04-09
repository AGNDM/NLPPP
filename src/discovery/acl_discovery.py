from typing import list, dict
from acl_anthology import Anthology

class ACLDiscovery:
    def __init__(self):
        """
        Initialises the Anthology data. 
        The first run will download/sync the metadata repository (~120MB).
        """
        self.anthology = Anthology.from_repo()
        # This stores the "Master list" of all discovered papers
        self.master_paper_list: list[dict] = []

    def load_event_papers(self, venue: str, year: int):
        """
        Populates self.master_paper_list with every paper from a specific event.
        Does not filter; simply builds the complete local registry for that venue/year.
        """
        pass

    def get_filtered_papers(self, keywords: list[str] = None, only_awards: bool = False) -> list[dict]:
        """
        Returns a subset of self.master_paper_list based on provided criteria.
        Crucially, this does NOT modify the master list or require new ACL calls.
        """
        pass

    def save_master_list(self, file_path: str = "acl_catalog.json"):
        """
        Serialises the complete master_paper_list to disk for persistence.
        """
        pass