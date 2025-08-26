import json
import logging
import time
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import requests
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("paper_fetcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    paper_id: str
    title: str = ""
    abstract: str = ""
    citations: List[str] = field(
        default_factory=list
    )  # List of paper IDs that this paper cites
    embedding: Optional[np.ndarray] = None
    publication_date: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[str] = None  # Changed to string for consistency
    citation_count: int = 0
    novel: Optional[str] = None
    authors: str = ""  # Changed to string for consistency with retrieval.py
    cited_paper: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert Paper object to dictionary, handling numpy arrays."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist() if value is not None else None
            else:
                result[key] = value
        return result


class SemanticScholarAPI:
    """Class to handle Semantic Scholar API interactions with rate limiting and error handling."""

    def __init__(self, api_key: str = None, rate_limit_delay: float = 1.0):
        if not api_key:
            api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        
        if not api_key:
            logger.warning("No Semantic Scholar API key provided. Using unauthenticated requests (lower rate limits).")
            
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
        self.headers = {"X-API-KEY": api_key} if api_key and api_key != "---" else {}
        self.last_request_time = 0

        logger.info(
            f"Initialized Semantic Scholar API client with rate limit: {rate_limit_delay}s"
        )

    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, requests.exceptions.Timeout)
        ),
    )
    def _make_request(self, params: Dict[str, Any]) -> requests.Response:
        """Make API request with retry logic."""
        self._rate_limit()

        try:
            response = requests.get(
                self.base_url, params=params, headers=self.headers, timeout=30
            )
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}")
            raise

    def get_paper_by_title(self, title: str) -> Optional[Paper]:
        """
        Fetch a paper from Semantic Scholar by title and return a Paper object.

        Args:
            title (str): Title of the paper to search for.

        Returns:
            Optional[Paper]: A Paper object with relevant details or None if not found.
        """
        if not title or not title.strip():
            logger.warning("Empty title provided")
            return None

        # Clean title
        title = title.strip()
        logger.debug(f"Searching for paper: '{title}'")

        params = {
            "query": title,
            "fields": "title,abstract,paperId,publicationDate,venue,year,citationCount,authors",
        }

        try:
            response = self._make_request(params)

            if response.status_code == 429:
                logger.warning("Rate limit exceeded, backing off...")
                time.sleep(10)  # Longer backoff for rate limiting
                response = self._make_request(params)

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.text}"
                )
                return None

            data = response.json()

            if "data" not in data or not data["data"]:
                logger.warning(f"No matching paper found for: '{title}'")
                return None

            paper_data = data["data"][0]  # Get the first result

            # Extract authors properly and convert to string
            authors = ""
            if "authors" in paper_data and paper_data["authors"]:
                author_names = [author.get("name", "") for author in paper_data["authors"]]
                authors = ", ".join(author_names)

            paper_obj = Paper(
                paper_id=paper_data.get("paperId", "Unknown ID"),
                title=paper_data.get("title", "Unknown Title"),
                abstract=paper_data.get("abstract", "No abstract available"),
                publication_date=paper_data.get("publicationDate"),
                venue=paper_data.get("venue"),
                year=str(paper_data.get("year")) if paper_data.get("year") is not None else None,
                citation_count=paper_data.get("citationCount", 0),
                authors=authors,
            )

            logger.info(f"✅ Found paper: '{paper_obj.title}' ({paper_obj.year})")
            return paper_obj

        except Exception as e:
            logger.error(f"Error fetching paper '{title}': {e}")
            return None


def process_for_pipeline(data_dir: str, submission_id: str, api_key: str = None, rate_limit_delay: float = 1.0):
    """
    Process a single submission for pipeline integration.
    
    Args:
        data_dir: Base data directory for pipeline
        submission_id: ID of the submission
        api_key: Semantic Scholar API key (optional)
        rate_limit_delay: Delay between API requests in seconds
    
    Returns:
        Processing statistics
    """
    # Input file path 
    input_file = Path(data_dir) / submission_id / f"{submission_id}.json"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load single submission data
    with open(input_file, "r", encoding="utf-8") as f:
        submission_data = json.load(f)
    
    # Initialize API client
    api_client = SemanticScholarAPI(api_key, rate_limit_delay)
    
    # Process cited papers
    cited_papers = submission_data.get("cited_papers", [])
    logger.info(f"Processing {len(cited_papers)} cited papers")
    
    found_papers = 0
    failed_papers = 0
    
    try:
        for paper_idx, paper in enumerate(cited_papers):
            title = paper.get("title", "").strip()
            
            if not title:
                logger.debug(f"Skipping paper {paper_idx + 1}: no title")
                continue
                
            logger.debug(f"Processing paper {paper_idx + 1}: '{title}'")

            # Fetch paper from Semantic Scholar
            ss_paper_obj = api_client.get_paper_by_title(title)

            if ss_paper_obj:
                paper["ss_paper_obj"] = ss_paper_obj.to_dict()
                found_papers += 1
            else:
                paper["ss_paper_obj"] = None
                failed_papers += 1
        
        # Save enriched data back to original file
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(submission_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Successfully processed {len(cited_papers)} papers: {found_papers} found, {failed_papers} failed")
        return True
        
    except Exception as e:
        logger.error(f"Error processing submission {submission_id}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich cited papers with Semantic Scholar data - single submission mode only"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data directory containing submission data"
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        required=True,
        help="Submission ID to process"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Semantic Scholar API key (or set SEMANTIC_SCHOLAR_API_KEY env var)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Rate limit delay in seconds between requests"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process single submission
    logger.info(f"Processing submission {args.submission_id}")
    logger.info(f"API key: {args.api_key}")
    success = process_for_pipeline(
        data_dir=args.data_dir,
        submission_id=args.submission_id,
        api_key=args.api_key,
        rate_limit_delay=args.rate_limit
    )
    
    if success:
        logger.info(f"✅ Successfully enriched submission {args.submission_id}")
        print(f"✅ Successfully processed submission {args.submission_id}")
    else:
        logger.error(f"❌ Failed to enrich submission {args.submission_id}")
        print(f"❌ Failed to process submission {args.submission_id}")
