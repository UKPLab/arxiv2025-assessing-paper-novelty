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
    year: Optional[int] = None
    citation_count: int = 0
    novel: Optional[str] = None
    authors: List[str] = field(default_factory=list)
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

            # Extract authors properly
            authors = []
            if "authors" in paper_data and paper_data["authors"]:
                authors = [author.get("name", "") for author in paper_data["authors"]]

            paper_obj = Paper(
                paper_id=paper_data.get("paperId", "Unknown ID"),
                title=paper_data.get("title", "Unknown Title"),
                abstract=paper_data.get("abstract", "No abstract available"),
                publication_date=paper_data.get("publicationDate"),
                venue=paper_data.get("venue"),
                year=paper_data.get("year"),
                citation_count=paper_data.get("citationCount", 0),
                authors=authors,
            )

            logger.info(f"✅ Found paper: '{paper_obj.title}' ({paper_obj.year})")
            return paper_obj

        except Exception as e:
            logger.error(f"Error fetching paper '{title}': {e}")
            return None


def process_submissions(
    input_file: str,
    output_jsonl: str,
    output_json: str,
    api_key: str = None,
    rate_limit_delay: float = 1.0,
):
    """
    Process submissions and fetch Semantic Scholar data for cited papers.

    Args:
        input_file: Path to input JSON file with submissions
        output_jsonl: Path to output JSONL file
        output_json: Path to output JSON file
        api_key: Semantic Scholar API key
        rate_limit_delay: Delay between API requests in seconds
    """
    # Initialize API client
    api_client = SemanticScholarAPI(api_key, rate_limit_delay)

    # Load submissions
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            submissions = json.load(f)
        logger.info(f"Loaded {len(submissions)} submissions from {input_file}")
    except Exception as e:
        logger.error(f"Failed to load input file {input_file}: {e}")
        return

    # Create output directory if it doesn't exist
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    # Clear output files
    with open(output_jsonl, "w") as f:
        pass

    # Initialize tracking variables
    total_papers = 0
    found_papers = 0
    failed_papers = 0
    submission_stats = []
    all_failed_titles = []

    # Process each submission
    for i, submission in enumerate(submissions):
        submission_id = submission.get("id", f"submission_{i+1}")
        logger.info(
            f"Processing submission {i+1}/{len(submissions)} (ID: {submission_id})"
        )

        # Initialize submission-level tracking
        submission_total = 0
        submission_found = 0
        submission_failed = 0
        submission_failed_titles = []

        if "cited_papers" not in submission:
            logger.warning(f"Submission {submission_id} has no 'cited_papers' field")
            submission_stats.append(
                {
                    "submission_id": submission_id,
                    "submission_index": i + 1,
                    "total_papers": 0,
                    "found_papers": 0,
                    "failed_papers": 0,
                    "success_rate": 0.0,
                    "failed_titles": [],
                    "error": "No cited_papers field",
                }
            )
            continue

        for j, paper in enumerate(submission["cited_papers"]):
            title = paper.get("title", "").strip()
            if not title:
                logger.debug(
                    f"Skipping paper {j+1} in submission {submission_id}: no title"
                )
                continue

            submission_total += 1
            total_papers += 1
            logger.debug(f"Processing paper {j+1}: '{title}'")

            # Fetch paper from Semantic Scholar
            ss_paper_obj = api_client.get_paper_by_title(title)

            if ss_paper_obj:
                paper["ss_paper_obj"] = ss_paper_obj.to_dict()
                submission_found += 1
                found_papers += 1
            else:
                paper["ss_paper_obj"] = None
                submission_failed += 1
                failed_papers += 1
                submission_failed_titles.append(title)
                all_failed_titles.append(
                    {
                        "submission_id": submission_id,
                        "submission_index": i + 1,
                        "title": title,
                    }
                )
                logger.warning(f"❌ Failed to find: '{title}'")

        # Calculate submission success rate
        success_rate = (
            (submission_found / submission_total * 100) if submission_total > 0 else 0
        )

        # Store submission statistics
        submission_stats.append(
            {
                "submission_id": submission_id,
                "submission_index": i + 1,
                "total_papers": submission_total,
                "found_papers": submission_found,
                "failed_papers": submission_failed,
                "success_rate": round(success_rate, 1),
                "failed_titles": submission_failed_titles,
            }
        )

        logger.info(
            f"📊 Submission {submission_id}: {submission_found}/{submission_total} papers found ({success_rate:.1f}%)"
        )

        # Write submission to JSONL file after processing
        try:
            with open(output_jsonl, "a", encoding="utf-8") as f:
                json.dump(submission, f, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write submission {submission_id} to JSONL: {e}")

    # Write final JSON output
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(submissions, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully wrote final output to {output_json}")
    except Exception as e:
        logger.error(f"Failed to write final JSON output: {e}")

    # Save detailed statistics and failure reports
    stats_file = Path(output_json).parent / "processing_stats.json"
    failures_file = Path(output_json).parent / "failed_titles.json"

    # Save submission statistics
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "total_submissions": len(submissions),
                        "total_papers": total_papers,
                        "papers_found": found_papers,
                        "papers_failed": failed_papers,
                        "overall_success_rate": round(
                            (
                                (found_papers / total_papers * 100)
                                if total_papers > 0
                                else 0
                            ),
                            1,
                        ),
                    },
                    "per_submission_stats": submission_stats,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"📊 Detailed statistics saved to {stats_file}")
    except Exception as e:
        logger.error(f"Failed to save statistics: {e}")

    # Save failed titles
    try:
        with open(failures_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_failed": len(all_failed_titles),
                    "failed_titles": all_failed_titles,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"❌ Failed titles report saved to {failures_file}")
    except Exception as e:
        logger.error(f"Failed to save failure report: {e}")

    # Log summary statistics
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total submissions processed: {len(submissions)}")
    logger.info(f"Total papers processed: {total_papers}")
    logger.info(f"Papers found: {found_papers} ({found_papers/total_papers*100:.1f}%)")
    logger.info(
        f"Papers not found: {failed_papers} ({failed_papers/total_papers*100:.1f}%)"
    )
    logger.info(f"Output files: {output_jsonl}, {output_json}")
    logger.info(f"Statistics file: {stats_file}")
    logger.info(f"Failures file: {failures_file}")

    # Log worst performing submissions
    if submission_stats:
        worst_submissions = sorted(submission_stats, key=lambda x: x["success_rate"])[
            :5
        ]
        logger.info("\n📉 WORST PERFORMING SUBMISSIONS:")
        for sub in worst_submissions:
            if sub["total_papers"] > 0:
                logger.info(
                    f"  - {sub['submission_id']}: {sub['success_rate']}% ({sub['found_papers']}/{sub['total_papers']})"
                )

        # Log most common failure patterns (if any)
        if all_failed_titles:
            logger.info(
                f"\n❌ TOTAL FAILURES: {len(all_failed_titles)} titles across all submissions"
            )
            logger.info(
                "   Check 'failed_titles.json' for complete list with submission mapping"
            )


def analyze_failures(
    failures_file: str = "./failed_titles.json",
    stats_file: str = "./processing_stats.json",
):
    """
    Analyze failure patterns and generate insights.

    Args:
        failures_file: Path to failed titles JSON file
        stats_file: Path to processing stats JSON file
    """
    try:
        # Load failure data
        with open(failures_file, "r", encoding="utf-8") as f:
            failures_data = json.load(f)

        with open(stats_file, "r", encoding="utf-8") as f:
            stats_data = json.load(f)

        print("\n" + "=" * 60)
        print("FAILURE ANALYSIS REPORT")
        print("=" * 60)

        # Overall stats
        total_failed = failures_data["total_failed"]
        total_papers = stats_data["summary"]["total_papers"]

        print(
            f"\n📊 OVERALL FAILURE RATE: {total_failed}/{total_papers} ({total_failed/total_papers*100:.1f}%)"
        )

        # Per-submission failure analysis
        submission_failures = {}
        for failure in failures_data["failed_titles"]:
            sub_id = failure["submission_id"]
            if sub_id not in submission_failures:
                submission_failures[sub_id] = []
            submission_failures[sub_id].append(failure["title"])

        print(f"\n📋 SUBMISSIONS WITH FAILURES: {len(submission_failures)}")

        # Find submissions with highest failure rates
        worst_subs = []
        for sub_stat in stats_data["per_submission_stats"]:
            if sub_stat["failed_papers"] > 0:
                worst_subs.append(
                    {
                        "id": sub_stat["submission_id"],
                        "failed_count": sub_stat["failed_papers"],
                        "total_count": sub_stat["total_papers"],
                        "failure_rate": 100 - sub_stat["success_rate"],
                    }
                )

        worst_subs.sort(key=lambda x: x["failure_rate"], reverse=True)

        print(f"\n🔥 TOP 10 SUBMISSIONS BY FAILURE RATE:")
        for i, sub in enumerate(worst_subs[:10]):
            print(
                f"  {i+1:2}. {sub['id']}: {sub['failed_count']}/{sub['total_count']} failed ({sub['failure_rate']:.1f}%)"
            )

        # Common title patterns in failures
        failed_titles = [f["title"] for f in failures_data["failed_titles"]]

        # Analyze title characteristics
        short_titles = [t for t in failed_titles if len(t) < 50]
        long_titles = [t for t in failed_titles if len(t) > 100]
        has_special_chars = [t for t in failed_titles if any(c in t for c in "[](){}")]

        print(f"\n🔍 FAILED TITLE CHARACTERISTICS:")
        print(
            f"  - Short titles (<50 chars): {len(short_titles)} ({len(short_titles)/len(failed_titles)*100:.1f}%)"
        )
        print(
            f"  - Long titles (>100 chars): {len(long_titles)} ({len(long_titles)/len(failed_titles)*100:.1f}%)"
        )
        print(
            f"  - With special characters: {len(has_special_chars)} ({len(has_special_chars)/len(failed_titles)*100:.1f}%)"
        )

        # Sample failed titles
        print(f"\n📝 SAMPLE FAILED TITLES (first 10):")
        for i, title in enumerate(failed_titles[:10]):
            print(f"  {i+1:2}. {title[:80]}{'...' if len(title) > 80 else ''}")

        print(f"\n💾 Full failure details saved in: {failures_file}")
        print(f"💾 Full statistics saved in: {stats_file}")

    except FileNotFoundError as e:
        print(f"❌ Analysis files not found: {e}")
        print("Run the main processing first to generate failure reports.")
    except Exception as e:
        print(f"❌ Error during failure analysis: {e}")


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
    # Input file path from GROBID preprocessing
    input_file = os.path.join(data_dir, submission_id, "ours", f"{submission_id}.json")
    
    # Output files
    output_dir = Path(data_dir) / submission_id / "ours"
    output_jsonl = output_dir / f"{submission_id}_ss_enriched.jsonl"
    output_json = output_dir / f"{submission_id}_ss_enriched.json"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load single submission data
    with open(input_file, "r", encoding="utf-8") as f:
        submission_data = json.load(f)
    
    # Wrap in list format expected by process_submissions
    submissions = [{
        "id": submission_id,
        "cited_papers": submission_data.get("cited_papers", [])
    }]
    
    # Save temporary input file
    temp_input = output_dir / f"{submission_id}_temp_input.json"
    with open(temp_input, "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=2)
    
    try:
        # Process with Semantic Scholar enrichment
        process_submissions(
            input_file=str(temp_input),
            output_jsonl=str(output_jsonl),
            output_json=str(output_json),
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
        )
        
        # Clean up temporary file
        temp_input.unlink()
        
        # Update original submission file with enriched data
        with open(output_json, "r", encoding="utf-8") as f:
            enriched_submissions = json.load(f)
        
        if enriched_submissions:
            enriched_submission = enriched_submissions[0]
            submission_data["cited_papers"] = enriched_submission["cited_papers"]
            
            # Save back to original file
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(submission_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing submission {submission_id}: {e}")
        # Clean up temporary file if it exists
        if temp_input.exists():
            temp_input.unlink()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich cited papers with Semantic Scholar data"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file with citations (batch mode) or data directory (pipeline mode)"
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        help="Output JSONL file path (batch mode only)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Output JSON file path (batch mode only)"
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        help="Submission ID for pipeline mode"
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
        "--analyze",
        action="store_true",
        help="Run failure analysis on existing results"
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
    
    if args.analyze:
        # Run failure analysis
        print("\n" + "🔍" * 20 + " RUNNING FAILURE ANALYSIS " + "🔍" * 20)
        analyze_failures()
    elif args.submission_id:
        # Pipeline mode - process single submission
        if not args.input:
            parser.error("--input (data directory) is required for pipeline mode")
        
        logger.info(f"Processing submission {args.submission_id} in pipeline mode")
        success = process_for_pipeline(
            data_dir=args.input,
            submission_id=args.submission_id,
            api_key=args.api_key,
            rate_limit_delay=args.rate_limit
        )
        
        if success:
            logger.info(f"✅ Successfully enriched submission {args.submission_id}")
        else:
            logger.error(f"❌ Failed to enrich submission {args.submission_id}")
    else:
        # Batch mode - process multiple submissions
        if not all([args.input, args.output_jsonl, args.output_json]):
            parser.error("--input, --output-jsonl, and --output-json are required for batch mode")
        
        logger.info("Starting paper fetching process in batch mode...")
        
        process_submissions(
            input_file=args.input,
            output_jsonl=args.output_jsonl,
            output_json=args.output_json,
            api_key=args.api_key,
            rate_limit_delay=args.rate_limit,
        )
        
        logger.info("Paper fetching process completed!")
