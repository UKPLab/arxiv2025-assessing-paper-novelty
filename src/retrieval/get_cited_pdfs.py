import requests
import os
import xml.etree.ElementTree as ET
import urllib.parse
import argparse
from acl_anthology import Anthology
from rapidfuzz import process, fuzz
from dataclasses import dataclass
from typing import List
import numpy as np
import json
import time
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ACL Anthology - lazy initialization
anthology = None
anthology_papers = None
paper_titles = None

INDENT = "    "  # 4-space indentation


def search_arxiv_by_title(title):
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f'ti:"{title}"', "start": 0, "max_results": 1}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        print(INDENT + f"❌ [arXiv] Failed to query")
        return None
    root = ET.fromstring(res.content)
    entry = root.find("{http://www.w3.org/2005/Atom}entry")
    if entry is not None:
        id_url = entry.find("{http://www.w3.org/2005/Atom}id").text
        arxiv_id = id_url.split("/abs/")[-1]
        print(INDENT + f"✅ [arXiv] Found: {arxiv_id}")
        return arxiv_id
    print(INDENT + "❌ [arXiv] Not found")
    return None


def fetch_acl_pdf_url(title):
    global anthology, anthology_papers, paper_titles
    
    # Initialize ACL Anthology on first use
    if anthology is None:
        logger.info("Initializing ACL Anthology...")
        anthology = Anthology.from_repo()
        anthology_papers = list(anthology.papers())
        paper_titles = [str(i.title).lower() for i in anthology_papers]
        logger.info(f"Loaded {len(anthology_papers)} ACL papers")
    
    # Use fuzzy matching to find the closest title
    closest_match = process.extractOne(
        title, paper_titles, scorer=fuzz.token_set_ratio, score_cutoff=80
    )
    if closest_match:
        closest_paper_index = paper_titles.index(closest_match[0])
        paper_obj = anthology_papers[closest_paper_index]
        pdf_url = paper_obj.pdf.url
        print(f"{INDENT}🎯 Closest match: {title[:30]}... | {paper_obj.title[:30]}...")
        return pdf_url
    else:
        print(INDENT + "❌ [ACL] Not found")
        return None


def download_pdf(pdf_url, title, paper_id, pdfs_dir):
    try:
        res = requests.get(pdf_url, timeout=30)
        if res.status_code == 200:
            os.makedirs(pdfs_dir, exist_ok=True)
            # Use paper_id as filename to avoid filesystem issues with titles
            filepath = f"{pdfs_dir}/{paper_id}.pdf"
            with open(filepath, "wb") as f:
                f.write(res.content)
            print(INDENT + f"📥 Downloaded → {filepath}")
            return True
        print(
            INDENT + f"❌ Failed to download from {pdf_url} (status: {res.status_code})"
        )
    except Exception as e:
        print(INDENT + f"❌ Error downloading: {e}")
    return False


def find_and_download_pdf(paper_data, pdfs_dir):
    """
    Download PDF for a paper from the ranking results.

    Args:
        paper_data: Dictionary containing paper information from ranking results
        pdfs_dir: Directory to save this paper's PDF

    Returns:
        str: Status code ("1"=failed, "2"=success, "3"=not_found)
    """
    paper_id = paper_data.get("paper_id", "unknown")
    title = paper_data.get("title", "Unknown Title")

    filepath = f"{pdfs_dir}/{paper_id}.pdf"
    print(f"\n🔎 {title[:60]}...")
    print(f"{INDENT}📋 Paper ID: {paper_id}")

    if os.path.exists(filepath):
        print(INDENT + f"📄 Already exists: {filepath}")
        return "2"

    # Try Semantic Scholar open access PDF first
    ss_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {"fields": "title,openAccessPdf,externalIds"}
    # Get API key from environment variable
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"X-API-KEY": api_key} if api_key else {}

    try:
        res = requests.get(ss_url, params=params, headers=headers, timeout=30)

        if res.status_code == 429:
            print(INDENT + "⏳ Rate limited, waiting...")
            time.sleep(5)
            res = requests.get(ss_url, params=params, headers=headers, timeout=30)

        if res.status_code != 200:
            print(INDENT + f"❌ [SS] Metadata fetch failed ({res.status_code})")
            return "1"

        data = res.json()
        pdf_url = data.get("openAccessPdf", {})

        if pdf_url and pdf_url.get("url"):
            print(INDENT + f"🔗 [SS] Trying openAccess PDF")
            if download_pdf(pdf_url["url"], title, paper_id, pdfs_dir):
                return "2"

        # Try arXiv
        arxiv_id = data.get("externalIds", {}).get("ArXiv")
        if not arxiv_id:
            print(INDENT + "🔍 [arXiv] Searching by title...")
            arxiv_id = search_arxiv_by_title(title)

        if arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            print(INDENT + f"🔗 [arXiv] Trying: {arxiv_id}")
            if download_pdf(pdf_url, title, paper_id, pdfs_dir):
                return "2"

        # Try ACL Anthology
        print(INDENT + "🔍 [ACL] Searching...")
        pdf_url = fetch_acl_pdf_url(title)
        if pdf_url:
            if download_pdf(pdf_url, title, paper_id, pdfs_dir):
                return "2"

        return "3"  # Not found

    except Exception as e:
        print(INDENT + f"❌ Error processing paper: {e}")
        return "1"


def process_for_pipeline(data_dir: str, submission_id: str) -> bool:
    """
    Process a single submission for pipeline integration.
    Download PDFs for ranked papers from a specific submission.
    
    Args:
        data_dir: Base data directory for pipeline
        submission_id: ID of the submission
    
    Returns:
        bool: Success status
    """
    submission_dir = Path(data_dir) / submission_id / "related_work_data"
    complete_results_file = submission_dir / "complete_results.json"
    
    if not complete_results_file.exists():
        logger.error(f"No complete results found for {submission_id}")
        return False
    
    try:
        with open(complete_results_file, "r") as f:
            results = json.load(f)
        
        # Get final ranked papers
        final_ranked = results.get("final_ranked_papers", [])
        if not final_ranked:
            logger.warning(f"No ranked papers found for {submission_id}")
            return False
        
        logger.info(f"Found {len(final_ranked)} ranked papers for {submission_id}")
        
        # Create pdfs subdirectory
        pdfs_dir = submission_dir / "pdfs"
        pdfs_dir.mkdir(exist_ok=True)
        
        # Download PDFs for each ranked paper
        stats = {"downloaded": 0, "failed": 0, "not_found": 0}
        
        for i, paper in enumerate(final_ranked):
            logger.info(f"Processing paper {i+1}/{len(final_ranked)}: {paper.get('title', 'Unknown')[:50]}...")
            
            result = find_and_download_pdf(paper, str(pdfs_dir))
            
            if result == "2":
                stats["downloaded"] += 1
            elif result == "3":
                stats["not_found"] += 1
            else:
                stats["failed"] += 1
            
            # Small delay to be nice to APIs
            time.sleep(1)
        
        # Update metadata
        metadata_file = submission_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        else:
            metadata = {}
        
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        metadata["pdf_download"] = {
            "total_papers": len(final_ranked),
            "pdfs_downloaded": len(pdf_files),
            "download_success_rate": (len(pdf_files) / len(final_ranked) * 100 if final_ranked else 0),
            "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": stats
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"PDF download completed for {submission_id}: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {submission_id}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDFs for ranked papers - single submission mode only")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Base data directory"
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        required=True,
        help="Submission ID to process"
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
    
    logger.info(f"Processing submission {args.submission_id}")
    success = process_for_pipeline(args.data_dir, args.submission_id)
    
    if success:
        logger.info(f"✅ Successfully downloaded PDFs for {args.submission_id}")
        print(f"✅ Successfully processed submission {args.submission_id}")
    else:
        logger.error(f"❌ Failed to download PDFs for {args.submission_id}")
        print(f"❌ Failed to process submission {args.submission_id}")
