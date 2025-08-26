import os
import re
import argparse
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_introduction(mmd_file_path):
    """
    Extract the introduction section from a Nougat-OCR generated MMD file of an ICLR 2025 paper.
    The function looks for Introduction or Background sections in various formats, including
    cases with Roman numerals and standalone lines without hashtags.

    Args:
        mmd_file_path (str): Path to the MMD file.

    Returns:
        str: The extracted introduction section text or None if not found.
    """
    try:
        with open(mmd_file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Patterns for Introduction
        intro_patterns = [
            r"## *1 *Introduction.*?\n(.*?)(?=## *\d|\Z)",  # Standard numbered intro
            r"## *I *Introduction.*?\n(.*?)(?=## *[IVX]|\Z)",  # Roman numeral I
            r"##+ *Introduction.*?\n(.*?)(?=##|\Z)",  # Any level heading named Introduction
            r"# *1[.: ]* *Introduction.*?\n(.*?)(?=# *\d|\Z)",  # Different heading level with number
            r"# *I[.: ]* *Introduction.*?\n(.*?)(?=# *[IVX]|\Z)",  # Different heading with Roman numeral
            r"\\section\{[1.: ]*Introduction.*?\}(.*?)(?:\\section\{|\Z)",  # LaTeX style
            r"^Introduction\s*\n(.*?)(?=^[A-Z].*\n|\Z)",  # Standalone "Introduction" line
            r"\n\nIntroduction\s*\n(.*?)(?=\n\n[A-Z].*\n|\Z)",  # Standalone after double newline
            r"\n1\.?\s*Introduction\s*\n(.*?)(?=\n\d\.|\Z)",  # Numbered without hashtag
            r"\nI\.?\s*Introduction\s*\n(.*?)(?=\n[IVX]\.|\Z)",  # Roman numeral without hashtag
        ]

        # Added patterns for Background sections
        background_patterns = [
            r"## *2 *Background.*?\n(.*?)(?=## *\d|\Z)",  # Standard numbered background
            r"## *II *Background.*?\n(.*?)(?=## *[IVX]|\Z)",  # Roman numeral II
            r"##+ *Background.*?\n(.*?)(?=##|\Z)",  # Any level heading named Background
            r"# *2[.: ]* *Background.*?\n(.*?)(?=# *\d|\Z)",  # Different heading level with number
            r"# *II[.: ]* *Background.*?\n(.*?)(?=# *[IVX]|\Z)",  # Different heading with Roman numeral
            r"\\section\{[2.: ]*Background.*?\}(.*?)(?:\\section\{|\Z)",  # LaTeX style
            r"^Background\s*\n(.*?)(?=^[A-Z].*\n|\Z)",  # Standalone "Background" line
            r"\n\nBackground\s*\n(.*?)(?=\n\n[A-Z].*\n|\Z)",  # Standalone after double newline
            r"\n2\.?\s*Background\s*\n(.*?)(?=\n\d\.|\Z)",  # Numbered without hashtag
            r"\nII\.?\s*Background\s*\n(.*?)(?=\n[IVX]\.|\Z)",  # Roman numeral without hashtag
        ]

        # Try Introduction patterns first
        for pattern in intro_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                intro_text = match.group(1).strip()
                # Clean up the text
                intro_text = re.sub(r"\n{3,}", "\n\n", intro_text)
                return intro_text

        # If Introduction not found, try Background patterns
        for pattern in background_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                intro_text = match.group(1).strip()
                # Clean up the text
                intro_text = re.sub(r"\n{3,}", "\n\n", intro_text)
                return intro_text

        logger.debug(
            f"Neither Introduction nor Background section found in {Path(mmd_file_path).name}"
        )
        return None

    except Exception as e:
        logger.error(f"Error processing {Path(mmd_file_path).name}: {str(e)}")
        return None


def find_ocr_file(paper_id: str, base_dir: str) -> str:
    """
    Find the OCR output file for a given paper ID in the single ocr directory.
    Supports multiple file formats (.md, .mmd, .txt).
    
    Args:
        paper_id: The paper ID to find OCR for
        base_dir: Base directory containing ocr/ subdirectory
    
    Returns:
        str: Path to the OCR file, or None if not found
    """
    base_path = Path(base_dir)
    ocr_dir = base_path / "ocr"
    
    if not ocr_dir.exists():
        return None
    
    # Try different file extensions
    for ext in ['.md', '.mmd', '.txt']:
        ocr_file = ocr_dir / f"{paper_id}{ext}"
        if ocr_file.exists():
            return str(ocr_file)
    
    return None


def process_for_pipeline(data_dir: str, submission_id: str) -> bool:
    """
    Process a single submission for pipeline integration.
    Extract introductions for both main paper and related papers.
    
    Args:
        data_dir: Base data directory for pipeline
        submission_id: ID of the submission
    
    Returns:
        bool: Success status
    """
    submission_dir = Path(data_dir) / submission_id
    
    logger.info(f"Processing introductions for {submission_id}")
    
    # Stats tracking
    stats = {
        "main_paper": {"found": False, "extracted": False},
        "related_papers": {"total": 0, "found": 0, "extracted": 0}
    }
    
    # Single OCR directory for all papers
    ocr_dir = submission_dir / "ocr"
    
    # 1. Extract main paper introduction
    main_paper_id = submission_id
    main_ocr_file = find_ocr_file(main_paper_id, str(submission_dir))
    
    if main_ocr_file:
        stats["main_paper"]["found"] = True
        logger.info(f"Found main paper OCR: {main_ocr_file}")
        
        introduction = extract_introduction(main_ocr_file)
        if introduction:
            # Save main paper introduction to universal introductions folder
            intro_dir = submission_dir / "introductions"
            intro_dir.mkdir(exist_ok=True)
            intro_file = intro_dir / f"{submission_id}_intro.txt"
            with open(intro_file, "w", encoding="utf-8") as f:
                f.write(introduction)
            
            stats["main_paper"]["extracted"] = True
            logger.info(f"Extracted main paper introduction: {intro_file}")
        else:
            logger.warning(f"Could not extract introduction from main paper OCR")
    else:
        logger.warning(f"No OCR output found for main paper {submission_id}")
    
    # 2. Extract related papers introductions
    related_work_dir = submission_dir / "related_work_data"
    pdfs_dir = related_work_dir / "pdfs"
    
    if not pdfs_dir.exists():
        logger.warning(f"No PDFs directory found: {pdfs_dir}")
        return stats["main_paper"]["extracted"]  # Return success if at least main paper worked
    
    # Get list of PDF files to process
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    stats["related_papers"]["total"] = len(pdf_files)
    
    logger.info(f"Found {len(pdf_files)} related paper PDFs")
    
    # Create output directory for related papers
    related_output_dir = submission_dir / "related_papers"
    related_output_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_file in pdf_files:
        paper_id = pdf_file.stem  # Remove .pdf extension
        
        # Find OCR file in the single OCR directory
        ocr_file = find_ocr_file(paper_id, str(submission_dir))
        
        if ocr_file:
            stats["related_papers"]["found"] += 1
            logger.debug(f"Found OCR for {paper_id}: {ocr_file}")
            
            introduction = extract_introduction(ocr_file)
            if introduction:
                # Save related paper introduction to universal introductions folder
                intro_dir = submission_dir / "introductions"
                intro_dir.mkdir(exist_ok=True)
                intro_file = intro_dir / f"{paper_id}_intro.txt"
                with open(intro_file, "w", encoding="utf-8") as f:
                    f.write(introduction)
                
                stats["related_papers"]["extracted"] += 1
                logger.debug(f"Extracted introduction for {paper_id}")
            else:
                logger.warning(f"Could not extract introduction from {paper_id}")
        else:
            logger.warning(f"No OCR output found for related paper {paper_id}")
    
    # Log final stats
    logger.info(f"Introduction extraction completed for {submission_id}:")
    logger.info(f"  Main paper: {'✅' if stats['main_paper']['extracted'] else '❌'}")
    logger.info(f"  Related papers: {stats['related_papers']['extracted']}/{stats['related_papers']['total']} extracted")
    
    # Return success if we got at least the main paper or some related papers
    return stats["main_paper"]["extracted"] or stats["related_papers"]["extracted"] > 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract introductions from OCR outputs - single submission mode only")
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
        logger.info(f"✅ Successfully extracted introductions for {args.submission_id}")
        print(f"✅ Successfully processed submission {args.submission_id}")
    else:
        logger.error(f"❌ Failed to extract introductions for {args.submission_id}")
        print(f"❌ Failed to process submission {args.submission_id}")
