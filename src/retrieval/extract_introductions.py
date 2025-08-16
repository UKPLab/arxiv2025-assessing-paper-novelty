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
            f"Neither Introduction nor Background section found in {os.path.basename(mmd_file_path)}"
        )
        return None

    except Exception as e:
        logger.error(f"Error processing {os.path.basename(mmd_file_path)}: {str(e)}")
        return None


def find_ocr_file(paper_id: str, base_dir: str) -> str:
    """
    Find the OCR output file for a given paper ID.
    Looks in both ocr_output and nougat_output directories.
    
    Args:
        paper_id: The paper ID to find OCR for
        base_dir: Base directory to search in
    
    Returns:
        str: Path to the OCR file, or None if not found
    """
    # Try OCR output first (MinerU format)
    ocr_path = os.path.join(base_dir, "ocr_output", paper_id, "auto", f"{paper_id}.md")
    if os.path.exists(ocr_path):
        return ocr_path
    
    # Try nougat output (Nougat format)
    nougat_path = os.path.join(base_dir, "nougat_output", f"{paper_id}.mmd")
    if os.path.exists(nougat_path):
        return nougat_path
    
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
    
    # 1. Extract main paper introduction
    main_paper_id = submission_id
    main_ocr_base = submission_dir
    
    # Check for main paper OCR in multiple locations
    main_ocr_file = None
    for ocr_dir in ["ocr_output", "nougat_output", "mineru_output"]:
        if ocr_dir == "ocr_output":
            candidate = main_ocr_base / ocr_dir / main_paper_id / "auto" / f"{main_paper_id}.md"
        elif ocr_dir == "nougat_output":
            candidate = main_ocr_base / ocr_dir / f"{main_paper_id}.mmd"
        elif ocr_dir == "mineru_output":
            candidate = main_ocr_base / ocr_dir / f"{main_paper_id}.md"
        
        if candidate.exists():
            main_ocr_file = str(candidate)
            break
    
    if main_ocr_file:
        stats["main_paper"]["found"] = True
        logger.info(f"Found main paper OCR: {main_ocr_file}")
        
        introduction = extract_introduction(main_ocr_file)
        if introduction:
            # Save main paper introduction
            output_dir = submission_dir / "ours"
            output_dir.mkdir(exist_ok=True)
            
            intro_file = output_dir / f"{submission_id}_intro.txt"
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
    related_output_dir = submission_dir / "ours" / "related_papers"
    related_output_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_file in pdf_files:
        paper_id = pdf_file.stem  # Remove .pdf extension
        
        # Find OCR file for this paper
        ocr_file = find_ocr_file(paper_id, str(related_work_dir))
        
        if ocr_file:
            stats["related_papers"]["found"] += 1
            logger.debug(f"Found OCR for {paper_id}: {ocr_file}")
            
            introduction = extract_introduction(ocr_file)
            if introduction:
                # Save related paper introduction
                intro_file = related_output_dir / f"{paper_id}_intro.txt"
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


def batch_process(data_dir: str) -> dict:
    """
    Process all submissions in batch mode (original functionality).
    
    Args:
        data_dir: Directory containing submissions
    
    Returns:
        dict: Processing statistics
    """
    submissions = os.listdir(data_dir)

    total_papers = 0
    total_papers_with_intro = 0
    papers_file_not_found = 0
    papers_extraction_failed = 0

    for submission in submissions:
        submission_id = submission.split("_")[0]
        pdfs_dir = f"{data_dir}/{submission}/related_work_data/pdfs"
        
        if not os.path.exists(pdfs_dir):
            logger.warning(f"No PDFs directory for {submission_id}")
            continue
            
        related_papers = os.listdir(pdfs_dir)
        logger.info(f"Processing {submission_id} with {len(related_papers)} papers")
        
        for related_paper in related_papers:
            total_papers += 1
            related_paper_id = related_paper.split(".")[0]
            
            # Find OCR file
            ocr_file = find_ocr_file(related_paper_id, f"{data_dir}/{submission}/related_work_data")
            
            if not ocr_file:
                logger.warning(f"No OCR output found for {related_paper}")
                papers_file_not_found += 1
                continue

            introduction = extract_introduction(ocr_file)
            if introduction is None:
                logger.warning(f"Failed to extract introduction for {related_paper}")
                papers_extraction_failed += 1
                continue
            else:
                logger.debug(f"Extracted introduction for {related_paper}")
                total_papers_with_intro += 1
                
            # Save introduction
            os.makedirs(f"{data_dir}/{submission}/ours/related_papers", exist_ok=True)
            with open(
                f"{data_dir}/{submission}/ours/related_papers/{related_paper_id}_intro.txt",
                "w",
                encoding="utf-8"
            ) as f:
                f.write(introduction)

    # Return statistics
    stats = {
        "total_papers": total_papers,
        "papers_with_intro": total_papers_with_intro,
        "papers_file_not_found": papers_file_not_found,
        "papers_extraction_failed": papers_extraction_failed,
        "success_rate": total_papers_with_intro / total_papers * 100 if total_papers > 0 else 0
    }
    
    logger.info(f"Batch processing completed:")
    logger.info(f"  Total papers: {total_papers}")
    logger.info(f"  Papers with intro: {total_papers_with_intro} ({stats['success_rate']:.1f}%)")
    logger.info(f"  Papers file not found: {papers_file_not_found}")
    logger.info(f"  Papers extraction failed: {papers_extraction_failed}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract introductions from OCR outputs")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Base data directory"
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        help="Submission ID for pipeline mode"
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
    
    if not args.data_dir:
        parser.error("--data-dir is required")
    
    if args.submission_id:
        # Pipeline mode - single submission
        logger.info(f"Processing submission {args.submission_id} in pipeline mode")
        success = process_for_pipeline(args.data_dir, args.submission_id)
        
        if success:
            logger.info(f"✅ Successfully extracted introductions for {args.submission_id}")
        else:
            logger.error(f"❌ Failed to extract introductions for {args.submission_id}")
    else:
        # Batch mode - process all submissions
        logger.info("Running in batch mode")
        stats = batch_process(args.data_dir)
