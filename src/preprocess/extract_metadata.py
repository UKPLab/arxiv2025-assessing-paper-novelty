#!/usr/bin/env python3
"""
Enhanced GROBID TEI XML Parser
Extracts title, abstract, cited papers, and citation contexts from GROBID-parsed PDFs.
Based on efficient lxml parsing approach.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
from lxml import etree
from dotenv import load_dotenv

# Load environment variables for consistency
load_dotenv()

# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class CitedPaper:
    """Represents a cited paper with its bibliographic information."""

    id: str
    title: Optional[str] = None
    authors: List[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    raw_citation: Optional[str] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []


@dataclass
class CitationContext:
    """Represents a citation context (sentence-level where a paper was cited)."""

    cited_paper_id: str
    context_sentence: str
    sentence_id: Optional[str] = None
    section: Optional[str] = None


@dataclass
class ParsedPaper:
    """Represents a parsed paper with all extracted information."""

    filename: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    cited_papers: List[CitedPaper] = None
    citation_contexts: List[CitationContext] = None

    def __post_init__(self):
        if self.cited_papers is None:
            self.cited_papers = []
        if self.citation_contexts is None:
            self.citation_contexts = []


class EnhancedGrobidParser:
    """Enhanced parser for GROBID TEI XML files using lxml."""

    def __init__(self):
        self.ns = NS

    def parse_tei_file(self, tei_path: Path) -> ParsedPaper:
        """Parse a single TEI file and extract all information."""
        try:
            root = etree.parse(str(tei_path))
            pdf_name = tei_path.stem.replace(".grobid.tei", "").replace(".tei", "")

            # Extract basic paper information
            title = self._extract_title(root)
            abstract = self._extract_abstract(root)

            # Extract cited papers from bibliography
            cited_papers = self._extract_cited_papers(root)

            # Extract citation contexts
            citation_contexts = self._extract_citation_contexts(root, cited_papers)

            return ParsedPaper(
                filename=pdf_name,
                title=title,
                abstract=abstract,
                cited_papers=cited_papers,
                citation_contexts=citation_contexts,
            )

        except Exception as e:
            logging.error(f"Error parsing {tei_path}: {e}")
            return ParsedPaper(filename=tei_path.stem)

    def _extract_title(self, root: etree._ElementTree) -> Optional[str]:
        """Extract the paper title."""
        # Try main title first
        title_elems = root.xpath(
            './/tei:titleStmt/tei:title[@type="main"]', namespaces=self.ns
        )
        if title_elems:
            return self._get_text_content(title_elems[0]).strip()

        # Fallback to any title in titleStmt
        title_elems = root.xpath(".//tei:titleStmt/tei:title", namespaces=self.ns)
        if title_elems:
            return self._get_text_content(title_elems[0]).strip()

        return None

    def _extract_abstract(self, root: etree._ElementTree) -> Optional[str]:
        """Extract the paper abstract."""
        abstract_elems = root.xpath(".//tei:abstract", namespaces=self.ns)
        if abstract_elems:
            abstract_elem = abstract_elems[0]
            # Get text from all paragraphs in abstract
            paragraphs = abstract_elem.xpath(".//tei:p", namespaces=self.ns)
            if paragraphs:
                abstract_text = " ".join(
                    self._get_text_content(p).strip() for p in paragraphs
                )
                return abstract_text.strip()
            else:
                return self._get_text_content(abstract_elem).strip()

        return None

    def _extract_cited_papers(self, root: etree._ElementTree) -> List[CitedPaper]:
        """Extract cited papers from bibliography."""
        cited_papers = []

        # Find all bibliography entries
        bib_entries = root.xpath(".//tei:listBibl/tei:biblStruct", namespaces=self.ns)

        for bib in bib_entries:
            # Get citation ID
            cite_id = bib.get("{http://www.w3.org/XML/1998/namespace}id")
            if not cite_id:
                continue

            # Extract title
            title_elems = bib.xpath(
                './/tei:analytic/tei:title[@type="main"] | .//tei:monogr/tei:title[@type="main"] | .//tei:title',
                namespaces=self.ns,
            )
            title = (
                self._get_text_content(title_elems[0]).strip() if title_elems else None
            )

            # Extract authors
            authors = []
            author_elems = bib.xpath(
                ".//tei:analytic/tei:author/tei:persName | .//tei:monogr/tei:author/tei:persName",
                namespaces=self.ns,
            )

            for author_elem in author_elems:
                forename_elems = author_elem.xpath("tei:forename", namespaces=self.ns)
                surname_elems = author_elem.xpath("tei:surname", namespaces=self.ns)

                forename = (
                    self._get_text_content(forename_elems[0]) if forename_elems else ""
                )
                surname = (
                    self._get_text_content(surname_elems[0]) if surname_elems else ""
                )

                full_name = f"{forename} {surname}".strip()
                if full_name:
                    authors.append(full_name)

            # Extract year
            year = None
            date_elems = bib.xpath(
                './/tei:imprint/tei:date[@type="published"]', namespaces=self.ns
            )
            if date_elems:
                when_attr = date_elems[0].get("when")
                if when_attr and len(when_attr) >= 4:
                    year = when_attr[:4]

            # Extract venue
            venue_elems = bib.xpath(
                './/tei:monogr/tei:title[@type="main"]', namespaces=self.ns
            )
            venue = (
                self._get_text_content(venue_elems[0]).strip() if venue_elems else None
            )

            # Extract DOI
            doi_elems = bib.xpath('.//tei:idno[@type="DOI"]', namespaces=self.ns)
            doi = self._get_text_content(doi_elems[0]).strip() if doi_elems else None

            # Get raw citation XML
            raw_citation = etree.tostring(bib, encoding="unicode")

            cited_paper = CitedPaper(
                id=cite_id,
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                raw_citation=raw_citation,
            )

            cited_papers.append(cited_paper)

        return cited_papers

    def _extract_citation_contexts(
        self, root: etree._ElementTree, cited_papers: List[CitedPaper]
    ) -> List[CitationContext]:
        """Extract citation contexts at sentence level."""
        citation_contexts = []

        # Create mapping of citation IDs for quick lookup
        cited_paper_ids = {cp.id for cp in cited_papers}

        # Find all reference pointers to bibliographic entries
        ref_elems = root.xpath('.//tei:ref[@type="bibr"]', namespaces=self.ns)

        for ref in ref_elems:
            targets = ref.get("target", "").split()
            for target in targets:
                bid = target.lstrip("#")
                if bid not in cited_paper_ids:
                    continue

                # Find the parent sentence element containing this citation
                sentence_elem = self._find_parent_sentence(ref)
                if sentence_elem is None:
                    continue

                # Extract sentence text
                sentence_text = " ".join(sentence_elem.itertext()).strip()
                sentence_id = sentence_elem.get(
                    "{http://www.w3.org/XML/1998/namespace}id", ""
                )

                # Try to find the section this sentence belongs to
                section_title = self._find_section_title(sentence_elem)

                citation_context = CitationContext(
                    cited_paper_id=bid,
                    context_sentence=sentence_text,
                    sentence_id=sentence_id,
                    section=section_title,
                )

                citation_contexts.append(citation_context)

        return citation_contexts

    def _find_parent_sentence(self, element):
        """Find the parent sentence element (<s>) containing the given element."""
        node = element
        while node is not None and node.tag != f"{{{self.ns['tei']}}}s":
            node = node.getparent()
        return node

    def _find_section_title(self, element) -> Optional[str]:
        """Find the section title for a given element by traversing up the tree."""
        node = element
        while node is not None:
            # Look for a parent div with a head element
            if node.tag == f"{{{self.ns['tei']}}}div":
                head_elems = node.xpath("./tei:head", namespaces=self.ns)
                if head_elems:
                    return self._get_text_content(head_elems[0]).strip()
            node = node.getparent()
        return None

    def _get_text_content(self, element) -> str:
        """Get all text content from an element."""
        if element is None:
            return ""
        return " ".join(element.itertext()).strip()

    def process_single_tei_file(self, tei_file_path: str, output_file_path: str) -> ParsedPaper:
        """
        Process a single TEI file and save the result to JSON.
        Pipeline-friendly method for integration with other stages.

        Args:
            tei_file_path: Path to the TEI XML file
            output_file_path: Path to save the parsed JSON output

        Returns:
            ParsedPaper object with extracted information
        """
        tei_path = Path(tei_file_path)
        if not tei_path.exists():
            raise FileNotFoundError(f"TEI file not found: {tei_file_path}")

        # Parse the TEI file
        parsed_paper = self.parse_tei_file(tei_path)

        # Save to JSON
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(parsed_paper), f, indent=2, ensure_ascii=False)

        logging.info(f"Processed {tei_path.name} -> {output_path}")
        return parsed_paper

    def process_for_pipeline(self, tei_file_path: str, data_dir: str, submission_id: str) -> ParsedPaper:
        """
        Process a single TEI file and save in the expected pipeline directory structure.
        
        Args:
            tei_file_path: Path to the TEI XML file
            data_dir: Base data directory for pipeline
            submission_id: ID of the submission
            
        Returns:
            ParsedPaper object with extracted information
        """
        # Create the expected output path: data_dir/submission_id/submission_id.json
        output_path = Path(data_dir) / submission_id / f"{submission_id}.json"
        return self.process_single_tei_file(tei_file_path, str(output_path))



def main():
    parser = argparse.ArgumentParser(description="Enhanced GROBID TEI XML Parser - single submission mode only")
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
        "--tei-file",
        type=str,
        help="TEI XML file path (default: data_dir/submission_id/submission_id_fulltext.tei.xml)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Use default TEI file path if not provided
    if not args.tei_file:
        args.tei_file = str(Path(args.data_dir) / args.submission_id / f"{args.submission_id}_fulltext.tei.xml")

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Initialize parser
    grobid_parser = EnhancedGrobidParser()

    try:
        # Process for pipeline
        parsed_paper = grobid_parser.process_for_pipeline(
            args.tei_file, args.data_dir, args.submission_id
        )
        
        logging.info(f"📊 Processing summary:")
        logging.info(f"   Submission ID: {args.submission_id}")
        logging.info(f"   Input: {args.tei_file}")
        logging.info(f"   Output: {args.data_dir}/{args.submission_id}/{args.submission_id}.json")
        logging.info(f"   Title: {parsed_paper.title}")
        logging.info(f"   Abstract: {len(parsed_paper.abstract) if parsed_paper.abstract else 0} chars")
        logging.info(f"   Cited papers: {len(parsed_paper.cited_papers)}")
        logging.info(f"   Citation contexts: {len(parsed_paper.citation_contexts)}")

        logging.info(f"✅ Processing completed successfully")
        print(f"✅ Successfully processed submission {args.submission_id}")

    except Exception as e:
        logging.error(f"❌ Error during processing: {e}")
        print(f"❌ Failed to process submission {args.submission_id}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
