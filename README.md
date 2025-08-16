<!-- <p  align="center">
  <img src='logo.png' width='200'>
</p> -->

# Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback

<!-- [![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com) -->
<!-- [![License](https://img.shields.io/github/license/UKPLab/arxiv2025-assessing-paper-novelty)](https://opensource.org/licenses/Apache-2.0) -->
<!-- [![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/) -->
<!-- [![CI](https://github.com/UKPLab/arxiv2025-assessing-paper-novelty/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/arxiv2025-assessing-paper-novelty/actions/workflows/main.yml) -->

Novelty assessment is a central yet understudied aspect of peer review, particularly in high-volume fields like NLP where reviewer capacity is strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence-based assessment. Informed by a large-scale analysis of human-written novelty reviews, our method captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human-annotated reviewer novelty assessments, it achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions, substantially outperforming existing LLM-based baselines. The method produces detailed, literature-aware analyses, improves consistency over ad hoc judgments, and demonstrates the potential of structured LLM-assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.

This repo contains the code and data used to produce the experiments in this paper.

Contact person: [Osama Mohammed Afzal](mailto:osama.afzal@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
pip install -r requirements.txt
```

### Prerequisites

1. **API Keys**: Copy `.env.example` to `.env` and add your API keys:

   - `OPENAI_API_KEY`: Required for structured extraction and novelty assessment
   - `SEMANTIC_SCHOLAR_API_KEY`: Optional, improves rate limits for paper retrieval

2. **GROBID Setup**: The pipeline requires GROBID for extracting structured metadata from PDFs:

   - **Installation**: Follow the installation instructions at [GROBID repository](https://github.com/kermitt2/grobid)
   - **Usage**: Process your PDF papers to generate TEI XML files:
     ```bash
     # Example GROBID processing (refer to GROBID docs for detailed instructions)
     curl -X POST -F "input=@paper.pdf" localhost:8070/api/processFulltextDocument
     ```
   - **Expected Output**: The pipeline expects GROBID TEI XML files in this structure:
     ```
     data/{submission_id}/{submission_id}.grobid.tei.xml
     ```

3. **OCR Processing**: The pipeline requires OCR processing of PDF papers to extract introductions. You can use either:

   - **Nougat OCR**: Follow installation instructions at [Nougat repository](https://github.com/facebookresearch/nougat)
   - **MinerU OCR**: Follow installation instructions at [MinerU repository](https://github.com/opendatalab/MinerU)

   The pipeline expects OCR output in these specific directory structures:

   **For Main Paper** (any one of these):

   ```
   data/{submission_id}/ocr_output/{submission_id}/auto/{submission_id}.md
   data/{submission_id}/nougat_output/{submission_id}.mmd
   data/{submission_id}/mineru_output/{submission_id}.md
   ```

   **For Related Papers** (any one of these):

   ```
   data/{submission_id}/related_work_data/ocr_output/{paper_id}/auto/{paper_id}.md
   data/{submission_id}/related_work_data/nougat_output/{paper_id}.mmd
   ```

   **Generated Introduction Files** (created by pipeline):

   ```
   data/{submission_id}/ours/{submission_id}_intro.txt              # main paper
   data/{submission_id}/ours/related_papers/{paper_id}_intro.txt    # related papers
   ```

## Usage

### Complete Pipeline

The novelty assessment pipeline consists of several stages. For a single submission:

0. **GROBID Processing**: Process submission PDF with GROBID (external step)

   ```bash
   # Start GROBID service (refer to GROBID documentation)
   # Process PDF to generate TEI XML file
   # Save as: data/{submission_id}/{submission_id}.grobid.tei.xml
   ```

1. **Preprocess**: Extract metadata from GROBID TEI XML

   ```bash
   cd src/preprocess
   python extract_metadata.py --data-dir /path/to/data --submission-id SUBMISSION_ID
   ```

2. **Enrich Citations**: Add Semantic Scholar data to citations

   ```bash
   cd src/retrieval
   python match_papers_to_s2.py --input /path/to/data --submission-id SUBMISSION_ID
   ```

3. **Retrieve Related Papers**: Find and rank related papers

   ```bash
   cd src/retrieval
   python retrieval.py --input /path/to/data --submission-id SUBMISSION_ID
   ```

4. **Download PDFs**: Download PDFs of ranked papers

   ```bash
   cd src/retrieval
   python get_cited_pdfs.py --data-dir /path/to/data --submission-id SUBMISSION_ID
   ```

5. **OCR Processing**: Process PDFs with Nougat or MinerU (external step)

   ```bash
   # Process main paper PDF and related paper PDFs with OCR tool of choice
   # Save outputs in expected directory structure (see Prerequisites)
   ```

6. **Extract Introductions**: Extract introductions from OCR output

   ```bash
   cd src/retrieval
   python extract_introductions.py --data-dir /path/to/data --submission-id SUBMISSION_ID
   ```

7. **Run Analysis**: Complete novelty assessment pipeline
   ```bash
   cd src/novelty_assessment
   python pipeline.py --data-dir /path/to/data --submission-id SUBMISSION_ID
   ```

### Individual Components

Each stage can also be run independently using the CLI interfaces provided:

- **`extract_metadata.py`**: Extracts structured metadata (title, abstract, citations, citation contexts) from GROBID TEI XML files
- **`match_papers_to_s2.py`**: Enriches citations with Semantic Scholar data (abstracts, paper IDs, publication info)
- **`retrieval.py`**: Generates search keywords, queries Semantic Scholar, ranks papers using SPECTER2 embeddings and RankGPT
- **`get_cited_pdfs.py`**: Downloads PDFs of ranked papers from ArXiv, ACL Anthology, and Semantic Scholar
- **`extract_introductions.py`**: Extracts introduction sections from OCR-processed papers using pattern matching
- **`structured_extraction.py`**: Uses LLMs to extract structured information (methods, problems, datasets, results, novelty claims)
- **`research_landscape.py`**: Analyzes the research landscape and identifies methodological clusters and relationships
- **`novelty_assessment.py`**: Performs detailed novelty analysis comparing submission against related work
- **`generate_summary.py`**: Generates final reviewer guidance summarizing the novelty assessment
- **`pipeline.py`**: Orchestrates the complete analysis pipeline with dependency checking

### Expected results

After running the complete pipeline, you should expect the following output structure and files:

```
data/{submission_id}/
├── {submission_id}.grobid.tei.xml           # Input: GROBID TEI XML file
├── ours/
│   ├── {submission_id}.json                 # Extracted metadata (title, abstract, citations)
│   ├── {submission_id}_intro.txt            # Main paper introduction
│   ├── related_papers/                      # Related papers introductions
│   │   ├── {paper_id}_intro.txt
│   │   └── ...
│   ├── s2_enriched_{submission_id}.json     # Citations enriched with Semantic Scholar data
│   ├── related_work_{submission_id}.json    # Retrieved and ranked related papers
│   ├── structured_extraction_{submission_id}.json  # LLM-extracted structured information
│   ├── research_landscape_{submission_id}.json     # Research landscape analysis
│   ├── novelty_assessment_{submission_id}.json     # Detailed novelty assessment
│   └── summary_{submission_id}.json         # **Final summary for reviewers**
└── related_work_data/
    ├── pdfs/                                # Downloaded related paper PDFs
    └── ocr_output/                          # OCR processed papers
```

**Key Output Files:**

1. **`summary_{submission_id}.json`** - The main output containing:

   - **Executive Summary**: High-level novelty assessment and recommendations
   - **Detailed Analysis**: Evidence-based comparison with related work
   - **Reviewer Guidance**: Structured feedback for peer reviewers
   - **Supporting Evidence**: Citations and specific comparisons

2. **`novelty_assessment_{submission_id}.json`** - Detailed technical analysis including:

   - Methodological comparisons with related work
   - Innovation assessment across different dimensions
   - Evidence-based novelty scoring
   - Specific technical differentiators

3. **`structured_extraction_{submission_id}.json`** - Structured information extracted from paper:
   - Methods and approaches used
   - Problems addressed and datasets
   - Key results and claims
   - Novelty assertions by authors

The pipeline produces comprehensive, literature-aware analyses that help reviewers assess novelty systematically rather than making ad hoc judgments. The final summary provides actionable guidance for peer review decisions.

### Key Parameters

The pipeline components accept these main parameters:

- `--data-dir`: Base directory containing submission data (required for all components)
- `--submission-id`: Unique identifier for the submission being processed (required for pipeline mode)
- `--input`: Input directory path (used by some retrieval components)
- `--verbose, -v`: Enable detailed logging output (optional)

## Development

For development work:

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. The codebase is organized into three main stages:

   - `src/preprocess/`: Metadata extraction from GROBID TEI files
   - `src/retrieval/`: Paper retrieval, PDF download, and introduction extraction
   - `src/novelty_assessment/`: LLM-based analysis and summary generation

3. Each component includes both CLI interface and pipeline integration methods for flexible usage.

## Cite

Please use the following citation:

```
@misc{afzal2025notnovelenoughenriching,
      title={Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback},
      author={Osama Mohammed Afzal and Preslav Nakov and Tom Hope and Iryna Gurevych},
      year={2025},
      eprint={2508.10795},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.10795},
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
