import os
import json
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


research_landscape_prompt = """
# Research Landscape Analysis

## Task
Analyze the collection of research papers provided below to create a comprehensive map of the research landscape they represent. The submission paper is the focus of our analysis, and the related papers provide context.

## Input Format
You will be provided with structured information extracted from multiple research papers including:
- A submission paper that is the focus of our analysis
- Multiple related papers that form the research context

Each paper contains:
- methods: List of methods/approaches proposed
- problems: List of problems addressed
- datasets: List of datasets used
- metrics: List of evaluation metrics
- results: Key quantitative results
- novelty_claims: Claims about what is novel in the work

## Output Format
Provide a comprehensive analysis with the following sections:

1. METHODOLOGICAL LANDSCAPE
   - Identify and describe the main methodological approaches across the papers
   - Group similar or related methods into clusters
   - Highlight methodological trends or patterns
   - Describe relationships between different methodological approaches

2. PROBLEM SPACE MAPPING
   - Identify the key problems being addressed across the papers
   - Analyze how different papers approach similar problems
   - Highlight patterns in problem formulation

3. EVALUATION LANDSCAPE
   - Analyze the common datasets and evaluation methods
   - Identify patterns in how performance is measured
   - Compare evaluation approaches across papers

4. RESEARCH CLUSTERS
   - Identify groups of papers that appear closely related
   - Describe the key characteristics of each cluster
   - Analyze relationships between clusters

5. TECHNICAL EVOLUTION
   - Identify any visible progression or evolution of ideas
   - Highlight building blocks and their extensions
   - Note any competing or complementary approaches

## Example Output Format
METHODOLOGICAL LANDSCAPE
- Cluster 1: [Description of similar methods across papers]
  - Papers X, Y, Z employ transformer-based approaches with variations in...
  - These methods share characteristics such as...
  - They differ primarily in...

PROBLEM SPACE MAPPING
- Problem Area 1: [Description of a common problem addressed]
  - Papers A, B, C all address this problem but differ in...
  - The problem is formulated differently in Paper D which focuses on...

... [additional sections] ...

Ensure your analysis is comprehensive, identifying significant patterns and relationships across the collection of papers.

## Papers:
{papers}

"""


# Define models for structured data
class PaperInformation(BaseModel):
    """Structured information extracted from a research paper."""

    paper_id: str
    title: str
    methods: List[str] = Field(
        description="List of methods/approaches proposed in the paper"
    )
    problems: List[str] = Field(description="List of problems the paper addresses")
    datasets: List[str] = Field(description="List of datasets used for evaluation")
    metrics: List[str] = Field(description="List of evaluation metrics used")
    results: List[Dict[str, str]] = Field(
        description="Key quantitative results with metric name -> value pairs"
    )
    novelty_claims: List[str] = Field(
        description="Claims about what is novel in this work"
    )
    is_submission: bool = False


class ResearchLandscapeAnalyzer:
    """
    Analyzes a collection of papers to create a comprehensive research landscape.
    """

    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the analyzer with the specified LLM.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature setting for the LLM
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = ChatOpenAI(
            model_name=model_name, temperature=temperature, api_key=api_key
        )

        self.research_landscape_prompt = research_landscape_prompt

    def load_papers(self, data_dir: str, submission_id: str) -> List[PaperInformation]:
        """
        Load paper information from JSON files in the specified directory.

        Args:
            data_dir: Directory containing paper information JSON files
            submission_id: ID of the submission paper

        Returns:
            List of PaperInformation objects
        """
        papers = []

        with open(
            os.path.join(data_dir, submission_id, f"{submission_id}.json"),
            "r",
            encoding="utf-8",
        ) as f:
            submission_paper_obj = json.load(f)

        with open(
            os.path.join(
                data_dir, submission_id, "related_work_data", "ranked_papers.json"
            ),
            "r",
            encoding="utf-8",
        ) as f:
            ranked_papers = json.load(f)


        structured_representation_path = os.path.join(
            data_dir, submission_id, "structured_representation.json"
        )
        if not os.path.exists(structured_representation_path):
            return []

        with open(structured_representation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            structured_representation = data["structured_representation"]
            structured_representation["main_paper"]["parsed"][
                "paper_id"
            ] = submission_id
            structured_representation["main_paper"]["parsed"]["is_submission"] = True
            structured_representation["main_paper"]["parsed"]["title"] = (
                submission_paper_obj["title"]
            )
            paper_info = PaperInformation(
                **structured_representation["main_paper"]["parsed"]
            )
            papers.append(paper_info)

            selected_papers = {i["paper_id"]: i for i in ranked_papers}

            for item in structured_representation["selected_papers"]:
                item["parsed"]["paper_id"] = item["paper_id"]
                item["parsed"]["title"] = selected_papers[item["paper_id"]]["title"]
                item["parsed"]["is_submission"] = False
                paper_info = PaperInformation(**item["parsed"])
                papers.append(paper_info)

        return papers

    def format_papers_for_prompt(self, papers: List[PaperInformation]) -> str:
        """
        Format the paper information for the prompt.

        Args:
            papers: List of PaperInformation objects

        Returns:
            Formatted string for the prompt
        """
        # Separate submission and related papers
        submission_papers = [p for p in papers if p.is_submission]
        related_papers = [p for p in papers if not p.is_submission]

        if not submission_papers:
            raise ValueError("No submission paper found")

        submission = submission_papers[0]

        # Format submission paper
        formatted_text = "## SUBMISSION PAPER\n\n"
        formatted_text += f"TITLE: {submission.title}\n\n"
        formatted_text += (
            f"METHODS:\n" + "\n".join([f"- {m}" for m in submission.methods]) + "\n\n"
        )
        formatted_text += (
            f"PROBLEMS:\n" + "\n".join([f"- {p}" for p in submission.problems]) + "\n\n"
        )
        formatted_text += (
            f"DATASETS:\n" + "\n".join([f"- {d}" for d in submission.datasets]) + "\n\n"
        )
        formatted_text += (
            f"METRICS:\n" + "\n".join([f"- {m}" for m in submission.metrics]) + "\n\n"
        )
        formatted_text += (
            f"RESULTS:\n"
            + "\n".join([f"- {json.dumps(r)}" for r in submission.results])
            + "\n\n"
        )
        formatted_text += (
            f"NOVELTY CLAIMS:\n"
            + "\n".join([f"- {c}" for c in submission.novelty_claims])
            + "\n\n"
        )

        # Format related papers
        formatted_text += "## RELATED PAPERS\n\n"

        for i, paper in enumerate(related_papers, 1):
            formatted_text += f"### RELATED PAPER {i}: {paper.title}\n\n"
            formatted_text += (
                f"METHODS:\n" + "\n".join([f"- {m}" for m in paper.methods]) + "\n\n"
            )
            formatted_text += (
                f"PROBLEMS:\n" + "\n".join([f"- {p}" for p in paper.problems]) + "\n\n"
            )
            formatted_text += (
                f"DATASETS:\n" + "\n".join([f"- {d}" for d in paper.datasets]) + "\n\n"
            )
            formatted_text += (
                f"METRICS:\n" + "\n".join([f"- {m}" for m in paper.metrics]) + "\n\n"
            )
            formatted_text += (
                f"RESULTS:\n"
                + "\n".join([f"- {json.dumps(r)}" for r in paper.results])
                + "\n\n"
            )
            formatted_text += (
                f"NOVELTY CLAIMS:\n"
                + "\n".join([f"- {c}" for c in paper.novelty_claims])
                + "\n\n"
            )

        return formatted_text

    def analyze_landscape(self, papers: List[PaperInformation]) -> str:
        """
        Analyze the research landscape based on the provided papers.

        Args:
            papers: List of PaperInformation objects

        Returns:
            Analysis of the research landscape
        """
        # Format papers for the prompt
        formatted_papers = self.format_papers_for_prompt(papers)

        prompt = self.research_landscape_prompt.format(papers=formatted_papers)

        # Run the analysis
        analysis = self.llm.invoke(prompt)

        return analysis

    def run_analysis(self, data_dir: str, submission_id: str, output_file: str) -> str:
        """
        Run the complete analysis pipeline.

        Args:
            data_dir: Directory containing paper information JSON files
            submission_id: ID of the submission paper
            output_file: Path to save the analysis output

        Returns:
            Analysis of the research landscape
        """
        # Load papers
        papers = self.load_papers(data_dir, submission_id)

        # Check if we have enough papers
        if len(papers) < 2:
            raise ValueError(
                f"Not enough papers found. Need at least 2, got {len(papers)}"
            )

        # Run analysis
        analysis = self.analyze_landscape(papers)

        output_path = os.path.join(data_dir, submission_id, output_file)

        # Save analysis to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(analysis.content)

        print(f"Analysis saved to {output_file}")

        from litellm import cost_per_token

        tok_used = analysis.response_metadata["token_usage"]
        total_cost = sum(
            cost_per_token(
                model="gpt-4.1",
                prompt_tokens=tok_used["prompt_tokens"],
                completion_tokens=tok_used["completion_tokens"],
            )
        )

        meta_path = os.path.join(data_dir, submission_id, "metadata.json")

        current_meta = {}
        if os.path.exists(meta_path):
            current_meta = json.load(open(meta_path, "r"))

        current_meta["research_landscape"] = {
            "total_cost": total_cost,
            "token_usage": tok_used,
        }

        with open(meta_path, "w") as f:
            json.dump(current_meta, f, indent=3)

        return analysis

    def prepare_batch_inference_data(
        self, data_dir: str, model_name: str = "gpt-4.1", output_dir: str = "./openai_inputs"
    ) -> str:
        """
        Prepare data for batched inference with OpenAI API.

        Args:
            data_dir: Directory containing paper information JSON files
            model_name: Name of the OpenAI model to use
            output_dir: Directory to save the batch input file

        Returns:
            Path to the generated JSONL batch file
        """
        batch_entries = []

        submission_ids = os.listdir(data_dir)

        for submission_id in submission_ids:
            # Load papers for this submission
            papers = self.load_papers(data_dir, submission_id)

            if len(papers) < 2:
                print(
                    f"{submission_id}: Not enough papers found. Need at least 2, got {len(papers)}"
                )
                continue

            # Format papers for the prompt
            formatted_papers = self.format_papers_for_prompt(papers)

            # Prepare the prompt
            prompt = self.research_landscape_prompt.format(papers=formatted_papers)

            # Create the batch entry
            batch_entry = {
                "custom_id": f"research-landscape-{submission_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            }

            batch_entries.append(batch_entry)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write to JSONL file
        output_path = os.path.join(output_dir, "batch_research_landscape_input.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in batch_entries:
                f.write(json.dumps(entry) + "\n")

        print(f"Batch inference data saved to {output_path}")

        print(f"Total number of submissions: {len(batch_entries)}")
        if batch_entries:
            print(f"Sample content length: {len(batch_entries[0]['body']['messages'][0]['content'])} chars")

        return output_path

    def process_batch_results(self, results_file: str, data_dir: str) -> Dict:
        """
        Process the results from the research landscape batch and save them to the appropriate files.

        Args:
            results_file: Path to the batch results file (JSONL)
            data_dir: Directory to store processed data

        Returns:
            Dict containing processing statistics and information
        """
        from litellm import cost_per_token

        print(f"Processing research landscape batch results from {results_file}")

        # Statistics for reporting
        stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_cost": 0.0,
            "submissions": [],
        }

        # Load the results
        with open(results_file, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f]

        # Process each result
        for result in results:
            custom_id = result.get("custom_id", "")

            if not custom_id.startswith("research-landscape-"):
                print(f"Skipping non-landscape result: {custom_id}")
                continue

            stats["total_processed"] += 1

            # Extract submission_id from custom_id
            submission_id = custom_id.replace("research-landscape-", "")

            # Extract the analysis content
            response_data = result.get("response", {}).get("body", {})
            choices = response_data.get("choices", [])

            submission_stats = {"submission_id": submission_id, "status": "success"}

            if not choices:
                print(f"No content found for {submission_id}")
                stats["failed"] += 1
                submission_stats["status"] = "failed"
                submission_stats["error"] = "No content in choices"
                stats["submissions"].append(submission_stats)
                continue

            # Get the content from the first choice
            content = choices[0].get("message", {}).get("content", "")

            if not content:
                print(f"Empty content for {submission_id}")
                stats["failed"] += 1
                submission_stats["status"] = "failed"
                submission_stats["error"] = "Empty content"
                stats["submissions"].append(submission_stats)
                continue

            # Create submission directory if it doesn't exist
            submission_dir = os.path.join(data_dir, submission_id, "ours")
            os.makedirs(submission_dir, exist_ok=True)

            # Save the content to a file
            output_path = os.path.join(submission_dir, "research_landscape.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Saved research landscape analysis for {submission_id}")

            # Save metadata
            token_usage = response_data.get("usage", {})

            # Calculate cost
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            try:
                cost = sum(
                    cost_per_token(
                        model="gpt-4.1",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
                stats["total_cost"] += cost
                submission_stats["cost"] = cost
            except Exception as e:
                print(f"Error calculating cost for {submission_id}: {e}")
                submission_stats["cost_error"] = str(e)

            submission_stats["token_usage"] = token_usage

            # Update metadata file
            meta_path = os.path.join(submission_dir, "metadata.json")
            current_meta = {}

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        current_meta = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading metadata file for {submission_id}")

            current_meta["research_landscape"] = {
                "total_cost": submission_stats.get("cost", 0),
                "token_usage": token_usage,
            }

            with open(meta_path, "w") as f:
                json.dump(current_meta, f, indent=3)

            stats["successful"] += 1
            stats["submissions"].append(submission_stats)

        # Create summary statistics
        success_rate = (
            (stats["successful"] / stats["total_processed"]) * 100
            if stats["total_processed"] > 0
            else 0
        )
        print(
            f"Processed {stats['total_processed']} submissions with {stats['successful']} successes ({success_rate:.1f}%)"
        )
        print(f"Total estimated cost: ${stats['total_cost']:.4f}")

        # Save summary
        os.makedirs("./openai_outputs", exist_ok=True)
        summary_path = os.path.join(
            "./openai_outputs", "research_landscape_processing_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=3)

        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate research landscape analysis")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../novelty_assessment/data_iclr_all_topics/iclr_compiled_v2",
        help="Directory containing submission data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        help="Process single submission (for run_analysis)"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare batch inference data"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process batch results"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="./openai_outputs/batch_research_landscape_output.jsonl",
        help="Path to batch results file"
    )
    
    args = parser.parse_args()
    
    analyzer = ResearchLandscapeAnalyzer(model_name=args.model)
    
    if args.submission_id:
        # Single submission analysis
        analyzer.run_analysis(args.data_dir, args.submission_id, "research_landscape.txt")
    
    if args.prepare:
        # Prepare batch inference data
        analyzer.prepare_batch_inference_data(args.data_dir, args.model)
    
    if args.process:
        # Process batch results
        analyzer.process_batch_results(args.results_file, args.data_dir)
