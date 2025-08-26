import os
import json
from typing import Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

reviewer_summary_prompt = """
Summarize the following assessment in 5 sentences for a reviewer reviewing at an AI conference.

## Delta Assessment
{novelty_assessment}
"""


class ReviewGuidanceGenerator:
    """
    Generates targeted guidance for reviewers based on novelty assessment and research landscape analysis.
    """

    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the review guidance generator with the specified LLM.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature setting for the LLM
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    def load_novelty_assessment(self, assessment_file: str) -> str:
        """
        Load the novelty assessment from file.

        Args:
            assessment_file: Path to the novelty assessment file

        Returns:
            Content of the novelty assessment
        """
        if not os.path.exists(assessment_file):
            raise ValueError(f"Novelty assessment file not found: {assessment_file}")

        with open(assessment_file, "r", encoding="utf-8") as f:
            assessment = f.read()

        return assessment


    def generate_guidance(
        self, novelty_assessment: str
    ) -> str:
        """
        Generate reviewer guidance based on novelty assessment and landscape analysis.

        Args:
            novelty_assessment: Novelty assessment of the submission paper
            landscape_analysis: Research landscape analysis

        Returns:
            Reviewer guidance
        """
        # Generate the summary
        prompt = reviewer_summary_prompt.format(novelty_assessment=novelty_assessment)

        guidance = self.llm.invoke(prompt)

        return guidance

    def run_guidance_generation(
        self, data_dir: str, submission_id: str
    ) -> str:
        """
        Run the complete review guidance generation pipeline.

        Args:
            data_dir: Directory containing paper information JSON files
            submission_id: ID of the submission paper

        Returns:
            Reviewer guidance
        """
        # Load novelty assessment (standard path)
        assessment_file = os.path.join(data_dir, submission_id, "novelty_delta_analysis.txt")
        novelty_assessment = self.load_novelty_assessment(assessment_file)

        # Generate guidance
        guidance = self.generate_guidance(novelty_assessment)

        # Save guidance to file (standard path)
        output_file = os.path.join(data_dir, submission_id, "summary.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(guidance.content)

        print(f"Review guidance saved to summary.txt")

        from litellm import cost_per_token

        tok_used = guidance.response_metadata["token_usage"]
        total_cost = sum(
            cost_per_token(
                model="gpt-4.1",
                prompt_tokens=tok_used["prompt_tokens"],
                completion_tokens=tok_used["completion_tokens"],
            )
        )

        # Extract data_dir from output_file path to construct metadata path
        data_dir = os.path.dirname(os.path.dirname(output_file))
        meta_path = os.path.join(data_dir, "metadata.json")
        
        current_meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                current_meta = json.load(f)

        current_meta["summary"] = {"total_cost": total_cost, "token_usage": tok_used}

        with open(meta_path, "w") as f:
            json.dump(current_meta, f, indent=3)

        return guidance

    def prepare_batch_inference_data(
        self,
        data_dir: str,
        assessment_file: str = "novelty_delta_analysis.txt",
        model_name: str = "gpt-4.1",
    ) -> str:
        """
        Prepare data for batched inference with OpenAI API for review guidance generation.

        Args:
            data_dir: Directory containing paper information and assessments
            submission_ids: List of submission paper IDs to process
            assessment_file: Name of the novelty assessment file
            landscape_file: Name of the landscape analysis file
            model_name: Name of the OpenAI model to use

        Returns:
            Path to the generated JSONL batch file
        """
        batch_entries = []
        submission_ids = os.listdir(data_dir)

        # Statistics for reporting
        total_submissions = len(submission_ids)
        valid_submissions = 0
        missing_assessment = 0
        missing_landscape = 0

        for submission_id in submission_ids:
            try:
                # Check if required files exist
                assessment_path = os.path.join(
                    data_dir, submission_id, "ours", assessment_file
                )
                if not os.path.exists(assessment_path):
                    print(f"Missing novelty assessment for {submission_id}")
                    missing_assessment += 1
                    continue

                # Load novelty assessment
                novelty_assessment = self.load_novelty_assessment(assessment_path)

                # Prepare the prompt
                prompt = reviewer_summary_prompt.format(
                    novelty_assessment=novelty_assessment
                )

                # Create the batch entry
                batch_entry = {
                    "custom_id": f"reviewer-guidance-{submission_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                    },
                }

                batch_entries.append(batch_entry)
                valid_submissions += 1

            except Exception as e:
                print(f"Error preparing batch data for {submission_id}: {e}")

        # Report statistics
        print(
            f"Valid submissions for review guidance: {valid_submissions}/{total_submissions} ({valid_submissions/total_submissions*100:.1f}%)"
        )
        if missing_assessment > 0:
            print(f"Submissions missing novelty assessment: {missing_assessment}")
        if missing_landscape > 0:
            print(f"Submissions missing landscape analysis: {missing_landscape}")

        # Create output directory if it doesn't exist
        os.makedirs("./openai_inputs", exist_ok=True)
        
        # Write to JSONL file
        output_path = os.path.join(
            "./openai_inputs", "batch_reviewer_guidance_input.jsonl"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in batch_entries:
                f.write(json.dumps(entry) + "\n")

        print(
            f"Batch inference data saved to {output_path} with {len(batch_entries)} total entries"
        )
        return output_path

    def process_batch_results(
        self,
        data_dir: str,
        results_file: str,
        output_file: str = "reviewer_guidance.txt",
    ) -> Dict:
        """
        Process the results from batched inference for review guidance generation.

        Args:
            data_dir: Directory to store processed data
            results_file: Path to the batch results file (JSONL)
            output_file: Name of the output file for each submission

        Returns:
            Dict with processing statistics
        """
        from litellm import cost_per_token

        print(f"Processing review guidance batch results from {results_file}")

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

            if not custom_id.startswith("reviewer-guidance-"):
                print(f"Skipping non-guidance result: {custom_id}")
                continue

            stats["total_processed"] += 1

            # Extract submission_id from custom_id
            submission_id = custom_id.replace("reviewer-guidance-", "")

            # Extract the guidance content
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

            # Save the content to a file
            submission_dir = os.path.join(data_dir, submission_id, "ours")
            os.makedirs(submission_dir, exist_ok=True)

            output_path = os.path.join(submission_dir, output_file)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"Saved review guidance for {submission_id}")

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

            current_meta["summary"] = {
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

        # Create output directory if it doesn't exist
        os.makedirs("./openai_outputs", exist_ok=True)
        
        # Save summary
        summary_path = os.path.join(
            "./openai_outputs", "reviewer_guidance_processing_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=3)

        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reviewer guidance summaries")
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
        help="Process single submission (for run_guidance_generation)"
    )
    parser.add_argument(
        "--assessment-file",
        type=str,
        default="novelty_delta_analysis.txt",
        help="Path to novelty assessment file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="summary.txt",
        help="Output file for summary"
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
        default="./openai_outputs/batch_reviewer_guidance_output.jsonl",
        help="Path to batch results file"
    )
    
    args = parser.parse_args()
    
    generator = ReviewGuidanceGenerator(model_name=args.model)
    
    if args.submission_id:
        # Single submission processing
        generator.run_guidance_generation(
            args.data_dir,
            args.submission_id
        )
    
    if args.prepare:
        # Prepare batch inference data
        generator.prepare_batch_inference_data(
            args.data_dir,
            args.assessment_file,
            args.model
        )
    
    if args.process:
        # Process batch results
        generator.process_batch_results(
            args.data_dir,
            args.results_file,
            args.output_file
        )
