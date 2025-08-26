import os
import argparse
from structured_extraction import StructuredRepresentationGenerator
from research_landscape import ResearchLandscapeAnalyzer
from novelty_assessment import NoveltyAssessor
from generate_summary import ReviewGuidanceGenerator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NoveltyAssessmentPipeline:
    """
    Complete pipeline for assessing paper novelty, from structured extraction to reviewer summary.
    """

    def __init__(self, model_name: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the pipeline with the specified LLM settings.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature setting for the LLM
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize all components
        self.extractor = StructuredRepresentationGenerator(model_name, temperature)
        self.analyzer = ResearchLandscapeAnalyzer(model_name, temperature)
        self.assessor = NoveltyAssessor(model_name, temperature)
        self.generator = ReviewGuidanceGenerator(model_name, temperature)

    def run_complete_pipeline(self, data_dir: str, submission_id: str) -> str:
        """
        Run the complete novelty assessment pipeline for a single submission.

        Args:
            data_dir: Directory containing paper information JSON files
            submission_id: ID of the submission paper

        Returns:
            Final reviewer guidance summary
        """
        print(f"Starting complete pipeline for submission: {submission_id}")
        
        # Step 1: Extract structured representations
        print("Step 1/4: Extracting structured representations...")
        self.extractor.process_single_submission(data_dir, submission_id)
        print("✓ Structured extraction completed")

        # Step 2: Generate research landscape analysis
        print("Step 2/4: Generating research landscape analysis...")
        self.analyzer.run_analysis(data_dir, submission_id, "research_landscape.txt")
        print("✓ Research landscape analysis completed")

        # Step 3: Assess novelty
        print("Step 3/4: Assessing novelty...")
        self.assessor.run_assessment(data_dir, submission_id)
        print("✓ Novelty assessment completed")

        # Step 4: Generate reviewer summary
        print("Step 4/4: Generating reviewer summary...")
        summary = self.generator.run_guidance_generation(data_dir, submission_id)
        print("✓ Reviewer summary completed")

        print(f"Pipeline completed successfully for {submission_id}")
        return summary

    def run_from_step(self, data_dir: str, submission_id: str, start_step: int) -> str:
        """
        Run pipeline starting from a specific step (useful if earlier steps are already completed).

        Args:
            data_dir: Directory containing paper information JSON files
            submission_id: ID of the submission paper
            start_step: Step number to start from (1-4)

        Returns:
            Final reviewer guidance summary
        """
        print(f"Starting pipeline from step {start_step} for submission: {submission_id}")
        
        if start_step <= 1:
            print("Step 1/4: Extracting structured representations...")
            self.extractor.process_single_submission(data_dir, submission_id)
            print("✓ Structured extraction completed")

        if start_step <= 2:
            print("Step 2/4: Generating research landscape analysis...")
            self.analyzer.run_analysis(data_dir, submission_id, "research_landscape.txt")
            print("✓ Research landscape analysis completed")

        if start_step <= 3:
            print("Step 3/4: Assessing novelty...")
            self.assessor.run_assessment(data_dir, submission_id)
            print("✓ Novelty assessment completed")

        if start_step <= 4:
            print("Step 4/4: Generating reviewer summary...")
            summary = self.generator.run_guidance_generation(data_dir, submission_id)
            print("✓ Reviewer summary completed")

        print(f"Pipeline completed successfully for {submission_id}")
        return summary

    def check_step_dependencies(self, data_dir: str, submission_id: str) -> dict:
        """
        Check which pipeline steps have been completed and which are missing.

        Args:
            data_dir: Directory containing paper information JSON files
            submission_id: ID of the submission paper

        Returns:
            Dict indicating status of each step
        """
        status = {
            "structured_extraction": False,
            "research_landscape": False,
            "novelty_assessment": False,
            "reviewer_summary": False
        }

        # Check for structured representation
        structured_path = os.path.join(data_dir, submission_id, "structured_representation.json")
        status["structured_extraction"] = os.path.exists(structured_path)

        # Check for research landscape
        landscape_path = os.path.join(data_dir, submission_id, "research_landscape.txt")
        status["research_landscape"] = os.path.exists(landscape_path)

        # Check for novelty assessment
        assessment_path = os.path.join(data_dir, submission_id, "novelty_delta_analysis.txt")
        status["novelty_assessment"] = os.path.exists(assessment_path)

        # Check for reviewer summary
        summary_path = os.path.join(data_dir, submission_id, "summary.txt")
        status["reviewer_summary"] = os.path.exists(summary_path)

        return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete novelty assessment pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../data",
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
        required=True,
        help="ID of the submission to process"
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Step to start from (1=extraction, 2=landscape, 3=assessment, 4=summary)"
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Check which steps have been completed"
    )
    
    args = parser.parse_args()
    
    pipeline = NoveltyAssessmentPipeline(model_name=args.model)
    
    if args.check_status:
        # Check status of pipeline steps
        status = pipeline.check_step_dependencies(args.data_dir, args.submission_id)
        print(f"Pipeline status for {args.submission_id}:")
        for step, completed in status.items():
            status_icon = "✓" if completed else "✗"
            print(f"  {status_icon} {step}")
    else:
        # Run pipeline
        if args.start_step == 1:
            pipeline.run_complete_pipeline(args.data_dir, args.submission_id)
        else:
            pipeline.run_from_step(args.data_dir, args.submission_id, args.start_step)