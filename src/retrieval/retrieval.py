# Standard library imports
import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Third-party imports
import numpy as np
import requests
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

sys.path.append("/home/afzal/novelty_assessment/RankGPT")

from rank_gpt import (
    create_permutation_instruction,
    run_llm,
    receive_permutation,
    sliding_windows,
)


PURPOSE_PROMPT_PREFIX = [
    {
        "role": "system",
        "content": "You are RankGPT, an intelligent assistant that can rank scientific papers based on their relevancy to the idea of the source paper in the query.",
    },
    {
        "role": "user",
        "content": "I will provide you with {num} scinetific papers, each indicated by number identifier []. \nRank the passages based on their relevance to the source paper's idea: {query}.",
    },
    {"role": "assistant", "content": "Okay, please provide the passages."},
]
PURPOSE_PROMPT_POST = [
    {
        "role": "user",
        "content": "Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain.",
    }
]


@dataclass
class Paper:
    paper_id: str
    title: str = ""
    abstract: str = ""
    publication_date: str = ""
    venue: str = ""
    year: str = ""
    citation_count: int = 0
    authors: str = ""
    novel: str = None
    cited_paper: bool = False
    embedding: np.ndarray = None

    def __repr__(self):
        return f"Date: {self.publication_date}\nPaper ID: {self.paper_id}\nTitle: {self.title}\nPaper Date/Year: {self.publication_date if self.publication_date else (self.year if self.year else '')}"

    def to_dict(self):
        """Return a dictionary representation of the Paper object excluding the embedding."""
        data = asdict(self)
        data.pop("embedding", None)
        return data


@dataclass
class RankingResults:
    submission_id: str
    source_paper: Paper
    cited_papers: List[Paper]
    query_papers: List[Paper]
    final_ranked_papers: List[Paper]
    all_retrieved_papers: List[Paper]
    queries_used: List[str]
    total_cost: float
    general_ranking: List[Paper]
    purpose_ranking: List[Paper]

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "submission_id": self.submission_id,
            "source_paper": self.source_paper.to_dict(),
            "cited_papers": [p.to_dict() for p in self.cited_papers],
            "query_papers": [p.to_dict() for p in self.query_papers],
            "final_ranked_papers": [p.to_dict() for p in self.final_ranked_papers],
            "all_retrieved_papers": [p.to_dict() for p in self.all_retrieved_papers],
            "queries_used": self.queries_used,
            "total_cost": self.total_cost,
            "general_ranking": [p.to_dict() for p in self.general_ranking],
            "purpose_ranking": [p.to_dict() for p in self.purpose_ranking],
        }


class APIError(Exception):
    """Custom exception for API-related errors."""

    pass


class PaperRankingSystem:
    def __init__(
        self,
        keyword_model: str = "gpt-4o",
        ranking_model: str = "gpt-3.5-turbo",
        embedding_model: str = "allenai/specter2_base",
        results_dir: str = "results",
        log_dir: str = "logs",
    ):
        """
        Initialize the paper ranking system.

        Args:
            keyword_model: Model for keyword generation
            ranking_model: Model for RankGPT ranking
            embedding_model: Model for computing embeddings
            results_dir: Directory to save results
            log_dir: Directory to save log files
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._setup_logging()

        self.logger.info("Initializing PaperRankingSystem")
        self.logger.info(f"Keyword model: {keyword_model}")
        self.logger.info(f"Ranking model: {ranking_model}")
        self.logger.info(f"Embedding model: {embedding_model}")

        # Initialize models
        self.keyword_llm = ChatLiteLLM(model=keyword_model)
        self.ranking_model = ranking_model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Get API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        if not self.openai_api_key:
            self.logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.logger.info("PaperRankingSystem initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration for both file and console output."""
        # Create logger
        self.logger = logging.getLogger("PaperRankingSystem")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # File handler for detailed logs
        log_file = self.log_dir / f"paper_ranking_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging setup complete. Log file: {log_file}")

    def _make_request_with_retry(
        self, url: str, headers: Dict = None, max_retries: int = 3
    ) -> Dict:
        """Make HTTP request with exponential backoff retry logic."""
        self.logger.debug(f"Making request to: {url}")

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    self.logger.debug(f"Request successful on attempt {attempt + 1}")
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2**attempt) * 5  # 5, 10, 20 seconds
                    self.logger.warning(
                        f"Rate limited. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Other HTTP errors
                    self.logger.warning(f"HTTP {response.status_code}: {response.text}")
                    if attempt == max_retries - 1:
                        self.logger.error(
                            f"Failed after {max_retries} attempts: {response.status_code}"
                        )
                        raise APIError(
                            f"Failed after {max_retries} attempts: {response.status_code}"
                        )
                    time.sleep(2**attempt)

            except requests.RequestException as e:
                self.logger.warning(
                    f"Request exception on attempt {attempt + 1}: {str(e)}"
                )
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Request failed after {max_retries} attempts: {str(e)}"
                    )
                    raise APIError(
                        f"Request failed after {max_retries} attempts: {str(e)}"
                    )
                time.sleep(2**attempt)

        return {}

    def generate_search_queries(self, source_paper: Paper) -> Tuple[List[str], float]:
        """Generate search queries based on the source paper's content."""
        self.logger.info(f"Generating search queries for paper: {source_paper.title}")

        prompt = """Your task is to extract keywords about the IDEA of the paper from the provided abstract that can be queried on a search engine like semantic scholar for finding similar research papers, which match in main purpose of the idea.
        Don't give vague keywords like machine learning or computer science, but something specific to this IDEA, which will help to understand the novelty of this IDEA.
        Please provide 3-4 unique keywords without overlapping terms.
        Each keyword should be 2 words or less.
        Paper Title: {title}
        Paper Abstract: {abstract}
        """

        messages = [
            SystemMessage(
                content="You are KeywordGPT, an intelligent assistant that can identify relevant keywords for searching documents related to the idea of the paper."
            ),
            HumanMessage(
                content=prompt.format(
                    title=source_paper.title, abstract=source_paper.abstract
                )
            ),
        ]

        class KeywordOutput(BaseModel):
            list_of_keywords: List[str] = Field(description="List of keyword queries.")

        try:
            self.logger.debug("Calling LLM for keyword generation")
            response = self.keyword_llm.with_structured_output(
                KeywordOutput, include_raw=True
            ).invoke(messages)

            prompt_tokens = (
                response["raw"].response_metadata["token_usage"].prompt_tokens
            )
            completion_tokens = (
                response["raw"].response_metadata["token_usage"].completion_tokens
            )

            from litellm.cost_calculator import cost_per_token

            cost = sum(
                cost_per_token(
                    model=self.keyword_llm.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

            queries = response["parsed"].list_of_keywords
            self.logger.info(f"Generated {len(queries)} queries: {queries}")
            self.logger.debug(f"Keyword generation cost: ${cost:.4f}")
            return queries, cost

        except Exception as e:
            self.logger.error(f"Error generating keywords: {str(e)}")
            # Fallback to simple keywords extracted from title
            fallback_queries = source_paper.title.lower().split()[:4]
            self.logger.warning(
                f"Using fallback queries from title: {fallback_queries}"
            )
            return fallback_queries, 0.0

    def fetch_papers_from_semantic_scholar(
        self,
        queries: List[str],
        max_papers_per_query: int = 20,
        year: Optional[int] = None,
    ) -> List[Paper]:
        """Fetch papers from Semantic Scholar with retry logic."""
        self.logger.info(
            f"Fetching papers from Semantic Scholar for {len(queries)} queries"
        )
        all_papers = []

        headers = {}
        if self.s2_api_key:
            headers["X-API-KEY"] = self.s2_api_key
            self.logger.debug("Using Semantic Scholar API key")
        else:
            self.logger.warning("No Semantic Scholar API key found - using public API")

        for i, query in enumerate(queries):
            try:
                self.logger.debug(f"Processing query {i+1}/{len(queries)}: '{query}'")
                url = f"{self.base_url}/search?query={query}&fields=title,abstract,paperId,publicationDate,year&limit={max_papers_per_query}"
                if year:
                    url += f"&year=-{year}"
                    self.logger.debug(f"Filtering papers before year {year}")

                data = self._make_request_with_retry(url, headers)

                if "data" in data:
                    papers_count = len(data["data"])
                    self.logger.info(f"Query '{query}' returned {papers_count} papers")
                    for paper_data in data["data"]:
                        all_papers.append(
                            Paper(
                                paper_id=paper_data.get("paperId", "Unknown ID"),
                                title=paper_data.get("title", "Unknown Title"),
                                abstract=paper_data.get(
                                    "abstract", "No abstract available"
                                ),
                                publication_date=paper_data.get("publicationDate", ""),
                                year=paper_data.get("year", ""),
                            )
                        )
                else:
                    self.logger.warning(f"No data returned for query: '{query}'")

            except APIError as e:
                self.logger.error(
                    f"Failed to fetch papers for query '{query}': {str(e)}"
                )
                continue

        self.logger.info(f"Total papers retrieved: {len(all_papers)}")
        return all_papers

    def merge_paper_collections(
        self, cited_papers: List[Paper], query_papers: List[Paper], source_paper: Paper
    ) -> List[Paper]:
        """Merge papers from citations and query results, removing duplicates and invalid papers."""
        unique_papers = {}

        def is_valid_paper(paper: Paper, source_paper: Paper) -> bool:

            # Title similarity check
            if fuzz.ratio(paper.title.lower(), source_paper.title.lower()) >= 90:
                self.logger.debug(f"Skipping similar title: {paper.title}")
                return False

            # PRIORITY 1: If both have full dates, use precise date comparison
            if source_paper.publication_date and paper.publication_date:
                try:
                    pub_date = datetime.strptime(paper.publication_date, "%Y-%m-%d")
                    source_date = datetime.strptime(
                        source_paper.publication_date, "%Y-%m-%d"
                    )

                    # Allow same-year papers if they're >3 months old
                    days_difference = (source_date - pub_date).days
                    if days_difference < 90:  # Less than 3 months
                        self.logger.debug(
                            f"Skipping recent paper: {paper.publication_date} (only {days_difference} days before source)"
                        )
                        return False
                    return True

                except ValueError:
                    self.logger.debug(f"Invalid date format: {paper.publication_date}")
                    # Fall through to year-based logic

            # PRIORITY 2: If only years available, be conservative
            if source_paper.year and (paper.year or paper.publication_date):
                try:
                    pub_year = (
                        int(paper.year)
                        if paper.year
                        else int(paper.publication_date[:4])
                    )
                    if pub_year >= int(source_paper.year):
                        self.logger.debug(
                            f"Skipping paper from same/later year: {pub_year}"
                        )
                        return False
                    return True
                except ValueError:
                    self.logger.debug(f"Invalid year format: {paper.year}")

            # PRIORITY 3: No date information - reject
            self.logger.debug(f"Skipping paper with no date: {paper.title}")
            return False

        # Add cited papers (always include)
        for paper in cited_papers:
            paper.cited_paper = True  # Add this flag
            unique_papers[paper.paper_id] = paper

        # Add query papers if valid and not duplicate
        filtered_count = 0
        for paper in query_papers:
            if paper.paper_id not in unique_papers and is_valid_paper(
                paper, source_paper
            ):
                unique_papers[paper.paper_id] = paper
            else:
                filtered_count += 1

        self.logger.info(
            f"Merged papers: {len(cited_papers)} cited + {len(query_papers) - filtered_count} query papers = {len(unique_papers)} total"
        )
        self.logger.info(
            f"Filtered out {filtered_count} papers (duplicates/invalid dates/similar titles)"
        )

        return list(unique_papers.values())

    def compute_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Compute embeddings for multiple texts in batch for efficiency."""
        if not texts:
            return []

        self.logger.info(f"Computing embeddings for {len(texts)} texts")

        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            self.logger.warning("No valid texts found for embedding computation")
            return [np.zeros(768) for _ in texts]

        self.logger.debug(f"Computing embeddings for {len(valid_texts)} valid texts")

        # Compute embeddings for valid texts
        embeddings = self.embedding_model.encode(valid_texts, convert_to_numpy=True)

        # Create result array with zeros for empty texts
        result = []
        valid_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                result.append(embeddings[valid_idx])
                valid_idx += 1
            else:
                result.append(np.zeros(768))

        self.logger.info("Embedding computation completed")
        return result

    def compute_similarity_scores(
        self, source_embedding: np.ndarray, paper_embeddings: List[np.ndarray]
    ) -> List[float]:
        """Compute cosine similarity between source paper and retrieved papers."""
        if source_embedding is None or len(paper_embeddings) == 0:
            return []

        source_embedding = source_embedding.reshape(1, -1)
        paper_embeddings = np.array(paper_embeddings)
        similarity_scores = cosine_similarity(source_embedding, paper_embeddings)[0]
        return similarity_scores.tolist()

    def filter_top_n_papers(
        self, papers: List[Paper], similarity_scores: List[float], n: int = 100
    ) -> List[Paper]:
        """Filter the top N papers based on embedding similarity scores."""
        if not papers or not similarity_scores:
            return []

        paper_score_pairs = list(zip(papers, similarity_scores))
        paper_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [paper for paper, _ in paper_score_pairs[:n]]

    def rankgpt_rerank(
        self, source_paper: Paper, papers: List[Paper], ranking_type: str = "general"
    ) -> Tuple[List[Paper], float]:
        """Re-rank papers using RankGPT with error handling."""
        if not papers:
            return [], 0.0

        item = {
            "query": f"Title: {source_paper.title}\n Abstract: {source_paper.abstract}",
            "hits": [
                {"content": f"Title: {paper.title}\n Abstract: {paper.abstract}"}
                for paper in papers
            ],
        }

        self.logger.info(f"Re-ranking {len(papers)} papers using {ranking_type} method")

        try:
            if ranking_type == "general":
                self.logger.debug("Using sliding windows approach for general ranking")
                ranked_item_list, total_cost = sliding_windows(item)
            else:
                self.logger.debug("Using purpose-based ranking approach")
                # Purpose-based ranking
                import copy

                rank_start = 0
                rank_end = len(papers)
                window_size = 20
                ranked_item_list = copy.deepcopy(item)
                end_pos = rank_end
                start_pos = rank_end - window_size
                step = 10
                total_cost = 0.0

                while start_pos >= rank_start:
                    start_pos = max(start_pos, rank_start)
                    self.logger.debug(
                        f"Processing ranking window: {start_pos}-{end_pos}"
                    )

                    messages = create_permutation_instruction(
                        item=ranked_item_list,
                        rank_start=start_pos,
                        rank_end=end_pos,
                        model_name=self.ranking_model,
                    )
                    messages = (
                        PURPOSE_PROMPT_PREFIX + messages[3:-1] + PURPOSE_PROMPT_POST
                    )

                    permutation, cost = run_llm(
                        messages,
                        model_name=self.ranking_model,
                        api_key=self.openai_api_key,
                    )

                    ranked_item_list = receive_permutation(
                        item, permutation, rank_start=start_pos, rank_end=end_pos
                    )

                    end_pos = end_pos - step
                    start_pos = start_pos - step
                    total_cost += cost

            # Reconstruct paper list from ranked results
            reranked_papers_list = []
            failed_matches = 0
            for r_paper in ranked_item_list["hits"]:
                match = re.search(
                    r"(?<=Title: ).*?(?=\n Abstract:)", r_paper["content"]
                )
                if match:
                    title = match.group(0)
                    for paper in papers:
                        if paper.title == title:
                            reranked_papers_list.append(paper)
                            break
                else:
                    failed_matches += 1
                    self.logger.warning(
                        "Failed to match paper title in ranking results"
                    )

            if failed_matches > 0:
                self.logger.warning(
                    f"Failed to match {failed_matches} papers in ranking results"
                )

            self.logger.info(
                f"{ranking_type.capitalize()} ranking completed. Cost: ${total_cost:.4f}. {len(reranked_papers_list)} papers returned"
            )
            return reranked_papers_list, total_cost

        except Exception as e:
            self.logger.error(f"Error in RankGPT ranking ({ranking_type}): {str(e)}")
            self.logger.warning("Returning original paper order due to ranking failure")
            return papers, 0.0  # Return original order if ranking fails

    def combine_rankings(
        self, general_ranked: List[Paper], purpose_ranked: List[Paper], k: int = 10
    ) -> List[Paper]:
        """Combine the top-k papers from both rankings."""
        unique_papers = {}

        # Add top-k papers from general ranking
        for paper in general_ranked[:k]:
            unique_papers[paper.paper_id] = paper

        # Add top-k papers from purpose ranking
        for paper in purpose_ranked[:k]:
            if paper.paper_id not in unique_papers:
                unique_papers[paper.paper_id] = paper

        return list(unique_papers.values())

    def save_results(self, results: RankingResults) -> None:
        """Save results directly to results_dir."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving results to {self.results_dir}")

        try:
            # Save final ranked papers
            with open(self.results_dir / "ranked_papers.json", "w") as f:
                json.dump(
                    [p.to_dict() for p in results.final_ranked_papers], f, indent=2
                )

            # Save all retrieved papers
            with open(self.results_dir / "all_retrieved_papers.json", "w") as f:
                json.dump(
                    [p.to_dict() for p in results.all_retrieved_papers], f, indent=2
                )

            # Save metadata
            metadata = {
                "submission_id": results.submission_id,
                "queries_used": results.queries_used,
                "total_cost": results.total_cost,
                "num_cited_papers": len(results.cited_papers),
                "num_query_papers": len(results.query_papers),
                "num_final_ranked": len(results.final_ranked_papers),
                "num_all_retrieved": len(results.all_retrieved_papers),
            }
            with open(self.results_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Save complete results
            with open(self.results_dir / "complete_results.json", "w") as f:
                json.dump(results.to_dict(), f, indent=2)

            self.logger.info(f"Results saved successfully to {self.results_dir}")

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def count_cited_papers(self, papers: List[Paper]) -> Tuple[int, int]:
        """Count cited vs non-cited papers in a list."""
        cited_count = sum(1 for p in papers if getattr(p, "cited_paper", False))
        non_cited_count = len(papers) - cited_count
        return cited_count, non_cited_count

    def find_similar_papers(
        self,
        source_paper: Paper,
        cited_papers: List[Paper],
        submission_id: str,
        max_papers_per_query: int = 20,
        top_n_similarity: int = 100,
        combination_k: int = 10,
    ) -> RankingResults:
        """
        Main pipeline to find papers similar to the source paper.

        Args:
            source_paper: Paper object containing the paper to compare against
            cited_papers: List of papers cited by the source paper
            submission_id: Unique identifier for this submission
            max_papers_per_query: Maximum papers to retrieve per search query
            top_n_similarity: Number of papers to keep after similarity filtering
            combination_k: Number of top papers to take from each ranking

        Returns:
            RankingResults object containing all results and metadata
        """
        self.logger.info(f"Starting pipeline for submission: {submission_id}")
        self.logger.info(f"Source paper: {source_paper.title}")
        self.logger.info(
            f"Parameters - max_papers_per_query: {max_papers_per_query}, top_n_similarity: {top_n_similarity}, combination_k: {combination_k}"
        )

        total_cost = 0.0

        try:
            # Step 1: Generate search queries
            self.logger.info("Step 1: Generating search queries")
            queries, cost = self.generate_search_queries(source_paper)
            total_cost += cost

            # Step 2: Fetch papers from Semantic Scholar
            self.logger.info("Step 2: Fetching papers from Semantic Scholar")
            year = int(source_paper.year) if source_paper.year else None
            query_papers = self.fetch_papers_from_semantic_scholar(
                queries, max_papers_per_query, year
            )

            # Step 3: Merge paper collections
            self.logger.info("Step 3: Merging paper collections")
            all_papers = self.merge_paper_collections(
                cited_papers, query_papers, source_paper
            )

            # Step 4: Compute embeddings in batch
            self.logger.info("Step 4: Computing embeddings")
            all_abstracts = [source_paper.abstract] + [p.abstract for p in all_papers]
            embeddings = self.compute_embeddings_batch(all_abstracts)

            source_paper.embedding = embeddings[0]
            for i, paper in enumerate(all_papers):
                paper.embedding = embeddings[i + 1]

            # Step 5: Filter top N papers using similarity
            self.logger.info("Step 5: Filtering papers by similarity")
            similarity_scores = self.compute_similarity_scores(
                source_paper.embedding, [p.embedding for p in all_papers]
            )

            top_n_papers = self.filter_top_n_papers(
                all_papers, similarity_scores, top_n_similarity
            )
            self.logger.info(
                f"Filtered to top {len(top_n_papers)} papers by similarity"
            )

            # Step 6: Perform RankGPT ranking
            self.logger.info("Step 6: Performing RankGPT ranking")
            general_ranked, cost1 = self.rankgpt_rerank(
                source_paper, top_n_papers, "general"
            )
            total_cost += cost1

            purpose_ranked, cost2 = self.rankgpt_rerank(
                source_paper, top_n_papers, "purpose"
            )
            total_cost += cost2

            # Step 7: Combine rankings
            self.logger.info("Step 7: Combining rankings")
            final_ranked = self.combine_rankings(
                general_ranked, purpose_ranked, combination_k
            )
            self.logger.info(f"Final ranking contains {len(final_ranked)} papers")
            cited_count, non_cited_count = self.count_cited_papers(final_ranked)
            self.logger.info(
                f"Final ranking: {cited_count} cited + {non_cited_count} non-cited = {len(final_ranked)} total"
            )

            # Create results object
            results = RankingResults(
                submission_id=submission_id,
                source_paper=source_paper,
                cited_papers=cited_papers,
                query_papers=query_papers,
                final_ranked_papers=final_ranked,
                all_retrieved_papers=all_papers,
                queries_used=queries,
                total_cost=total_cost,
                general_ranking=general_ranked,
                purpose_ranking=purpose_ranked,
            )

            # Print final summary
            print(f"\n=== FINAL RANKING SUMMARY ===")
            print(f"Total papers in final ranking: {len(final_ranked)}")
            print(f"Cited papers: {cited_count}")
            print(f"Non-cited papers: {non_cited_count}")
            print(
                f"Percentage cited: {(cited_count/len(final_ranked)*100):.1f}%"
                if final_ranked
                else "0%"
            )

            # Save results
            self.save_results(results)

            self.logger.info(
                f"Pipeline completed successfully. Total cost: ${total_cost:.4f}"
            )
            return results

        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            # Save partial results if possible
            self.logger.info("Attempting to save partial results")
            try:
                partial_results = RankingResults(
                    submission_id=submission_id,
                    source_paper=source_paper,
                    cited_papers=cited_papers,
                    query_papers=query_papers if "query_papers" in locals() else [],
                    final_ranked_papers=[],
                    all_retrieved_papers=all_papers if "all_papers" in locals() else [],
                    queries_used=queries if "queries" in locals() else [],
                    total_cost=total_cost,
                    general_ranking=[],
                    purpose_ranking=[],
                )
                self.save_results(partial_results)
                self.logger.info("Partial results saved successfully")
            except Exception as save_error:
                self.logger.error(f"Failed to save partial results: {str(save_error)}")

            raise


def process_for_pipeline(data_dir: str, submission_id: str) -> bool:
    """
    Process a single submission for pipeline integration.
    
    Args:
        data_dir: Base data directory for pipeline
        submission_id: ID of the submission
    
    Returns:
        Success status
    """
    # Load submission data
    input_file = Path(data_dir) / submission_id / f"{submission_id}.json"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        submission_data = json.load(f)
    
    # Initialize ranking system
    ranking_system = PaperRankingSystem(
        keyword_model="gpt-4o",
        ranking_model="gpt-3.5-turbo",
        embedding_model="allenai/specter2_base",
        results_dir=str(Path(data_dir) / submission_id / "related_work_data"),
        log_dir=str(Path(data_dir) / submission_id / "logs"),
    )
    
    try:
        # Create source paper
        source_paper = Paper(
            paper_id=submission_id,
            title=submission_data["title"],
            abstract=submission_data.get("abstract", ""),
            year="2024",
            publication_date="2024-10-01",
        )
        
        # Process cited papers
        valid_cited_papers = []
        for cited_paper_data in submission_data.get("cited_papers", []):
            if cited_paper_data.get("ss_paper_obj"):
                ss_paper_obj = cited_paper_data["ss_paper_obj"].copy()
                ss_paper_obj.pop("citations", None)
                ss_paper_obj["cited_paper"] = True
                
                try:
                    cited_paper = Paper(**ss_paper_obj)
                    valid_cited_papers.append(cited_paper)
                except Exception as e:
                    ranking_system.logger.warning(f"Failed to create Paper object: {e}")
                    continue
        
        if not valid_cited_papers:
            ranking_system.logger.warning(f"No valid cited papers found for {submission_id}")
            return False
        
        # Run the pipeline
        ranking_system.find_similar_papers(
            source_paper=source_paper,
            cited_papers=valid_cited_papers,
            submission_id=submission_id,
            max_papers_per_query=20,
            top_n_similarity=100,
            combination_k=15,
        )
        
        ranking_system.logger.info(f"Successfully processed {submission_id}")
        return True
        
    except Exception as e:
        ranking_system.logger.error(f"Error processing {submission_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Paper retrieval and ranking system - single submission mode only")
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
        "--keyword-model",
        type=str,
        default="gpt-4o",
        help="Model for keyword generation"
    )
    parser.add_argument(
        "--ranking-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model for paper ranking"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="allenai/specter2_base",
        help="Embedding model for similarity"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Process single submission
    success = process_for_pipeline(args.data_dir, args.submission_id)
    if success:
        print(f"✅ Successfully processed submission {args.submission_id}")
    else:
        print(f"❌ Failed to process submission {args.submission_id}")


if __name__ == "__main__":
    main()
