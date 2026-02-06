#!/usr/bin/env python3
"""
run_multi_judge_eval.py
=======================

Example script for running CATS evaluation with multi-judge setup.

Usage:
    # Quick test with development models
    python run_multi_judge_eval.py --phase test --dataset data/test_samples.jsonl
    
    # Full production evaluation
    python run_multi_judge_eval.py --phase production --dataset data/full_dataset.jsonl
    
    # Validation with best models
    python run_multi_judge_eval.py --phase validation --dataset data/validation_set.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

# Add improved CATS to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import load_config_with_env_override, ConfigLoader
from rag_eval.multi_judge_llm import MultiJudgeLLM, CostTracker
from rag_eval.data import load_dataset
from rag_eval.evaluator import Evaluator
from rag_eval.config import EvaluationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CATS evaluation with multi-judge setup"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/multi_judge_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to input dataset (JSONL format)"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        choices=["test", "dev", "production", "validation"],
        default="production",
        help="Evaluation phase (determines judge pool)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--budget",
        type=float,
        help="Override budget limit from config (USD)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of samples to evaluate (for testing)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable response caching"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main():
    """Main evaluation workflow."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config, api_key = load_config_with_env_override(
            args.config,
            args.api_key
        )
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Apply command-line overrides
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    if args.budget:
        config.cost_limits.total_budget = args.budget
    
    if args.no_cache:
        config.performance.use_cache = False
    
    # Get judge pool for phase
    logger.info(f"Evaluation phase: {args.phase}")
    judge_pool = ConfigLoader.get_judge_pool(config, args.phase)
    
    logger.info(f"Using {len(judge_pool)} judge models:")
    for judge in judge_pool:
        logger.info(f"  - {judge['model']} (weight: {judge.get('weight', 1.0)})")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    try:
        dataset = load_dataset(args.dataset)
        logger.info(f"Loaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Limit samples if requested
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        logger.info(f"Limited to {len(dataset)} samples for testing")
    
    # Initialize cost tracker
    cost_tracker = CostTracker(budget_limit=config.cost_limits.total_budget)
    
    # Initialize multi-judge LLM
    logger.info("Initializing multi-judge LLM...")
    multi_judge_llm = MultiJudgeLLM(
        judges=judge_pool,
        openrouter_api_key=api_key,
        aggregation_strategy=config.aggregation.strategy,
        min_judges=config.aggregation.min_judges,
        confidence_threshold=config.aggregation.confidence_threshold,
        cost_tracker=cost_tracker,
        cache_dir=config.performance.cache_dir if config.performance.use_cache else None,
        rate_limit=config.performance.rate_limit
    )
    
    # Create evaluator config
    eval_cfg = EvaluationConfig(
        output_dir=config.output.output_dir,
        report_md=config.output.report_md
    )
    eval_cfg.conflict.enable_conflict_eval = config.evaluation.enable_conflict_eval
    eval_cfg.conflict.max_claims_per_answer = config.evaluation.max_claims_per_answer
    eval_cfg.conflict.single_truth_types = tuple(config.evaluation.single_truth_types)
    
    # Create output directory
    Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    logger.info("=" * 60)
    
    evaluator = Evaluator(eval_cfg, multi_judge_llm)
    
    try:
        results = evaluator.evaluate(dataset)
        
        logger.info("=" * 60)
        logger.info("Evaluation complete!")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        if "conflict_overall" in results:
            overall = results["conflict_overall"]
            print(f"\n📊 Overall Metrics:")
            print(f"  Samples evaluated: {overall['n']}")
            print(f"  F1 Grounded Refusal: {overall['f1_gr']:.3f}")
            print(f"  Behavior Adherence: {overall['behavior']:.3f}")
            print(f"  Factual Grounding: {overall['factual_grounding']:.3f}")
            print(f"  Single-Truth Recall: {overall['single_truth_recall']:.3f}")
        
        # Cost report
        cost_report = cost_tracker.get_report()
        print(f"\n💰 Cost Report:")
        print(f"  Total cost: ${cost_report['total_cost']:.2f}")
        print(f"  Duration: {cost_report['duration_seconds']:.1f}s")
        print(f"  Cost per hour: ${cost_report['cost_per_hour']:.2f}/hr")
        
        print(f"\n📈 Per-Model Costs:")
        for model, stats in cost_report['per_model'].items():
            model_name = model.split('/')[-1]
            print(f"  {model_name}:")
            print(f"    Tokens: {stats['total_tokens']:,}")
            print(f"    Cost: ${stats['cost']:.2f}")
        
        # Save detailed results
        detailed_output = Path(config.output.detailed_json)
        with open(detailed_output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n📄 Detailed results saved to: {detailed_output}")
        
        # Save cost report
        cost_output = Path(config.output.cost_report)
        with open(cost_output, 'w') as f:
            json.dump(cost_report, f, indent=2)
        logger.info(f"💰 Cost report saved to: {cost_output}")
        
        # Report path
        if config.output.report_md:
            logger.info(f"📝 Markdown report: {config.output.report_md}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Cleanup
        await multi_judge_llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
