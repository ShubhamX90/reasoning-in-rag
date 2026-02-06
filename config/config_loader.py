# improved_cats/config/config_loader.py
"""
Configuration Loader for Multi-Judge CATS Pipeline
==================================================

Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os


@dataclass
class JudgeConfig:
    """Configuration for a single judge model."""
    model: str
    weight: float = 1.0
    max_concurrent: int = 10
    description: str = ""


@dataclass
class AggregationConfig:
    """Configuration for result aggregation."""
    strategy: str = "majority_vote"
    min_judges: int = 2
    confidence_threshold: float = 0.7
    require_arbiter_on_tie: bool = True


@dataclass
class CostConfig:
    """Configuration for cost management."""
    total_budget: float = 50.0
    per_model_limit: float = 20.0
    alert_threshold: float = 40.0
    strict_budget: bool = False


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    use_cache: bool = True
    cache_dir: str = "./cache/judge_responses"
    cache_ttl_days: int = 30
    rate_limit: int = 100
    max_concurrent_requests: int = 20
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 60.0


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    structured: bool = True
    output_dir: str = "./logs"
    track_diagnostics: bool = True
    diagnostics_output: str = "./logs/diagnostics.json"
    log_judge_decisions: bool = True
    flag_disagreements: bool = True
    disagreement_threshold: float = 0.3


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    enable_conflict_eval: bool = True
    enable_trust_score: bool = True
    max_claims_per_answer: int = 12
    single_truth_types: list = field(default_factory=lambda: [1, 2, 4, 5])
    per_type_breakdown: bool = True


@dataclass
class OutputConfig:
    """Configuration for output files."""
    output_dir: str = "./outputs"
    report_md: str = "./outputs/evaluation_report.md"
    detailed_json: str = "./outputs/detailed_results.json"
    cost_report: str = "./outputs/cost_report.json"
    save_judge_decisions: bool = True
    judge_decisions_file: str = "./outputs/judge_decisions.jsonl"


@dataclass
class MultiJudgeConfig:
    """Master configuration for multi-judge evaluation."""
    judge_pools: Dict[str, Any] = field(default_factory=dict)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    cost_limits: CostConfig = field(default_factory=CostConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    phases: Dict[str, Any] = field(default_factory=dict)
    model_settings: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """Loads and validates multi-judge configuration."""
    
    @staticmethod
    def load(config_path: str) -> MultiJudgeConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            MultiJudgeConfig instance
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path) as f:
            raw_config = yaml.safe_load(f)
        
        # Build structured config
        config = MultiJudgeConfig()
        
        # Judge pools
        if "judge_pools" in raw_config:
            config.judge_pools = raw_config["judge_pools"]
        
        # Aggregation
        if "aggregation" in raw_config:
            agg = raw_config["aggregation"]
            config.aggregation = AggregationConfig(
                strategy=agg.get("strategy", "majority_vote"),
                min_judges=agg.get("min_judges", 2),
                confidence_threshold=agg.get("confidence_threshold", 0.7),
                require_arbiter_on_tie=agg.get("require_arbiter_on_tie", True)
            )
        
        # Cost limits
        if "cost_limits" in raw_config:
            cost = raw_config["cost_limits"]
            config.cost_limits = CostConfig(
                total_budget=cost.get("total_budget", 50.0),
                per_model_limit=cost.get("per_model_limit", 20.0),
                alert_threshold=cost.get("alert_threshold", 40.0),
                strict_budget=cost.get("strict_budget", False)
            )
        
        # Performance
        if "performance" in raw_config:
            perf = raw_config["performance"]
            config.performance = PerformanceConfig(
                use_cache=perf.get("use_cache", True),
                cache_dir=perf.get("cache_dir", "./cache/judge_responses"),
                cache_ttl_days=perf.get("cache_ttl_days", 30),
                rate_limit=perf.get("rate_limit", 100),
                max_concurrent_requests=perf.get("max_concurrent_requests", 20),
                max_retries=perf.get("max_retries", 3),
                retry_delay=perf.get("retry_delay", 2.0),
                timeout=perf.get("timeout", 60.0)
            )
        
        # Logging
        if "logging" in raw_config:
            log = raw_config["logging"]
            config.logging = LoggingConfig(
                level=log.get("level", "INFO"),
                structured=log.get("structured", True),
                output_dir=log.get("output_dir", "./logs"),
                track_diagnostics=log.get("track_diagnostics", True),
                diagnostics_output=log.get("diagnostics_output", "./logs/diagnostics.json"),
                log_judge_decisions=log.get("log_judge_decisions", True),
                flag_disagreements=log.get("flag_disagreements", True),
                disagreement_threshold=log.get("disagreement_threshold", 0.3)
            )
        
        # Evaluation
        if "evaluation" in raw_config:
            ev = raw_config["evaluation"]
            config.evaluation = EvaluationConfig(
                enable_conflict_eval=ev.get("enable_conflict_eval", True),
                enable_trust_score=ev.get("enable_trust_score", True),
                max_claims_per_answer=ev.get("max_claims_per_answer", 12),
                single_truth_types=ev.get("single_truth_types", [1, 2, 4, 5]),
                per_type_breakdown=ev.get("per_type_breakdown", True)
            )
        
        # Output
        if "output" in raw_config:
            out = raw_config["output"]
            config.output = OutputConfig(
                output_dir=out.get("output_dir", "./outputs"),
                report_md=out.get("report_md", "./outputs/evaluation_report.md"),
                detailed_json=out.get("detailed_json", "./outputs/detailed_results.json"),
                cost_report=out.get("cost_report", "./outputs/cost_report.json"),
                save_judge_decisions=out.get("save_judge_decisions", True),
                judge_decisions_file=out.get("judge_decisions_file", "./outputs/judge_decisions.jsonl")
            )
        
        # Phases
        if "phases" in raw_config:
            config.phases = raw_config["phases"]
        
        # Model settings
        if "model_settings" in raw_config:
            config.model_settings = raw_config["model_settings"]
        
        return config
    
    @staticmethod
    def get_judge_pool(
        config: MultiJudgeConfig,
        phase: str = "production"
    ) -> list[Dict[str, Any]]:
        """
        Get judge pool for specific phase.
        
        Args:
            config: MultiJudgeConfig instance
            phase: Phase name (e.g., "development", "production", "validation")
            
        Returns:
            List of judge configurations
        """
        # Get phase config
        phase_config = config.phases.get(phase, {})
        pool_name = phase_config.get("active_pool", phase)
        
        # Get judge pool
        if pool_name not in config.judge_pools:
            raise ValueError(f"Judge pool '{pool_name}' not found in configuration")
        
        return config.judge_pools[pool_name]
    
    @staticmethod
    def create_default_config(output_path: str):
        """Create a default configuration file."""
        default_config = {
            "judge_pools": {
                "development": [
                    {
                        "model": "qwen/qwen-2.5-7b-instruct",
                        "weight": 1.0,
                        "max_concurrent": 10
                    }
                ],
                "production": [
                    {
                        "model": "deepseek/deepseek-chat",
                        "weight": 1.5,
                        "max_concurrent": 20
                    },
                    {
                        "model": "qwen/qwen3-8b",
                        "weight": 1.2,
                        "max_concurrent": 20
                    }
                ]
            },
            "aggregation": {
                "strategy": "majority_vote",
                "min_judges": 2,
                "confidence_threshold": 0.7
            },
            "cost_limits": {
                "total_budget": 50.0,
                "alert_threshold": 40.0
            },
            "performance": {
                "use_cache": True,
                "rate_limit": 100
            }
        }
        
        with open(output_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)


def load_config_with_env_override(
    config_path: str,
    openrouter_api_key: Optional[str] = None
) -> tuple[MultiJudgeConfig, str]:
    """
    Load configuration and API key, with environment variable override.
    
    Args:
        config_path: Path to YAML config
        openrouter_api_key: Optional API key (overrides env var)
        
    Returns:
        (config, api_key)
    """
    config = ConfigLoader.load(config_path)
    
    # Get API key from param or environment
    api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OpenRouter API key not provided. "
            "Set OPENROUTER_API_KEY environment variable or pass as parameter."
        )
    
    return config, api_key
