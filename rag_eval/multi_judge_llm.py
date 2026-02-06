# rag_eval/multi_judge_llm.py
"""
Multi-Judge LLM Client for CATS Evaluation Pipeline
====================================================

Supports:
- Multiple judge models via OpenRouter API
- Majority voting and weighted aggregation
- Cost tracking and budget management
- Async batch processing with rate limiting
- Response caching for efficiency

Authors: Enhanced CATS Pipeline Team
Based on original work by: Gorang Mehrishi, Samyek Jain
"""

import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
from datetime import datetime

import httpx
from tqdm.asyncio import tqdm as tqdm_asyncio

from .judge_prompts import nli_prompt

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks API costs across models and provides budget alerts."""
    
    # Approximate costs per 1M tokens (February 2025 pricing)
    MODEL_COSTS = {
        "qwen/qwen-2.5-7b-instruct": {"input": 0.05, "output": 0.05},
        "qwen/qwen3-8b": {"input": 0.05, "output": 0.05},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.05},
        "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek/deepseek-chat-v3.1": {"input": 0.27, "output": 1.10},
        "deepseek/deepseek-r1-distill-qwen-32b": {"input": 0.14, "output": 0.28},
        "mistralai/mistral-8b": {"input": 0.05, "output": 0.05},
        "anthropic/claude-haiku-4.5": {"input": 0.80, "output": 4.00},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    
    def __init__(self, budget_limit: Optional[float] = None):
        self.usage = defaultdict(lambda: {"input": 0, "output": 0})
        self.budget_limit = budget_limit
        self.start_time = datetime.now()
        
    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Record token usage and calculate cost."""
        self.usage[model]["input"] += prompt_tokens
        self.usage[model]["output"] += completion_tokens
        
        current_cost = self.total_cost()
        
        if self.budget_limit and current_cost > self.budget_limit:
            logger.warning(
                f"⚠️  Budget limit exceeded! "
                f"Current cost: ${current_cost:.2f} / ${self.budget_limit:.2f}"
            )
    
    def total_cost(self) -> float:
        """Calculate total cost across all models."""
        cost = 0.0
        for model, usage in self.usage.items():
            if model in self.MODEL_COSTS:
                cost += (
                    usage["input"] / 1_000_000 * self.MODEL_COSTS[model]["input"] +
                    usage["output"] / 1_000_000 * self.MODEL_COSTS[model]["output"]
                )
            else:
                logger.warning(f"Unknown model for cost calculation: {model}")
        return cost
    
    def get_report(self) -> Dict:
        """Generate detailed cost report."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            "total_cost": self.total_cost(),
            "duration_seconds": duration,
            "cost_per_hour": self.total_cost() / (duration / 3600) if duration > 0 else 0,
            "per_model": {}
        }
        
        for model, usage in self.usage.items():
            if model in self.MODEL_COSTS:
                model_cost = (
                    usage["input"] / 1_000_000 * self.MODEL_COSTS[model]["input"] +
                    usage["output"] / 1_000_000 * self.MODEL_COSTS[model]["output"]
                )
                report["per_model"][model] = {
                    "input_tokens": usage["input"],
                    "output_tokens": usage["output"],
                    "total_tokens": usage["input"] + usage["output"],
                    "cost": model_cost
                }
        
        return report


class AsyncRateLimiter:
    """Token bucket rate limiter for async operations."""
    
    def __init__(self, rate: int, period: float = 60.0):
        """
        Args:
            rate: Maximum number of operations per period
            period: Time period in seconds (default: 60s)
        """
        self.rate = rate
        self.period = period
        self.allowance = rate
        self.last_check = datetime.now()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = datetime.now()
            time_passed = (now - self.last_check).total_seconds()
            
            # Replenish tokens
            self.allowance += time_passed * (self.rate / self.period)
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            self.last_check = now
            
            if self.allowance < 1.0:
                # Need to wait
                sleep_time = (1.0 - self.allowance) * (self.period / self.rate)
                await asyncio.sleep(sleep_time)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0


class ResponseCache:
    """Cache judge responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self._memory_cache = {}
    
    def _hash_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_or_compute(
        self,
        prompt: str,
        model: str,
        compute_fn: Callable
    ) -> Dict:
        """Get cached result or compute new one."""
        if not self.enabled:
            return await compute_fn()
        
        cache_key = self._hash_key(prompt, model)
        
        # Check memory cache
        if cache_key in self._memory_cache:
            logger.debug(f"Cache hit (memory): {model}")
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    result = json.load(f)
                    self._memory_cache[cache_key] = result
                    logger.debug(f"Cache hit (disk): {model}")
                    return result
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Compute new result
        result = await compute_fn()
        
        # Cache to memory and disk
        self._memory_cache[cache_key] = result
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
        
        return result


class OpenRouterClient:
    """Client for OpenRouter API supporting multiple models."""
    
    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        timeout: float = 60.0
    ):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 300
    ) -> tuple[str, Dict]:
        """
        Make async chat completion request.
        
        Returns:
            (response_text, usage_dict)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/CATS-Eval-Pipeline",
            "X-Title": "CATS Multi-Judge Evaluation"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                return (
                    data["choices"][0]["message"]["content"],
                    data.get("usage", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0
                    })
                )
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Rate limit hit for {model}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"HTTP error for {model}: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Request error for {model}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {self.max_retries} retries")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MultiJudgeLLM:
    """
    Multi-judge LLM evaluation client.
    
    Supports multiple judge models with aggregation strategies.
    """
    
    def __init__(
        self,
        judges: List[Dict[str, Any]],
        openrouter_api_key: str,
        aggregation_strategy: str = "majority_vote",
        min_judges: int = 2,
        confidence_threshold: float = 0.7,
        cost_tracker: Optional[CostTracker] = None,
        cache_dir: Optional[str] = None,
        rate_limit: int = 100  # requests per minute
    ):
        """
        Args:
            judges: List of judge configs, each with 'model' and optional 'weight'
            openrouter_api_key: API key for OpenRouter
            aggregation_strategy: "majority_vote", "weighted_vote", or "hierarchical"
            min_judges: Minimum number of successful judges required
            confidence_threshold: Minimum confidence for accepting result
            cost_tracker: Optional cost tracker instance
            cache_dir: Directory for caching responses
            rate_limit: Max requests per minute
        """
        self.judges = judges
        self.aggregation_strategy = aggregation_strategy
        self.min_judges = min_judges
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.client = OpenRouterClient(openrouter_api_key)
        self.cost_tracker = cost_tracker or CostTracker()
        self.cache = ResponseCache(cache_dir or "./cache") if cache_dir else ResponseCache("./cache", enabled=False)
        self.rate_limiter = AsyncRateLimiter(rate_limit)
        
        # Extract model IDs and weights
        self.judge_models = [j["model"] for j in judges]
        self.judge_weights = {
            j["model"]: j.get("weight", 1.0) for j in judges
        }
        
        logger.info(
            f"Initialized MultiJudgeLLM with {len(self.judges)} judges: "
            f"{', '.join(self.judge_models)}"
        )
    
    async def _call_single_judge(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 300
    ) -> Dict:
        """Call a single judge model and parse response."""
        
        async def _compute():
            await self.rate_limiter.acquire()
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an evaluation judge. Respond ONLY with JSON."
                },
                {"role": "user", "content": prompt}
            ]
            
            text, usage = await self.client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track costs
            self.cost_tracker.record(
                model=model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0)
            )
            
            return text
        
        # Use cache
        text = await self.cache.get_or_compute(prompt, model, _compute)
        
        # Parse JSON response
        try:
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object found")
            
            obj = json.loads(text[start:end+1])
            
            return {
                "judge_id": model,
                "adherent": bool(obj.get("adherent", False)),
                "rationale": str(obj.get("rationale", "")),
                "raw_response": text
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse response from {model}: {e}")
            return {
                "judge_id": model,
                "adherent": False,
                "rationale": f"Parse error: {e}",
                "error": True
            }
    
    async def ajudge_behavior(self, prompt: str) -> Dict:
        """
        Async behavior judgment using multiple judges.
        
        Returns:
            {
                "adherent": bool,
                "confidence": float,
                "rationale": str,
                "judge_results": List[Dict],
                "agreement_metrics": Dict
            }
        """
        # Call all judges in parallel
        tasks = [
            self._call_single_judge(model, prompt)
            for model in self.judge_models
        ]
        
        judge_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and errors
        valid_results = [
            r for r in judge_results
            if not isinstance(r, Exception) and not r.get("error", False)
        ]
        
        if len(valid_results) < self.min_judges:
            logger.error(
                f"Insufficient valid judges: {len(valid_results)} < {self.min_judges}"
            )
            return {
                "adherent": False,
                "confidence": 0.0,
                "rationale": "Insufficient judges responded",
                "judge_results": valid_results,
                "error": True
            }
        
        # Aggregate results
        if self.aggregation_strategy == "majority_vote":
            result = self._majority_vote(valid_results)
        elif self.aggregation_strategy == "weighted_vote":
            result = self._weighted_vote(valid_results)
        else:
            result = self._majority_vote(valid_results)  # fallback
        
        result["judge_results"] = valid_results
        result["agreement_metrics"] = self._compute_agreement(valid_results)
        
        return result
    
    def _majority_vote(self, judge_results: List[Dict]) -> Dict:
        """Simple majority voting across judges."""
        votes = [j["adherent"] for j in judge_results]
        vote_counts = Counter(votes)
        
        decision = max(vote_counts, key=vote_counts.get)
        confidence = vote_counts[decision] / len(votes)
        
        # Collect rationales from judges who voted with majority
        majority_rationales = [
            j["rationale"] for j in judge_results
            if j["adherent"] == decision
        ]
        
        return {
            "adherent": decision,
            "confidence": confidence,
            "rationale": "; ".join(majority_rationales[:2]),  # Top 2
            "vote_distribution": dict(vote_counts)
        }
    
    def _weighted_vote(self, judge_results: List[Dict]) -> Dict:
        """Weighted voting based on judge weights."""
        total_weight = sum(
            self.judge_weights.get(j["judge_id"], 1.0)
            for j in judge_results
        )
        
        weighted_sum = sum(
            self.judge_weights.get(j["judge_id"], 1.0) * (1 if j["adherent"] else 0)
            for j in judge_results
        )
        
        decision = weighted_sum / total_weight > 0.5
        confidence = abs(weighted_sum / total_weight - 0.5) * 2
        
        return {
            "adherent": decision,
            "confidence": confidence,
            "rationale": "; ".join([j["rationale"] for j in judge_results[:2]]),
            "weighted_score": weighted_sum / total_weight
        }
    
    def _compute_agreement(self, judge_results: List[Dict]) -> Dict:
        """Compute inter-judge agreement metrics."""
        decisions = [j["adherent"] for j in judge_results]
        
        if len(set(decisions)) == 1:
            # Perfect agreement
            return {
                "unanimous": True,
                "agreement_ratio": 1.0,
                "fleiss_kappa": 1.0
            }
        
        # Compute agreement ratio
        counts = Counter(decisions)
        majority_count = max(counts.values())
        agreement_ratio = majority_count / len(decisions)
        
        # Simple Fleiss' kappa approximation
        p_observed = agreement_ratio
        p_expected = sum((count / len(decisions)) ** 2 for count in counts.values())
        
        kappa = (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 1.0
        
        return {
            "unanimous": False,
            "agreement_ratio": agreement_ratio,
            "fleiss_kappa": kappa
        }
    
    # Synchronous wrapper for backward compatibility
    def judge_behavior(self, prompt: str) -> Dict:
        """Sync wrapper for behavior judgment."""
        try:
            return asyncio.run(self.ajudge_behavior(prompt))
        except RuntimeError:
            # Already in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.ajudge_behavior(prompt))
    
    async def anli_entailment(self, premise: str, hypothesis: str) -> str:
        """
        Async NLI entailment using multiple judges.
        
        Returns: "entails" | "contradicts" | "neutral"
        """
        prompt = nli_prompt(premise, hypothesis)
        
        # For NLI, we want to be conservative, so use all judges
        tasks = [
            self._call_nli_single(model, prompt)
            for model in self.judge_models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid results
        valid_results = [
            r for r in results
            if not isinstance(r, Exception) and r != "error"
        ]
        
        if not valid_results:
            return "neutral"  # Conservative fallback
        
        # Majority vote for NLI
        counts = Counter(valid_results)
        return max(counts, key=counts.get)
    
    async def _call_nli_single(self, model: str, prompt: str) -> str:
        """Call single judge for NLI task."""
        try:
            result = await self._call_single_judge(model, prompt)
            
            if result.get("error"):
                return "error"
            
            # Parse relation from response
            text = result.get("raw_response", "")
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                obj = json.loads(text[start:end+1])
                rel = obj.get("relation", "neutral")
                if rel in ("entails", "contradicts", "neutral"):
                    return rel
            
            return "neutral"
            
        except Exception as e:
            logger.warning(f"NLI error for {model}: {e}")
            return "error"
    
    # Sync wrapper for NLI
    def nli_entailment(self, premise: str, hypothesis: str) -> str:
        """Sync wrapper for NLI entailment."""
        try:
            return asyncio.run(self.anli_entailment(premise, hypothesis))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.anli_entailment(premise, hypothesis))
    
    async def aclose(self):
        """Close all connections."""
        await self.client.close()
    
    def get_cost_report(self) -> Dict:
        """Get cost tracking report."""
        return self.cost_tracker.get_report()
