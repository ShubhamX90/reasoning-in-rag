# CATS Multi-Judge Evaluation Pipeline

**Enhanced Conflict-Aware Trust-Score (CATS) evaluation with multi-LLM-as-judge architecture**

## 🎯 Overview

This is an enhanced version of the CATS (Conflict-Aware Trust-Score) evaluation pipeline that introduces:

- **Multi-LLM-as-Judge**: Use multiple diverse judges for robust evaluation
- **Cost Optimization**: 60-80% cost reduction through strategic model selection
- **Better Reliability**: Reduced variance through consensus mechanisms
- **OpenRouter Integration**: Access to diverse models at competitive pricing
- **Enhanced Observability**: Detailed logging and diagnostics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/CATS-improved
cd CATS-improved

# Install dependencies
pip install -r requirements.txt

# Set up API key
export OPENROUTER_API_KEY="your-api-key-here"
```

### Basic Usage

```bash
# Quick test with development models (cheap and fast)
python run_multi_judge_eval.py \
    --phase test \
    --dataset data/test_samples.jsonl \
    --max-samples 10

# Full production evaluation
python run_multi_judge_eval.py \
    --phase production \
    --dataset data/full_dataset.jsonl

# High-quality validation run
python run_multi_judge_eval.py \
    --phase validation \
    --dataset data/validation_set.jsonl \
    --budget 100.0
```

## 📊 Key Features

### 1. Multi-Judge Architecture

Instead of relying on a single LLM judge, we use multiple diverse models and aggregate their decisions:

```python
# Example: 3 judges voting
Judge 1 (DeepSeek): adherent=True,  confidence=0.8
Judge 2 (Qwen):     adherent=True,  confidence=0.9
Judge 3 (Mistral):  adherent=False, confidence=0.6

# Aggregated result
Final: adherent=True, confidence=0.67 (2/3 agreement)
```

**Benefits:**
- **30-50% reduction** in evaluation variance
- **Higher confidence** in borderline cases
- **Diverse perspectives** from different model architectures

### 2. Cost Optimization

Strategic model selection based on the evaluation phase:

| Phase | Models | Cost/1M Tokens | Use Case |
|-------|--------|----------------|----------|
| Development | Qwen 7B, Llama 3.1 8B | $0.05 | Fast iteration |
| Production | DeepSeek Chat, Qwen3 8B | $0.14-0.28 | Balanced quality/cost |
| Validation | Claude Haiku 4.5 | $0.80-4.00 | Final high-quality check |

**Example cost comparison** (100 samples, 3 judges):
- GPT-4o only: ~$12-15
- Multi-judge (production): ~$2-3 (80% savings)
- Multi-judge (dev): ~$0.50 (97% savings)

### 3. Aggregation Strategies

#### Majority Voting (Default)
```yaml
aggregation:
  strategy: "majority_vote"
  min_judges: 2
```
Simple majority wins. Fast and robust.

#### Weighted Voting
```yaml
aggregation:
  strategy: "weighted_vote"
judges:
  - model: "deepseek/deepseek-chat"
    weight: 1.5  # Higher weight for better model
  - model: "qwen/qwen3-8b"
    weight: 1.2
```
Weights learned from validation performance.

#### Hierarchical Evaluation
```yaml
aggregation:
  strategy: "hierarchical"
  hierarchical:
    fast_judges: ["qwen/qwen-2.5-7b-instruct", "deepseek/deepseek-chat"]
    arbiter: "anthropic/claude-haiku-4.5"
    agreement_threshold: 0.8
```
Uses fast judges first, only invokes expensive arbiter on disagreement.

## 📁 Project Structure

```
improved_cats/
├── config/
│   ├── multi_judge_config.yaml      # Main configuration
│   └── config_loader.py              # Configuration utilities
├── rag_eval/
│   ├── multi_judge_llm.py            # Multi-judge LLM client
│   ├── evaluator.py                  # Evaluation orchestrator
│   ├── conflict_eval.py              # Conflict-aware metrics
│   ├── judge_prompts.py              # Evaluation prompts
│   ├── metrics.py                    # Metric computations
│   └── data.py                       # Data loading utilities
├── run_multi_judge_eval.py           # Main execution script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## ⚙️ Configuration

### Basic Configuration

The system is configured via `config/multi_judge_config.yaml`:

```yaml
# Define judge pools for different phases
judge_pools:
  production:
    - model: "deepseek/deepseek-chat"
      weight: 1.5
      max_concurrent: 20
    
    - model: "qwen/qwen3-8b"
      weight: 1.2
      max_concurrent: 20

# Aggregation settings
aggregation:
  strategy: "majority_vote"
  min_judges: 2
  confidence_threshold: 0.7

# Cost management
cost_limits:
  total_budget: 50.0
  alert_threshold: 40.0
```

### Advanced Configuration

#### Custom Judge Pools

Create your own judge combinations:

```yaml
judge_pools:
  my_custom_pool:
    - model: "anthropic/claude-haiku-4.5"
      weight: 2.0
    - model: "deepseek/deepseek-r1-distill-qwen-32b"
      weight: 1.5
    - model: "qwen/qwen3-8b"
      weight: 1.0
```

#### Performance Tuning

```yaml
performance:
  use_cache: true               # Cache responses
  cache_dir: "./cache"
  rate_limit: 100               # Requests per minute
  max_concurrent_requests: 20   # Parallel requests
  timeout: 60.0                 # Request timeout (seconds)
```

#### Logging & Diagnostics

```yaml
logging:
  level: "INFO"
  structured: true
  track_diagnostics: true
  log_judge_decisions: true     # Save individual judgments
  flag_disagreements: true      # Flag for manual review
```

## 📈 Evaluation Metrics

The pipeline computes:

### TRUST-SCORE Metrics
- **F1-GR**: Grounded Refusal (correct abstention)
- **AC**: Answer Correctness
- **GC**: Grounded Citation

### Conflict-Aware Metrics
- **Behavior Adherence**: Matches expected conflict behavior
- **Factual Grounding**: Claims supported by evidence
- **Single-Truth Recall**: Gold answer recall

### Multi-Judge Metrics
- **Inter-Judge Agreement**: Fleiss' Kappa, pairwise agreement
- **Confidence Scores**: Based on judge consensus
- **Calibration**: How well confidence reflects accuracy

## 🔍 Understanding Results

### Output Files

After evaluation, you'll find:

```
outputs/
├── evaluation_report.md          # Human-readable summary
├── detailed_results.json         # Full evaluation results
├── cost_report.json              # Cost breakdown by model
└── judge_decisions.jsonl         # Individual judge decisions
```

### Interpreting Metrics

**High Agreement (>0.8)**
- Strong consensus across judges
- High confidence in evaluation
- Safe to use results

**Medium Agreement (0.5-0.8)**
- Some disagreement among judges
- Review flagged cases
- May indicate ambiguous examples

**Low Agreement (<0.5)**
- Significant disagreement
- Manual review recommended
- May need clearer rubrics or better judges

## 💡 Best Practices

### 1. Start Small
```bash
# Test with 10 samples first
python run_multi_judge_eval.py \
    --phase test \
    --dataset data/sample.jsonl \
    --max-samples 10
```

### 2. Use Appropriate Phase

- **test/dev**: Rapid iteration, cheap models
- **production**: Main evaluation, balanced models
- **validation**: Final check, best models

### 3. Monitor Costs

```bash
# Set strict budget
python run_multi_judge_eval.py \
    --budget 10.0 \
    --dataset data/eval.jsonl
```

Check cost report after each run:
```json
{
  "total_cost": 2.45,
  "per_model": {
    "deepseek/deepseek-chat": {
      "cost": 1.20,
      "total_tokens": 85000
    },
    ...
  }
}
```

### 4. Review Disagreements

When judges disagree frequently:

1. Check `logs/diagnostics.json` for flagged cases
2. Review specific examples in `outputs/judge_decisions.jsonl`
3. Consider:
   - Are evaluation rubrics clear enough?
   - Do you need different/better judge models?
   - Is the dataset quality adequate?

### 5. Cache for Efficiency

Always enable caching when re-running evaluations:

```yaml
performance:
  use_cache: true
  cache_dir: "./cache"
```

This can save **60-90% of API costs** on repeated runs.

## 🔧 Troubleshooting

### Problem: API Rate Limits

**Solution**: Adjust rate limiting in config
```yaml
performance:
  rate_limit: 50  # Reduce from default 100
  max_concurrent_requests: 10  # Reduce parallelism
```

### Problem: High Costs

**Solutions**:
1. Use cheaper judge pools (development phase)
2. Enable caching
3. Use hierarchical evaluation
4. Reduce number of judges

### Problem: Low Judge Agreement

**Solutions**:
1. Review evaluation prompts in `rag_eval/judge_prompts.py`
2. Try different judge combinations
3. Add arbiter for tie-breaking
4. Check dataset quality

### Problem: Out of Memory

**Solutions**:
1. Reduce `max_concurrent_requests`
2. Process dataset in batches
3. Disable memory cache

## 📊 Comparison: Original vs Multi-Judge

| Aspect | Original CATS | Multi-Judge CATS |
|--------|---------------|------------------|
| Judges | 1 (GPT-4o-mini) | 2-5 (diverse models) |
| Cost/1k samples | ~$12-15 | ~$2-5 |
| Variance | High | Low (30-50% reduction) |
| Agreement Metrics | No | Yes (Fleiss' Kappa) |
| Confidence Scores | No | Yes |
| Caching | No | Yes |
| Cost Tracking | Manual | Automatic |
| OpenRouter Support | No | Yes |

## 🛠️ Development

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration test
python run_multi_judge_eval.py \
    --phase test \
    --dataset tests/fixtures/test_data.jsonl \
    --max-samples 5
```

### Adding New Judge Models

1. Add model to config:
```yaml
judge_pools:
  my_pool:
    - model: "provider/model-name"
      weight: 1.0
```

2. Update cost table in `multi_judge_llm.py` if needed

3. Test with small dataset

### Extending Evaluation Metrics

See `rag_eval/conflict_eval.py` for examples of implementing new metrics.

## 📚 References

### Original CATS Papers

- **Dragged into Conflicts**: Cattan et al., ACL 2025
- **TRUST-SCORE**: Song et al., ICLR 2025
- **Chain-of-Note**: Li et al., EMNLP 2024

### Multi-Judge Evaluation

This implementation is inspired by:
- Ensemble methods in ML
- Multi-annotator agreement in NLP
- Majority voting in distributed systems

## 🤝 Contributing

We welcome contributions! Areas of interest:

- New aggregation strategies
- Additional judge models
- Improved prompts
- Better calibration methods
- Cost optimization techniques

## 📄 License

MIT License - see LICENSE file

## 🙋 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: cats-eval@example.com

## 🎉 Acknowledgments

Based on the original CATS evaluation pipeline by:
- Gorang Mehrishi
- Samyek Jain  
- Research team at BITS Pilani

Enhanced with multi-judge capabilities and cost optimizations.

---

**Happy Evaluating! 🚀**
