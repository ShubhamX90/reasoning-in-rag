# 🚀 CATS Multi-Judge Pipeline - Quick Start Guide

## ⏱️ 5-Minute Setup

### 1. Get Your API Key (1 min)

Visit https://openrouter.ai and sign up for an account. You'll get $5 in free credits to start.

### 2. Install (1 min)

```bash
cd improved_cats
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 3. Test It (2 min)

```bash
# Download sample data from original CATS repo
# Or use your own JSONL file

python run_multi_judge_eval.py \
    --phase test \
    --dataset path/to/your/data.jsonl \
    --max-samples 5

# Expected output:
# ✅ Evaluation complete!
# 💰 Total cost: $0.02
# 📊 Agreement: 0.85
```

### 4. Run Full Evaluation (1 min to start)

```bash
python run_multi_judge_eval.py \
    --phase production \
    --dataset path/to/full/data.jsonl \
    --budget 10.0
```

## 📋 Checklist

- [ ] OpenRouter account created
- [ ] API key exported (`echo $OPENROUTER_API_KEY`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test run successful (5 samples)
- [ ] Full run started

## 🎯 What to Expect

### Test Run (5 samples)
- ⏰ Time: ~30 seconds
- 💰 Cost: $0.02-0.05
- 📊 Output: Basic metrics + cost report

### Production Run (500 samples)
- ⏰ Time: ~5-10 minutes
- 💰 Cost: $2-5 (vs $12-15 with original)
- 📊 Output: Full metrics + detailed reports

## 🔍 Troubleshooting

### "API key not found"
```bash
export OPENROUTER_API_KEY="your-key"
# OR
python run_multi_judge_eval.py --api-key "your-key" ...
```

### "Module not found"
```bash
pip install -r requirements.txt
# Make sure you're in the improved_cats directory
```

### "Budget exceeded"
```bash
# Increase budget or use cheaper phase
python run_multi_judge_eval.py --phase dev --budget 20.0 ...
```

## 📊 Understanding Your First Results

After your test run, you'll see:

```
📊 Overall Metrics:
  Samples evaluated: 5
  F1 Grounded Refusal: 1.000      ← Perfect refusal behavior
  Behavior Adherence: 0.800       ← 80% following expected behavior
  Factual Grounding: 0.750        ← 75% of claims grounded
  Single-Truth Recall: 1.000      ← Found all gold answers

💰 Cost Report:
  Total cost: $0.03               ← Super cheap!
  Duration: 28.4s                 ← Fast execution
  Cost per hour: $3.80/hr         ← Extrapolated rate

📈 Per-Model Costs:
  deepseek-chat:
    Tokens: 2,341                 ← Tokens used
    Cost: $0.02                   ← Cost for this model
  qwen3-8b:
    Tokens: 1,823
    Cost: $0.01
```

## ✅ Next Steps

1. **Review outputs/**
   - `evaluation_report.md` - Human-readable summary
   - `detailed_results.json` - Full metrics
   - `cost_report.json` - Cost breakdown

2. **Tune configuration**
   - Adjust judge models in `config/multi_judge_config.yaml`
   - Try different aggregation strategies
   - Set custom budgets

3. **Run at scale**
   - Use production phase for full dataset
   - Enable caching for repeated runs
   - Monitor costs and agreement

## 💡 Pro Tips

1. **Always start with `--phase test`** to verify everything works
2. **Enable caching** (`use_cache: true`) for development iterations
3. **Monitor agreement metrics** - low agreement might indicate issues
4. **Use hierarchical mode** for best cost/quality tradeoff
5. **Check logs/** if something goes wrong

## 📚 Learn More

- Full documentation: `README.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Configuration guide: `config/multi_judge_config.yaml`

---

**Need Help?** Open a GitHub issue or check the troubleshooting section in README.md

**Happy Evaluating! 🎉**
