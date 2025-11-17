# Path Engineering Research: Causal Manipulation of the "Road Not Taken"

**Research Domain**: Natural Language Processing / Mechanistic Interpretability
**Date**: November 16, 2025
**Status**: ✅ Complete

---

## Quick Summary

This research investigated whether manipulating the "path space" in language model hidden representations causally affects output uncertainty. We tested this by applying PCA dimensionality reduction and noise injection to GPT-2's activations during reasoning tasks.

### 🔑 Key Findings

1. **PCA dimensionality reduction INCREASED uncertainty by 36-52%** (p < 0.001, Cohen's d > 1.2)
   - Contrary to hypothesis: constraining "path space" increased (not decreased) uncertainty
   - Clear dose-response: more reduction → higher uncertainty
   - Very large, highly significant effects

2. **Orthogonal noise injection had NO effect** (p = 0.40)
   - Model robust to perturbations in null space
   - Suggests task-relevant information lies in principal components

3. **Interpretation**: "Path space" cannot be straightforwardly operationalized as linear dimensionality
   - Information loss from PCA impairs confident prediction
   - Even <1% discarded variance substantially impacts uncertainty
   - Need alternative theories of what "path representations" mean

---

## Repository Structure

```
path-engineering-nlp-62d0/
├── README.md                          # This file
├── REPORT.md                          # Full research report (detailed)
├── planning.md                        # Experimental design document
├── notebooks/
│   └── 2025-11-16-22-55_PathEngineering.ipynb  # Main analysis notebook
├── results/
│   ├── experiment_results.json        # Raw numerical results
│   ├── results.csv                    # Results in tabular format
│   └── uncertainty_analysis.png       # Main visualization
├── pyproject.toml                     # Project dependencies
└── .venv/                             # Virtual environment (not in repo)
```

---

## How to Reproduce

### 1. Environment Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add torch transformers datasets numpy scipy matplotlib pandas scikit-learn
```

**Required packages:**
- Python 3.10+
- PyTorch 2.9+
- Transformers 4.57+
- NumPy, SciPy, Matplotlib, Pandas, Scikit-learn

### 2. Run Experiments

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/2025-11-16-22-55_PathEngineering.ipynb
```

**Runtime**: ~30 minutes on GPU, ~2 hours on CPU

**Outputs**: Results saved to `results/` directory

### 3. View Results

- **Quick overview**: See visualizations in `results/uncertainty_analysis.png`
- **Full details**: Read `REPORT.md`
- **Raw data**: Check `results/experiment_results.json` or `results.csv`

---

## Experimental Design Summary

### Hypothesis (Original)
Constraining path space → ⬇️ uncertainty
Expanding path space → ⬆️ uncertainty

### Method
- **Model**: GPT-2 (124M parameters)
- **Task**: 20 math reasoning problems
- **Intervention**: Modified hidden states at Layer 6 during generation
  - **Path constraint**: PCA projection to {384, 192, 96} dimensions
  - **Path expansion**: Orthogonal noise injection (scales: 0.01, 0.05, 0.1)
- **Metrics**: Output entropy, confidence scores

### Results

| Condition   | Mean Entropy | Change from Baseline | p-value  | Cohen's d |
|-------------|--------------|----------------------|----------|-----------|
| Baseline    | 2.40         | —                    | —        | —         |
| PCA-384     | 3.27         | **+36%** ⬆️          | <0.0001  | 1.22      |
| PCA-192     | 3.44         | **+43%** ⬆️          | <0.0001  | 1.54      |
| PCA-96      | 3.65         | **+52%** ⬆️          | <0.0001  | 1.84      |
| Noise-0.05  | 2.33         | -3% (ns)             | 0.40     | -0.09     |

**Interpretation**: Hypothesis refuted for PCA (opposite effect), null result for noise.

---

## Key Insights

### What We Learned

✅ **Direct causal interventions on representations affect uncertainty**
- Not just correlation—we manipulated activations and measured effects

✅ **Information loss increases uncertainty**
- Even variance-preserving PCA (99.9% variance retained) hurts confidence
- The "last 1%" contains critical information

✅ **Model is robust to null-space noise**
- Orthogonal perturbations don't affect outputs
- Task-relevant info concentrated in principal components

❌ **"Path space" ≠ simple dimensionality**
- Original hypothesis: reducing dimensions constrains paths → less uncertainty
- Reality: reducing dimensions loses information → more uncertainty
- Need better operationalization of "path space" concept

### Implications

**For AI safety / uncertainty quantification:**
- Compressing representations can hurt model confidence (without changing correctness)
- Be cautious when intervening on hidden states

**For mechanistic interpretability:**
- Linear dimensionality is insufficient to capture "path representations"
- Models use their full representational capacity (even small variance components matter)

**For theory:**
- Challenges simple interpretations of activation space as "path space"
- Correlation between uncertainty and activation diversity (from prior work) doesn't imply our causal mechanism

---

## Limitations

- **Small scale**: 20 problems, 1 model (GPT-2)
- **Single intervention layer**: Only tested Layer 6
- **Simple task**: Math reasoning only
- **Operationalization**: PCA may not be the right way to manipulate "path space"

**Future work should:**
- Test on larger models (GPT-3, LLaMA, etc.)
- Try alternative interventions (autoencoders, steering vectors, etc.)
- Explore different tasks and layers
- Develop better formalizations of "path space"

---

## Citation

If you build on this work:

```
Path Engineering Research: Causal Manipulation of the "Road Not Taken" (2025)
Investigation of dimensionality reduction and noise injection effects on LLM uncertainty
GitHub: [Repository URL]
```

**Related paper**: ["Are language models aware of the road not taken?"](https://arxiv.org/abs/2511.04527) (arXiv:2511.04527)

---

## Contact & Questions

For questions about this research or to discuss extensions, please open an issue in this repository.

**Research conducted as part of automated AI research system.**

---

## License

MIT License - Feel free to use and build upon this research.

---

**📊 For full details, see [REPORT.md](REPORT.md)**
