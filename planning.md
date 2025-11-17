# Path Engineering Research: Detailed Experimental Plan

## Research Question

**Can we causally manipulate language model uncertainty by artificially constraining or expanding the "path space" in hidden representations, independent of the reasoning problem itself?**

## Background and Motivation

Recent work (arXiv:2511.04527) demonstrates that language models' hidden activations correlate with uncertainty and represent "the space of possible paths." This suggests models internally encode multiple reasoning trajectories. However, correlation ≠ causation.

**Our contribution**: We test whether this relationship is causal by directly manipulating the path space through:
1. **Dimensionality reduction** (constraining paths)
2. **Noise injection** (expanding paths)

If the hypothesis is correct, we should observe:
- Reduced dimensionality → Lower uncertainty (more confident, even if wrong)
- Expanded dimensionality → Higher uncertainty (less confident)

This has implications for:
- Understanding how models represent uncertainty
- Controlling model confidence during deployment
- Improving calibration and reliability

## Hypothesis Decomposition

### Main Hypothesis
Artificially constraining the model's path space by reducing the dimensionality of hidden representations along critical directions will decrease uncertainty, while expanding the path space will increase uncertainty.

### Sub-Hypotheses

**H1 (Path Constraint)**: Projecting hidden states onto lower-dimensional PCA subspaces will:
- H1a: Decrease output entropy
- H1b: Increase confidence scores
- H1c: Reduce variance in predicted token probabilities

**H2 (Path Expansion)**: Adding orthogonal noise to hidden states will:
- H2a: Increase output entropy
- H2b: Decrease confidence scores
- H2c: Increase variance in predicted token probabilities

**H3 (Task Independence)**: These effects should occur:
- H3a: Regardless of problem difficulty
- H3b: Without changing the most likely answer (in most cases)
- H3c: Proportionally to the magnitude of intervention

### Null Hypothesis
H0: Manipulating hidden representation dimensionality has no systematic effect on output uncertainty metrics.

## Proposed Methodology

### Approach Overview

Since we have CPU-only constraints and cannot easily access internal hidden states of frontier models via API, we will use a **two-pronged approach**:

#### **Primary Approach: API-based with logprobs**
- Use GPT-4 or similar models via API
- Manipulate uncertainty indirectly through prompting techniques that simulate path constraint/expansion
- Measure uncertainty via output logprobs and token probabilities
- Validate that uncertainty changes as predicted

#### **Secondary Approach: Local model with direct intervention** (if time permits)
- Download GPT-2 or small open model (e.g., GPT-2-medium, ~350M params)
- Directly manipulate hidden states using hooks
- Apply PCA dimensionality reduction and noise injection
- Measure both internal representations and output uncertainty

**Rationale**: The primary approach is faster and stays within compute constraints. The secondary approach provides stronger evidence but requires more implementation time.

### Experimental Steps

#### Phase A: Baseline Measurement (30 min)
1. Select 50-100 reasoning problems from GSM8K (varying difficulty)
2. Run baseline inference with standard prompting
3. Collect:
   - Output tokens and answers
   - Log probabilities for top-5 tokens at each position
   - Entropy at decision points
   - Correctness of final answer

#### Phase B: Path Constraint Intervention (45 min)
**API Version**:
- Use constrained prompting: "Think step-by-step with only the most direct approach"
- Compare uncertainty metrics to baseline

**Local Model Version** (if implemented):
- Extract hidden states at middle layers (e.g., layer 12/24)
- Apply PCA to find top-k principal components (k = {128, 64, 32})
- Project activations onto reduced subspace
- Continue forward pass
- Measure output uncertainty

#### Phase C: Path Expansion Intervention (45 min)
**API Version**:
- Use expansive prompting: "Consider multiple different approaches and perspectives"
- Compare uncertainty metrics to baseline

**Local Model Version** (if implemented):
- Add Gaussian noise orthogonal to principal components
- Noise scale: σ ∈ {0.01, 0.05, 0.1} × std(activations)
- Continue forward pass
- Measure output uncertainty

#### Phase D: Comparative Analysis (30 min)
- Statistical tests comparing intervention groups
- Effect size calculations
- Visualizations of uncertainty distributions

### Baselines

1. **Unmodified baseline**: Standard inference, no intervention
2. **Random perturbation**: Shuffle tokens or add random prompt variations (control for intervention effects)
3. **Difficulty-matched subsets**: Compare interventions within same difficulty levels

### Evaluation Metrics

#### Uncertainty Metrics (Primary)
1. **Output Entropy**: H = -Σ p(x) log p(x) over token distribution
2. **Top-1 Confidence**: Max probability assigned to any token
3. **Probability Spread**: Variance in top-5 token probabilities
4. **Perplexity**: exp(average negative log-likelihood)

#### Quality Metrics (Secondary)
1. **Accuracy**: % correct final answers (should not drastically change)
2. **Answer Stability**: % problems where most-likely answer changes
3. **Reasoning Coherence**: Manual check that outputs remain sensible

#### Effect Size Metrics
1. **Cohen's d**: Standardized mean difference between conditions
2. **% Change**: Relative change in uncertainty metrics
3. **Correlation**: Relationship between intervention magnitude and uncertainty change

### Statistical Analysis Plan

#### Tests
- **Paired t-tests**: Compare same problems under different interventions
- **Wilcoxon signed-rank**: Non-parametric alternative if distributions non-normal
- **ANOVA**: Compare multiple intervention levels (e.g., PCA dimensions: 128 vs 64 vs 32)
- **Correlation analysis**: Intervention magnitude vs uncertainty change

#### Significance Level
- α = 0.05 (two-tailed)
- Bonferroni correction for multiple comparisons: α' = 0.05 / n_comparisons

#### Sample Size
- Minimum 50 problems per condition for 80% power (medium effect size d=0.5)
- Target: 100 problems for robustness

#### Assumptions to Check
- Normality: Shapiro-Wilk test, Q-Q plots
- Paired differences: Check for systematic outliers
- Effect consistency: Analyze by problem difficulty quartiles

## Expected Outcomes

### If Hypothesis Supported
1. **Path Constraint** → Reduced uncertainty metrics:
   - Mean entropy decreases by 15-30%
   - Top-1 confidence increases by 5-15 percentage points
   - Effect stronger for medium-difficulty problems (floor/ceiling effects for easy/hard)

2. **Path Expansion** → Increased uncertainty metrics:
   - Mean entropy increases by 10-25%
   - Top-1 confidence decreases by 5-10 percentage points
   - More variable outputs, potentially changing answers

3. **Dose-response relationship**:
   - Larger PCA reduction → greater uncertainty reduction
   - Higher noise levels → greater uncertainty increase

### If Hypothesis Rejected
1. No systematic relationship between intervention and uncertainty
2. p-values > 0.05 across tests
3. Effect sizes near zero (d < 0.2)

### Ambiguous Outcomes
1. Effects only appear for certain problem types
2. Answer quality degrades severely (intervention too disruptive)
3. API-based prompting doesn't effectively manipulate "path space"

## Timeline and Milestones

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Planning | 20 min | This document |
| Environment Setup | 15 min | Dependencies installed, data loaded |
| Baseline Implementation | 30 min | Can run inference and collect metrics |
| Intervention Implementation | 60 min | Both constraint and expansion working |
| Experiment Execution | 60 min | All conditions run, data collected |
| Statistical Analysis | 30 min | Tests complete, visualizations created |
| Documentation | 25 min | REPORT.md with findings |
| **TOTAL** | **240 min (4h)** | Complete research cycle |

### Critical Path
1. Get baseline working (cannot proceed without this)
2. Implement at least one intervention type (constraint OR expansion)
3. Collect sufficient data (n≥50) for statistical power
4. Run statistical tests
5. Document findings

### Contingency Buffer
- 20% time buffer built into each phase
- If local model approach too slow, pivot fully to API approach
- If GSM8K too complex, use simpler QA dataset (e.g., CommonsenseQA)

## Potential Challenges

### Challenge 1: API Limitations
**Issue**: Cannot directly access hidden states via API

**Mitigation**:
- Focus on output uncertainty (still valid test of hypothesis)
- Use prompting as proxy for path manipulation
- Clearly document limitations in conclusions

**Fallback**: Download GPT-2 for direct intervention (adds ~30 min)

### Challenge 2: Intervention Too Disruptive
**Issue**: Model outputs become gibberish/wrong

**Mitigation**:
- Start with small intervention magnitudes
- Validate that answer quality remains reasonable
- Report separately: "uncertainty without severe quality degradation"

**Contingency**: Reduce intervention strength until outputs valid

### Challenge 3: No Significant Effect
**Issue**: Interventions don't change uncertainty

**Mitigation**:
- Check implementation correctness
- Try stronger interventions
- Analyze subgroups (maybe effects only on certain problems)
- Document as negative result (still valuable!)

**Interpretation**: Either hypothesis wrong OR methodology insufficient

### Challenge 4: Confounding Factors
**Issue**: Changes in uncertainty due to other factors (e.g., answer changes)

**Mitigation**:
- Measure answer stability
- Analyze only problems where answer unchanged
- Use random baseline to control for non-specific effects

### Challenge 5: Computational Constraints
**Issue**: CPU-only, limited time

**Mitigation**:
- Prioritize API approach (faster)
- Use small local model if needed (GPT-2-small: 117M params)
- Reduce dataset size if necessary (50 vs 100 problems)
- Cache API responses to avoid redundant calls

## Success Criteria

### Minimum Viable Success
1. ✓ Ran experiments with ≥50 problems
2. ✓ Implemented at least 1 intervention type (constraint OR expansion)
3. ✓ Collected uncertainty metrics
4. ✓ Performed statistical tests
5. ✓ Documented findings in REPORT.md

### Full Success
1. ✓ All of above PLUS:
2. ✓ Both intervention types implemented
3. ✓ Significant results (p < 0.05) in predicted direction
4. ✓ Effect sizes ≥ medium (d ≥ 0.5)
5. ✓ Visualizations showing clear patterns
6. ✓ Validated across problem difficulty levels

### Exceptional Success
1. ✓ All of above PLUS:
2. ✓ Direct hidden state manipulation (local model)
3. ✓ Dose-response relationship demonstrated
4. ✓ Mechanistic insights into which layers/components matter
5. ✓ Reproducible code and clear documentation

## Risks and Limitations

### Methodological Limitations
1. **Correlation vs Causation**: Even with interventions, other explanations possible
2. **Limited Generalization**: Results may not transfer to larger/different models
3. **Prompting Proxy**: API approach doesn't directly manipulate representations
4. **Small Sample**: 50-100 problems may miss subtle effects

### Theoretical Limitations
1. **"Path space" definition**: Operationalized via dimensionality, but may be incomplete
2. **Uncertainty != Calibration**: High uncertainty doesn't mean well-calibrated
3. **Task specificity**: Results may only apply to mathematical reasoning

### Technical Limitations
1. **CPU-only**: Cannot run large models locally with interventions
2. **API black-box**: Cannot inspect internal mechanisms
3. **Time constraint**: 4 hours limits depth of investigation

### Addressing Limitations
- Acknowledge all limitations explicitly in REPORT.md
- Suggest future work with better resources
- Frame findings as preliminary/exploratory
- Focus on effect demonstration rather than mechanism

## Next Steps After This Plan

1. **Immediate**: Install dependencies (transformers, openai, datasets, scipy, matplotlib)
2. **Load Data**: Download GSM8K or create small test set
3. **Implement Baseline**: Get working inference pipeline
4. **Implement Interventions**: Start with simpler API approach
5. **Run Experiments**: Execute experimental protocol
6. **Analyze**: Statistical tests and visualizations
7. **Document**: Write REPORT.md with findings

---

**Plan Status**: ✓ Complete - Ready for implementation

**Key Decision**: Start with API-based approach for speed, add local model if time permits

**Success Metric**: Demonstrate causal relationship between path space manipulation and uncertainty
