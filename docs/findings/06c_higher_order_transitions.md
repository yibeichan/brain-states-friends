# Findings: 06c Higher-Order Transition Structure Diagnostic

_Script: `script/06c_higher_order_transitions.py`. Tier: SUPP (Supp)._

_Model adequacy diagnostic quantifying how much sequential structure in decoded state sequences exceeds the HMM's 1st-order transition assumption; per-subject, n=6._

## Method (as run)

- Parcellation: atlas-4S156Parcels, vt0.95
- Input: decoded state sequences from 04 (select mode); recurrence summary from 05a
- State-change sequences constructed by collapsing consecutive identical states (min_dwell_tr=2)
- Four analyses: (1) conditional entropy reduction H1 vs H2 with Miller-Madow bias correction and bootstrap CI (n=1000 resamples over episodes); (2) per-(A,B,C) context-dependence binomial test with leave-one-out baseline and BH-FDR; (3) BIC/AIC comparison of order-1 vs restricted order-2 (VLMC) evaluated on the same trigram observation set; (4) hierarchical 4-gram null against a 2nd-order Markov baseline with leave-one-out P(D|B,C) and BH-FDR
- All analyses are per-subject; no group statistic is computed
- min_context_count=10 for analyses 2, 3, 4; 4-gram tests exclude expected count < 5 (normal approximation unreliable below this floor)

## Results

### Conditional entropy reduction (Analysis 1)

| Subject | H1 (bits) | H2 (bits) | DeltaH (bits) | DeltaH (%) | Boot median (bits) | 95% CI | CI > 0 |
|---------|-----------|-----------|---------------|------------|--------------------|--------|--------|
| sub-01 | 4.655 | 3.948 | 0.707 | 15.2% | 1.169 | [1.115, 1.233] | Yes |
| sub-02 | 4.704 | 4.020 | 0.684 | 14.5% | 1.151 | [1.094, 1.209] | Yes |
| sub-03 | 4.600 | 3.990 | 0.610 | 13.3% | 1.034 | [0.982, 1.092] | Yes |
| sub-04 | 4.508 | 3.789 | 0.718 | 15.9% | 1.157 | [1.088, 1.232] | Yes |
| sub-05 | 4.569 | 3.902 | 0.668 | 14.6% | 1.127 | [1.070, 1.183] | Yes |
| sub-06 | 4.469 | 3.924 | 0.545 | 12.2% | 0.920 | [0.870, 0.971] | Yes |

Bootstrap CI excludes zero for all six subjects. The point estimate is consistently lower than the bootstrap median because full-data observations are spread across many sparse 2nd-order contexts (inflating the Miller-Madow correction); bootstrap samples concentrate into fewer contexts, producing higher median estimates. The point estimate is conservative.

### Context-dependence test per trigram (Analysis 2, BH-FDR)

| Subject | Testable (A,B,C) triples | Significant (FDR < 0.05) | Fraction |
|---------|--------------------------|--------------------------|----------|
| sub-01 | 13,583 | 30 | 0.22% |
| sub-02 | 14,381 | 25 | 0.17% |
| sub-03 | 14,360 | 15 | 0.10% |
| sub-04 | 9,910 | 23 | 0.23% |
| sub-05 | 14,360 | 39 | 0.27% |
| sub-06 | 13,926 | 47 | 0.34% |

Leave-one-out binomial test; all well-observed contexts tested (not pre-selected), with BH-FDR correction over ~9,900-14,400 tests per subject. Range across subjects: 0.10%-0.34% of contexts significant.

### BIC/AIC Markov order comparison (Analysis 3)

| Subject | BIC order-1 | BIC order-2 | ΔBIC (1 minus 2) | BIC prefers | AIC order-1 | AIC order-2 | AIC prefers |
|---------|-------------|-------------|------------------|-------------|-------------|-------------|-------------|
| sub-01 | 221,284 | 321,510 | -100,227 | order-1 | 206,294 | 199,814 | order-2 |
| sub-02 | 226,648 | 333,556 | -106,909 | order-1 | 212,384 | 206,176 | order-2 |
| sub-03 | 243,116 | 351,359 | -108,243 | order-1 | 228,322 | 222,503 | order-2 |
| sub-04 | 171,359 | 242,470 | -71,110 | order-1 | 158,081 | 153,468 | order-2 |
| sub-05 | 225,034 | 332,210 | -107,176 | order-1 | 210,441 | 204,043 | order-2 |
| sub-06 | 249,819 | 355,751 | -105,932 | order-1 | 236,110 | 230,568 | order-2 |

BIC prefers order-1 for all six subjects (ΔBIC range: -71,110 to -108,243). AIC prefers order-2 in all cases. The restricted order-2 model has approximately 10,952-15,256 total parameters (order-1 fallback plus extra 2nd-order parameters) evaluated on 25,000-38,000 trigram observations. BIC penalizes the large parameter count heavily at this observation-to-parameter ratio.

### 4-gram hierarchical null (Analysis 4, 2nd-order Markov baseline)

| Subject | 4-grams tested | Survive 2nd-order null (FDR < 0.05, CEI > 0) | Absorbed by 2nd-order | Survival rate |
|---------|----------------|-----------------------------------------------|-----------------------|---------------|
| sub-01 | 344 | 0 | 344 | 0% |
| sub-02 | 284 | 1 | 283 | 0.4% |
| sub-03 | 543 | 1 | 542 | 0.2% |
| sub-04 | 334 | 0 | 334 | 0% |
| sub-05 | 313 | 2 | 311 | 0.6% |
| sub-06 | 921 | 0 | 921 | 0% |

4-grams with expected count < 5 are excluded from significance testing (z-score and p-value set to NaN; normal approximation unreliable at low expected counts). The expected-count floor applies to expected counts only; surviving 4-grams have observed counts well above that floor (sub-02: 22, sub-03: 27, sub-05: 16 and 292).

## Outputs

- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/higher_order_summary.json
- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/conditional_entropy.json
- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/markov_order_comparison.json
- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/context_dependence.csv
- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/hierarchical_null_4grams.csv
- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/conditional_entropy.{png,pdf}
- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/context_dependence_scatter.{png,pdf}
- output/06c_higher_order_transitions/atlas-4S156Parcels/{sub_id}/vt0.95/markov_order_comparison.{png,pdf}
