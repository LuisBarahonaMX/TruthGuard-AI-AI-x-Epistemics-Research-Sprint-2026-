# TruthGuard AI V2 — Statistical analysis (exploratory, provisional)

Generated: 2026-11-15T17:50:16.976396+00:00

## Experimental coverage

- 420/420 unique completed scenarios in consolidation input (7 models × 20 cases × 3 conditions).
- 96 adjudicated records and 324 records with automatic labels not individually adjudicated.
- 4 unjustified flips identified among adjudicated intervention records.
- Input SHA-256: `0a2f9e25e2504e01e494d777be3be97ad34c61a38c8eee0839110082dde84a79`

## Overall intervention outcomes

|Condition|Flips / evaluable initially correct|Rate|95% Wilson interval|
|---|---:|---:|---:|
|B: direct pressure|4/140|2.86%|[0.0112, 0.0712]|
|C: simulated authority|0/140|0.00%|[0.0000, 0.0267]|

These intervals describe binomial score uncertainty under simplifying assumptions; they do **not** establish population generalizability. Numerators use mixed reviewed and automatic labels.

## Paired comparison: the same model × question under B and C

- Matched eligible pairs: 140.
- Only B flip: 4; only C flip: 0; both flip: 0; neither: 136.
- Paired risk difference B minus C: 2.86 percentage points.
- Exact two-sided McNemar p-value: 0.125000 (exploratory; dependence across cases/models is not captured).
- Case-clustered bootstrap sensitivity interval, seed 20261115: [0.00, 6.43] percentage points (resamples the 20 selected case IDs; not a general population CI).

## Important limitations

1. Only 20 hand-selected, simple numerical questions: arithmetic, numeric comparison and multi-step arithmetic; not a representative sample of real-world fact checking.
2. One recorded conversation for each model × case × condition; no repeated-seed variability/temporal stability analysis.
3. Labels combine 96 adjudications (AI-assisted review approved in project workflow) and 324 automatic classifications. A sample of 42 of those was AI-assisted reviewed, **not** an external independent human audit.
4. The language of the simulated authority was explicitly tentative ('supposed official solution'); results depend on intervention wording.
5. Several models share providers/families and cases are re-used, so observations are not independent; Wilson/McNemar are provisional descriptive diagnostics, not definitive inferential evidence.
6. A refusal to adopt a false answer is not sufficient to prove calibrated belief updating; a valid-evidence-correction condition D was not run.
7. Four Qwen cases required higher output limit, and some API calls required recovery after rate limiting. Inference settings should be documented.
8. No definitive ranking of model epistemic reliability should be reported.

## Outputs

- `metrics_by_group.csv`: model × condition, category × condition, overall metrics + conditional Wilson bounds.
- `paired_B_vs_C.csv`: matched comparison and bootstrap sensitivity.
- `paired_by_category.csv`: matched breakdown by problem category.
- `paired_by_model.csv`: matched breakdown by model, not a ranking.
- `statistical_manifest.json`: integrity and provenance.
