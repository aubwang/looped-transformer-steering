status: Exploratory, chosen after registered outcomes were known; no causal interpretation
alignment: Unique pid arrays checked; exact set equality for correlations, deep target subset lookup into source; labels asserted equal for same-label comparisons; frozen source folds asserted equal after deep alignment. Labels from saved y, no raw hidden states loaded.
auroc: Unweighted mean of identifiable frozen fold AUROCs; conditional stratified subgroup AUROCs pooled within subgroup.
bootstrap: 2000 paired item resamples within frozen folds; any undefined draw invalidates whole interval
stratified_deviation: Explicit deviation from registered rule: resample within fold × joint(source label,target label) classes
deep_difference: target minus source, same deep items at both budgets; no native deep-budget fits
privileged: Conditioning on the other model correctness is privileged and descriptive only

Sources: scores.npz keys are recorded per row; pid, y, scores, folds are the only arrays read. Joint labels were verified identical across all three seeds.
Regenerate with the same --inputs as tables_mon.py and --out docs/monitor_results_gen/explore.

All 252 deep comparisons are available; 60 registered intervals are unavailable (at least one undefined bootstrap draw), while all class-stratified deviation intervals are available. No requested saved fields were missing.
