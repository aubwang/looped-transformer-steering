#import "tables.typ": *
#import "numbers.typ": *
#set text(size: 8pt)
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_accuracy
#tab_accuracy
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_table
#tab_table
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_best
#tab_best
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_added
#tab_added
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_added_best
#tab_added_best
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_loopcounts
#tab_loopcounts
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_robust
#tab_robust
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_cross
#tab_cross
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_cross_mean
#tab_cross_mean
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_budget
#text(size: 5pt)[#tab_budget]
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_budget32
#text(size: 5pt)[#tab_budget32]
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_budget_L32_families
#tab_budget_L32_families
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_forecast
#tab_forecast
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_conformal
#tab_conformal
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_dynamics
#tab_dynamics
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_dyn
#tab_dyn
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_seeds
#text(size: 5pt)[#tab_seeds]
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_sites
#tab_sites
#pagebreak()
#set page(paper: "a4", flipped: false, margin: 10mm)
= tab_coverage
#tab_coverage
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_threshold
#text(size: 5pt)[#tab_threshold]
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_forecast_main
#text(size: 5pt)[#tab_forecast_main]
#import "explore/explore.typ": *
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_joint
#text(size: 7pt)[#tab_joint]
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_scorecorr
#text(size: 7pt)[#tab_scorecorr]
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_stratified
Conditioning on privileged correctness of the other model; descriptive only. Subgroup AUROCs pool items.
#text(size: 7pt)[#tab_stratified]
#pagebreak()
#set page(paper: "a4", flipped: true, margin: 10mm)
= tab_deep
Same 256 items at source and target. AUROCs average frozen folds. Δ = target minus source. Class-stratified intervals deviate from the registered rule.
#text(size: 7pt)[#tab_deep]
Privileged correctness conditioning is descriptive only. Deep Δ = target minus source on the same 256 items; AUROC averages frozen folds. Stratified intervals deviate from the registered rule; see explore/explore.json.
