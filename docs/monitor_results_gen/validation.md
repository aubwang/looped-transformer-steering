Validation: ALL CHECKS PASSED; independent tests check 245 cells in the 2×2, best-site, fixed-added and best-added tables directly against raw JSON.

Manually compared these five cells against summary.txt (m1a):

- table/L32/L32 / family_mean: 0.687 [0.671, 0.701]
- table/L32/D / family_mean: 0.680 [0.664, 0.697]
- table/D/L32 / family_mean: 0.672 [0.656, 0.688]
- table/D/D / family_mean: 0.685 [0.667, 0.699]
- diagonal / family_mean: 0.002 [−0.017, 0.025]

Compiled with the supplied typstenv Python using typst.compile(..., root="."). Inspected all 19 rendered pages via pages/contact.png; no table extends outside the page. The conformal table spans four landscape pages with repeated headers. Budget tables use landscape A4 at 5pt; forecast and conformal use landscape A4 at 8pt. Other tables use portrait A4 at 8pt. The cross-family mean-over-targets block is exported separately as tab_cross_mean to retain all five family columns.

Raw false_stability is unavailable in at least one seed for L8/4, L16/4, L32/4 and L32/8; those aggregates show –. All state_conformal candidates lack fraction_forecast_stable, false_stability and decision_changed; these show –. The requested corresponding score_conformal quantities exist. Every unavailable bootstrap interval shows [n/a]. D accuracy is included using m1a table/D/D class counts.
