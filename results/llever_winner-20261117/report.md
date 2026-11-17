# Predicting the next-token winner after steering Recurrent-Llama

The frozen 48-loop forecast called the exact next token on 21 of 24 selected near ties at the primary dose. A one-loop forecast called 20 and keeping the original winner called 11. The full forecast more accurately predicted how far the runner-up moved relative to the original winner on 21 of 24 prompts. The experiment demonstrates prospective prediction of targeted token changes in this sample. The difference in exact-winner accuracy between recurrent forecasts is unresolved.

## Selection

The 768 fresh prompts comprised 192 in each of four arithmetic families. Before intervention, 34 qualified as near ties and 215 as clear leads. The fixed ordering selected 24 per group. Of the 24 near ties, 22 were multiply-add and two were three-term sums. The clear-lead group included 11 ordinary sums, six three-term sums, five subtractions and two multiply-adds. Differences between groups also reflect that task mix. Both groups had stable sham winners on all selected prompts.

## Results

For each row the denominator is 24. Target wins count moves to the original runner-up. Winner accuracy uses the whole vocabulary. Margin error is the mean absolute prediction error in logits for the original runner-up minus original winner, measured against sham.

| Group | Dose | Target wins | Third-token wins | Winners correct, full / one / unchanged | Margin error, full / one / unchanged |
|---|---:|---:|---:|---|---|
| Near | 1% | 0 | 0 | 24 / 24 / 24 | .001 / .019 / .058 |
| Near | 3% | 5 | 0 | 23 / 22 / 19 | .010 / .060 / .175 |
| Near | 9% | 12 | 1 | 21 / 20 / 11 | .086 / .197 / .514 |
| Far | 1% | 0 | 0 | 24 / 24 / 24 | .001 / .016 / .067 |
| Far | 3% | 0 | 0 | 24 / 24 / 24 | .013 / .051 / .200 |
| Far | 9% | 0 | 0 | 24 / 24 / 24 | .121 / .204 / .604 |

At the primary 9% dose the full forecast correctly called 10 of the 12 target wins, 10 of the 11 retained winners and the single third-token win. In paired exact-winner comparisons, both forecasts succeeded on 19 prompts and both missed on two. Only the full forecast was right on two and only the one-loop forecast on one. The selected sample measures control of the next token. Accuracy improvement on arithmetic was outside the design.

## Run record

The screen, forecast and measurement used the same implementation hash and pinned checkpoint. Forecast SHA-256 logged before measurement was ef0031f7da7db85512648f6554d3040efc976101dc713f497e3437069c15a44d. The harness agreement threshold was 1e-6 centered logits. The three workers recorded 162.7, 27.4 and 28.9 function-seconds, respectively, totaling about $0.13 in GPU time at the base A100-40GB rate, before platform overhead, CPU, memory or credits.

[Protocol](../../docs/llever_winner_design.md) · [Screen](screen.json) · [Frozen forecasts](predict.json) · [Measurements](evaluate.json) · [Summary](summary.json)
