# SOLUTION.md

## Current best solution snapshot
- **best LB**: {{score}} ({{exp_id}}, {{date}})
- **best CV**: {{score}} ({{exp_id}}, {{date}})

## Architecture summary
{{1–3 paragraphs: model family, key features, ensemble structure, post-processing chain.}}

## Components
| component | role | source | added by exp |
|---|---|---|---|
| {{Perch v2 ONNX}} | embedding | external | exp001 |
| {{ProtoSSM}} | classifier | src/models/ | exp012 |
| {{SED CNN}} | aux classifier | src/models/ | exp020 |

## Ensemble recipe
{{rank-avg / weighted prob avg / stacking — with weights per component}}

## Post-processing chain
{{ordered list of PP stages with hyper-params}}

## Known weaknesses (= what to attack next)
- {{rare class A still < 0.4}}
- {{site B underperforms}}

## Things tried and rejected
- {{model X — no signal at exp042}}
- {{aug Y — overfit at exp048}}
