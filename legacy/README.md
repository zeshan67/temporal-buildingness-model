# v1.0 reference (submitted manuscript)

The submitted manuscript (ARIN-D-26-00098, tag `v1.0-submitted`) reports 30-seed results.
The original multi-seed script was not preserved. `reproduce_v1_0.py` reconstructs it from
`run_all_modes` in `src/simulation/mode_comparison_runner.py`:

- base seeds 42–71 set the street graph and agent random streams;
- each mode's demand zones and anchors use base seed + mode number (YAML order, bus = 10);
- each mode runs on its own city-context-modified copy of the graph.

The only code change from the tag is the B0 indentation fix in `config/transport_modes.yaml`.

## Agreement with the manuscript (mean TBI, 30 seeds)

| Condition | Reconstructed | Manuscript | Difference |
|---|---|---|---|
| Jeepney | 0.406 | 0.410 | −0.004 |
| Tro-tro | 0.403 | 0.397 | +0.006 |
| Matatu | 0.396 | 0.403 | −0.007 |
| Colectivo/Pesero | 0.384 | 0.392 | −0.009 |
| Rickshaw | 0.346 | 0.349 | −0.003 |
| Formal bus | 0.273 | 0.278 | −0.005 |
| Boda-boda | 0.267 | 0.266 | +0.001 |
| Jitney | 0.258 | 0.259 | −0.001 |
| Conventional baseline | 0.022 | 0.022 | 0.000 |

Six of nine reported means agree within 0.005; the other three differ by 0.006–0.009
(about half an SD), consistent with a different seed scheme rather than different model
behaviour. Jitney and bus agree, so B0 did not affect the manuscript's results.
These reconstructed values (`outputs/v1.0_reproduction/`) are the reference against which
every v1.1 change is measured.
