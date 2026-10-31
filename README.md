# WiseDynamics

WiseDynamics is the simulation home for Wiseware's GNC work. Every simulation plant is a sibling
directory under `sim/` — Basilisk first (`sim/basilisk/`), with other hosts such as Trick
(`sim/trick/`) joining later as siblings rather than replacements. It also holds the Monte Carlo
campaign drivers that run those plants in bulk, and the exporters that write recorded runs out to
the analysis data contract.

Flight, simulation and analysis are split into three homes so any one can be replaced without
touching the other two: flight GNC algorithms live in
[WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign), framework-free and
simulation-blind; simulation plants and Monte Carlo campaigns that import those algorithms live
here, in WiseDynamics; analysis of the runs they produce lives in
[WiseGncAnalysis](https://github.com/WisewareOrg/WiseGncAnalysis), which reads the data contract
and knows nothing about which simulation produced it.

## Layout

- `sim/basilisk/` — the Basilisk simulation host (first; see its own `README.md`).
- `sim/<host>/` — a future simulation host, as a sibling directory.
- Monte Carlo drivers and the exporters to the analysis data contract live alongside the
  simulation hosts they serve.
- `docs/decisions/` — this repository's ADRs.
- `docs/evaluation/` — evaluation evidence for simulation-host choices.
