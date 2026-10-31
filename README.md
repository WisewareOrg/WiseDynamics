# WiseDynamics

WiseDynamics is the simulation home for Wiseware's GNC work. Every simulation plant is a sibling
directory under `sim/` — Basilisk first (`sim/basilisk/`), with other hosts such as Trick
(`sim/trick/`) joining later as siblings rather than replacements. A Monte Carlo campaign is a
study's own pytest module run at scale (`pytest <scenarios dir> --runs N --out DIR [-n WORKERS]`),
writing one run file per case as it goes, in the layout WiseGncAnalysis's own loader reads.

Flight, simulation and analysis are split into three homes so any one can be replaced without
touching the other two: flight GNC algorithms live in
[WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign), framework-free and
simulation-blind; simulation plants and Monte Carlo campaigns that import those algorithms live
here, in WiseDynamics; analysis of the runs they produce lives in
[WiseGncAnalysis](https://github.com/WisewareOrg/WiseGncAnalysis), which reads those run files
and knows nothing about which simulation produced them. WiseDynamics checks out WiseGncDesign
and WiseGncAnalysis as sibling directories and depends on each by editable path rather than by a
pinned revision; the WiseGncAnalysis dependency is dev-only, for checking an exported run
against its real loader.

## Layout

- `sim/basilisk/` — the Basilisk simulation host, the first plant
  ([ADR 0001 rev 1](docs/decisions/0001-basilisk-first-simulation-host.md)); see its own `README.md`.
- `sim/<host>/` — a future simulation host, as a sibling directory.
- A simulation host's studies, and the export to the format WiseGncAnalysis's loader reads, live
  inside that host's own directory, alongside the plants and the seam that host the algorithms
  under study.
- `docs/decisions/` — this repository's ADRs.
