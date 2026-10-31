# WiseDynamics

WiseDynamics is the simulation home for Wiseware's GNC work. Every simulation plant is a sibling
directory under `sim/` — Basilisk first (`sim/basilisk/`), with other hosts such as Trick
(`sim/trick/`) joining later as siblings rather than replacements. A Monte Carlo campaign is a
study's own pytest module run at scale (`pytest <scenarios dir> --runs N --out DIR [-n WORKERS]`),
writing one run file per case as it goes, in the layout the analysis package's own loader reads.

Flight and simulation are split into two homes so either can be replaced without touching the
other: [WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign) holds the flight GNC
algorithms, the C&DH partition that hosts a named one on a simulator, and the analysis package
(`gnc_analysis`) that reads a run back and knows nothing about which simulation produced it;
simulation plants and the Monte Carlo campaigns that boot those algorithms live here, in
WiseDynamics. WiseDynamics checks out WiseGncDesign as a sibling directory and depends on it by
editable path rather than by a pinned revision.

## Layout

- `sim/basilisk/` — the Basilisk simulation host, the first plant
  ([ADR 0001 rev 2](docs/decisions/0001-basilisk-first-simulation-host.md)); see its own `README.md`.
- `sim/<host>/` — a future simulation host, as a sibling directory.
- A simulation host's plants and studies, and the export to the format the analysis package's
  loader reads, live inside that host's own directory. The host adapter that boots a named
  flight-software instance onto a plant lives in WiseGncDesign, not here.
- `docs/decisions/` — this repository's ADRs.
