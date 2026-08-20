# Project-page asset staging

Drop site assets here. This whole folder is git-ignored (see `.gitignore`), so
nothing you put here lands on `main`. During the site build these files are
copied onto the `gh-pages` orphan branch under `static/`.

## Where things go

- `static/figures/` — main manuscript figures for the page body.
  Suggested names (used by the Results/Method sections):
  - `fig1_*.png` … `fig5_*.png` (or `.svg`). Web-friendly export, ~1600 px wide,
    < ~500 KB each. PNG or SVG both fine.
- `static/states/` — individual per-state brain renders for the interactive
  teaser (regenerated via yabplot poster-components mode; see the spec §
  regeneration). One image per state, e.g.
  `sub-01_rank1.png`, `sub-01_rank2.png`, …, `sub-06_rank5.png`.
  A `states.json` manifest (built during implementation) maps each image to its
  subject, recurrence score, and dominant network.

## Also needed (paste into the design/spec, not files here)

- Author homepage URLs + affiliation strings.
- Final abstract + key-results wording (or confirm the README overview is fine).
