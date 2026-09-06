# ICLR 2027 H-SVDQuant Draft

This directory contains an anonymous paper draft and its experiment plan.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Files

- `main.tex`: paper draft using the official ICLR 2027 style.
- `references.bib`: primary related-work references used by the draft.
- `tables/`: current accuracy and runtime-status tables.
- `EXPERIMENT_PLAN.md`: claim-driven experiment matrix and acceptance gates.
- `CLOUD_RUNBOOK.md`: commands for the cloud agent.

## Before submission

- Replace every `\draftnote` and `\pending` entry.
- Add the second model family and matched SVDQuant/LRC rows.
- Resolve cached/no-cache Hadamard generation behavior.
- Include fused runtime numbers only after numerical and three-scope performance
  gates pass.
- Confirm author list, acknowledgements, citations, and AI-use disclosure.
- Keep the initial main text within the ICLR 2027 nine-page limit.
