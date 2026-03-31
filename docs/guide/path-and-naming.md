# Path And Naming Conventions (Guide Track)

This page records **where Guide content lives** and how URLs map to files. Editors and reviewers use this so new episodes stay consistent.

## Responsibility

- Document stable URL paths for the Engineer Guide and name new dialogue episodes predictably.

## Directory And URL Rules

- **Directory**: all public Guide pages live under `docs/guide/`.
- **Root URL**: `docs/guide/index.md` maps to `/guide/` (site base is applied by VitePress).
- **Episode files**: one Markdown file per episode. File name is **kebab-case**, short, and describes the topic (not episode numbers only), so URLs remain stable if order changes.
- **Internal links**: from Guide pages, link to theory with paths such as [Theory overview](../theory/concepts.md). Prefer repo-relative links consistent with the rest of `docs/`.

## Current Episode Slugs

| Topic (v1) | File name | URL path |
| ------------ | --------- | -------- |
| Continuous optimization for engineers | `continuous-optimization-for-engineers.md` | `/guide/continuous-optimization-for-engineers` |
| Why gradient descent is not enough | `gradient-descent-is-not-enough.md` | `/guide/gradient-descent-is-not-enough` |
| L-BFGS as a practical default | `lbfgs-practical-default.md` | `/guide/lbfgs-practical-default` |

## Maintainer-Only Pages

These support the quality process; they are linked from the Guide index under **For authors and reviewers**.
They are intentionally **not** part of the primary reader-facing Guide sidebar.

| File | Purpose |
| ---- | ------- |
| `dialogue-quality-rubric.md` | Scoring rubric, hard failures, review artifact format, episode template |
| `path-and-naming.md` | This file |
| `reviews/episode-01-review-record.md` | Review record for Episode 1 |
| `reviews/episode-02-review-record.md` | Review record for Episode 2 |
| `reviews/episode-03-review-record.md` | Review record for Episode 3 |

## What Not To Do

- Do not add unfinished dialogue episodes to the top navigation or the primary Guide sidebar until they **pass** the rubric in `dialogue-quality-rubric.md`.
- Do not rename episode files after publication without redirects or an explicit decision (breaks external links).
