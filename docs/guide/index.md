# Engineer Guide (Dialogue Track)

This track is for **engineers who are new to continuous optimization**. It uses a teacher / learner dialogue to build intuition and motivation before you dive into the evidence-first Theory pages and code.

## How this differs from Theory

| Track | You read it for… |
| ----- | ---------------- |
| **Guide (here)** | Plain-language motivation, typical use cases, safe intuition, and “what to read next.” |
| **[Theory](../theory/concepts.md)** | Precise notation, algorithm statements, Wolfe line search, invariants, and implementation anchors. |
| **[Evidence](../evidence/baseline_results.md)** | Whether this repository’s implementations match reference behavior under stated methodology. |

The Theory pages remain the **canonical** reference for definitions and mathematics. If anything here ever disagrees with Theory, **Theory wins** and this Guide should be revised.

## Episode list (v1)

1. [Continuous optimization for engineers](./continuous-optimization-for-engineers.md) — what we are minimizing, and why it shows up in engineering work.
2. [Why gradient descent is not enough](./gradient-descent-is-not-enough.md) — scaling, ill-conditioning, and why “first-order only” methods can feel stuck.
3. [Why L-BFGS is often the practical default](./lbfgs-practical-default.md) — memory, curvature heuristics, and **scoped** claims about when L-BFGS is a strong default.

## For authors and reviewers

- [Path and naming conventions](./path-and-naming.md) — where files live and how URLs are chosen.
- [Dialogue quality rubric](./dialogue-quality-rubric.md) — pass/fail scoring, hard failures, required section template.
- [Episode 1 review record](./reviews/episode-01-review-record.md) — scored review cycle with one revision.
- [Episode 2 review record](./reviews/episode-02-review-record.md)
- [Episode 3 review record](./reviews/episode-03-review-record.md)

Do **not** link unfinished episodes from the site navigation until they pass the rubric.

## Read next

After Episode 3, continue with [Theory overview](../theory/concepts.md), then [L-BFGS](../theory/lbfgs.md) when you are ready for algorithm detail.
