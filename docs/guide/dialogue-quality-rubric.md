# Dialogue Quality Rubric And Episode Template

This page defines the **quality line** for English dialogue episodes in the Engineer Guide. A page is accepted only when it passes this rubric and **none** of the hard-failure rules apply.

## Responsibility

- Give authors and reviewers a **repeatable** way to decide pass or rewrite.
- Keep the Guide complementary to [Theory](../theory/concepts.md): motivation and bridges, not duplicate proofs or full algorithm details.

## Locked Audience And Claims

- **Reader**: engineers who are **new to optimization** but can read code and technical prose.
- **Language**: English only for Guide episodes.
- **L-BFGS positioning**: describe it as a **practical default** mainly for **large-scale smooth** problems with **reliable gradients**, not as a universal best method.

## Required Sections In Every Episode

Each published episode **must** include these headings (exact order):

1. `## Who this is for`
2. `## What you will understand after this page`
3. `## Dialogue`
4. `## Practical takeaway for engineers`
5. `## When this framing breaks down`
6. `## Strict reference` (links to canonical Theory / References pages)
7. `## Read next`

If you change this structure, record an explicit reviewer-approved reason in the review artifact.

## Scoring Rubric (1–5 Per Criterion)

Score each row **independently**. Use integers 1–5.

| ID | Criterion | Score 5 means |
| -- | --------- | ------------- |
| R1 | Target reader fit | Clearly written for engineers new to optimization; not infantilizing a general audience. |
| R2 | Conceptual accuracy | Consistent with canonical docs; L-BFGS scope not overstated. |
| R3 | Practical relevance | Ties to real engineering tasks (fitting, losses, calibration, tuning). |
| R4 | Dialogue naturalness | Teacher / learner exchange supports explanation; not forced theater. |
| R5 | Clarity before notation | Meaning comes before symbols; reader keeps thread of topic. |
| R6 | Learning outcome strength | Post-reading competency is explicit; one main theme per page. |
| R7 | Bridge quality | `Strict reference` sends readers to the right canonical pages without competing. |
| R8 | Tone discipline | Plain professional English; not comedy-first. |

### Pass Threshold

- **Reject** if any criterion scores **3 or below** (treat as “4 未満” in the integrated plan: strictly, scores below 4 fail — i.e. **≤3 fails**).
- **Accept** only if **every criterion is ≥4** and the **average is ≥4.2**.

### Hard Failure (Automatic Reject)

Reject immediately if **any** of the following holds:

- L-BFGS is described as the **universal best** optimizer.
- Scope and limitations for L-BFGS (where relevant) are **missing** on pages that recommend it.
- The episode does not explain **why the idea matters in engineering work**.
- Avoiding math makes the story **wrong** or misleading versus canonical docs.
- The learner role is **mockingly ignorant** or insulting to the reader.
- The teacher delivers **long monologues** so the dialogue format adds no value.
- Long **duplicate** derivations that already live in Theory pages.
- `Strict reference` is missing or points to the wrong / vague destination.

## Review Artifact Format

For each episode, maintain a review record under `docs/guide/reviews/` named `episode-<slug>-review-record.md`.

Include:

| Field | Meaning |
| ----- | ------- |
| Episode | Link to the episode file path |
| Draft | Draft number (`Draft 1`, `Draft 2`, …) |
| Date | ISO date of review |
| Rubric scores | Table R1–R8 |
| Hard failures | `none` or list violated rules |
| Verdict | `reject` or `accept` |
| Weak points | One sentence per low score |
| Revision plan | Bullet points: what to change before next draft |
| Re-score | Repeat after revision |

Publish to navigation **only** after an `accept` verdict.

## Revision Loop (Operational)

1. Write a draft.
2. Self-score R1–R8; list hard failures.
3. Have a second reviewer score independently when possible.
4. If rejected, **name weak points in one sentence each** and revise **locally** before rewriting whole-cloth.
5. Repeat until accept threshold is met.

## Relation To Canonical Theory

- **Canonical** definitions, symbols, Wolfe conditions, and update formulas live under [Theory](../theory/concepts.md) and method pages.
- **Guide** episodes narrate *why* and *when* and link out; they must not become a second copy of the same math.
