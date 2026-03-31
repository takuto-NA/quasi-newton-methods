# Why L-BFGS Is Often The Practical Default

## Who this is for

Engineers who want a **scoped** explanation of when **L-BFGS** is a strong default among classical smooth optimizers, without overstating “always best.”

## What you will understand after this page

- What problem class L-BFGS targets: **smooth**, **large-scale**, often **unconstrained** (or handled via a constrained variant such as L-BFGS-B).
- How L-BFGS relates to **BFGS** and why **limited memory** matters in $\mathbb{R}^n$ when $n$ is large.
- Why it is reasonable to call L-BFGS a **practical default** in that scoped setting—and when it is **not**.

## Dialogue

**Learner:** Everyone says “try L-BFGS.” Why is it the default story for smooth optimization?

**Teacher:** Think of the goal: find a search direction that uses **approximate curvature**, like BFGS, but avoid storing an $n \times n$ matrix when $n$ is huge. L-BFGS keeps a **short history** of past steps and gradient differences to reconstruct useful curvature information cheaply.

**Learner:** So it is still “only gradients,” like BFGS?

**Teacher:** For a smooth objective, you typically evaluate the **gradient** at iterates and run a **line search** along each direction. This project’s Theory pages document strong Wolfe line search as part of that story.

**Learner:** When is it a *practical* default?

**Teacher:** In **large-scale smooth problems** with reliable gradients—classic examples include certain scientific computing objectives, some statistical estimation problems, and some differentiable engineering surrogates—L-BFGS is often the first quasi-Newton you try because it balances memory and progress per iteration.

**Learner:** “Default” sounds universal. You sound cautious.

**Teacher:** Good catch. L-BFGS is **not** the universal best optimizer. It is a strong **first candidate** inside the scoped family: smooth objective, moderate noise in gradients, you can afford gradient evaluations, and dimension is large enough that full-matrix BFGS is too heavy.

**Learner:** When should I *not* assume L-BFGS?

**Teacher:** When the objective is **non-smooth**, when gradients are **unavailable or untrustworthy**, when **noise dominates** unless you use specialized stochastic methods, or when **general constraints** reshape the problem so projection or other constrained algorithms matter more. Also, if dimension is tiny, simpler methods may suffice.

**Learner:** What do I read before touching implementations?

**Teacher:** Read the L-BFGS reference page for the actual update logic and invariants you debug in code. Then read Evidence if you need to see how this repository validates against references.

## Practical takeaway for engineers

For **large-scale smooth** minimization with solid gradients, **try L-BFGS early**, but pair that default with a checklist: smoothness, line-search behavior, gradient correctness, and whether your problem is actually unconstrained in the way your solver assumes.

## When this framing breaks down

- The objective is **non-smooth** or contains kinks that break smooth quasi-Newton assumptions.
- Gradients are **unavailable, noisy, or untrustworthy**, so curvature estimates inherit bad signals.
- The problem is dominated by **general constraints** rather than the unconstrained geometry L-BFGS exploits.
- The setting is highly **stochastic**, so specialized stochastic optimizers may be a better first candidate.
- The problem is small enough that a simpler method is easier to reason about and maintain.

## Strict reference

- Algorithm detail and debugging invariants: [L-BFGS](../theory/lbfgs.md)
- Bound-constrained variant pointer: [L-BFGS-B](../theory/lbfgsb.md)
- SciPy context: [SciPy reference](../references/scipy.md) and [Implementation notes](../references/implementations.md)

## Read next

[Theory overview](../theory/concepts.md), then [L-BFGS](../theory/lbfgs.md) for the exact two-loop recursion and implementation alignment.
