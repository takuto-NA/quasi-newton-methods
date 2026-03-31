# Why Gradient Descent Is Not Enough

## Who this is for

Engineers who understand “step opposite the gradient,” but wonder why practical optimizers use more machinery than vanilla gradient descent.

## What you will understand after this page

- Why **scaling** and **curvature** make naive steepest descent slow in realistic losses.
- Why engineers add diagonal preconditioning, better directions, or second-order *approximations*.
- How this motivates **quasi-Newton** methods without deriving their formulas here.

## Dialogue

**Learner:** If the gradient points downhill, why not always move a little in $-\nabla f(x)$?

**Teacher:** You can, and people do: that is steepest descent. It is stable if you choose the step size sensibly. The pain is **how many steps** you need when the valley is elongated or poorly scaled.

**Learner:** What do you mean by elongated?

**Teacher:** Near a minimum, many objectives look like a narrow banana-shaped valley. Steepest descent can **zig-zag** across the valley floor because the locally steepest downhill direction is not aimed along the valley toward the low point.

**Learner:** Is this only a geometry metaphor?

**Teacher:** It shows up numerically as **ill-conditioning**: some directions in parameter space change $f$ much faster than others. If you use a single global learning rate, you either crawl along flat directions or overshoot sharp ones.

**Learner:** I have seen people tweak learning rates, Adam, batch size… is that the same issue?

**Teacher:** Same family of issues: you are fighting curvature and noise without explicitly modeling curvature. Adaptive first-order methods change the effective scaling. Newton-like ideas try to account for curvature more directly, but they get expensive at large dimension.

**Learner:** Where do quasi-Newton methods sit?

**Teacher:** They build an **approximation of curvature information** using only gradients from recent steps. That often gives much better directions than raw steepest descent, but stays cheaper than forming a full Hessian every iteration.

**Learner:** So gradient descent is not “wrong,” just sometimes slow?

**Teacher:** Correct. It is a baseline. When dimension is modest and time is cheap, simple methods can be enough. When evaluations are expensive, dimensions are large, and you have trustworthy gradients, better directions usually pay off.

## Practical takeaway for engineers

If progress stalls while gradients look reasonable, suspect **scaling and curvature**, not just “hyperparameter luck.” Before jumping to a fancy optimizer, log what matters: objective value, gradient norm, step sizes, and whether parameters oscillate. Then consider methods that approximate curvature—BFGS-family methods are a common family for smooth problems.

## When this framing breaks down

- Gradients are **noisy or wrong** (buggy autograd, bad finite differences, stochastic objectives handled poorly).
- The objective is **non-smooth** (then quasi-Newton assumptions may not apply without extra care).
- Constraints dominate the geometry (projected or constrained methods are needed).

## Strict reference

For descent directions, inner products, and why quasi-Newton updates matter in this project, read [Theory overview](../theory/concepts.md). The full-memory update narrative starts in [BFGS](../theory/bfgs.md).

## Read next

[Why L-BFGS is often the practical default](./lbfgs-practical-default.md)
