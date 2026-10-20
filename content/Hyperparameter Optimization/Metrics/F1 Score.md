##### Definition
==Balances false positives and false negatives==

$$

\begin{aligned}

\text{F1} &= 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}

\end{aligned}

$$

- **Interpretation:** Balances precision and recall into one score
- Goal: maximize, as close to 1 as possible.
- **Use when:**
	- ==Imbalanced datasets (preferred over accuracy)==
	- You care equally about precision AND recall
	- Accuracy alone would hide poor performance on a minority class

**Why harmonic mean?** Heavily penalizes extreme imbalances

- Precision 100%, Recall 50% → F1 ≈ 67% (not 75%)

- A high score requires both precision and recall to be reasonably high.

For multiclass tasks, specify how class-level F1 scores should be averaged (macro, weighted, or micro); the choice changes how class imbalance affects the result.