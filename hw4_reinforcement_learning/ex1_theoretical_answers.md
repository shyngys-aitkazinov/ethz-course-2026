# Exercise 1: Theoretical Answers

## 1. Policy Iteration vs Value Iteration

- **Policy Iteration**: Two phases per iteration -- full policy evaluation (iterates until V^pi converges), then one greedy policy improvement. Fewer outer iterations, but each is expensive.
- **Value Iteration**: Single Bellman optimality backup per iteration: `V(s) = max_a Q(s,a)`. More iterations, but each is cheap. Policy extracted once at the end.

## 2. Discount factor gamma

- **gamma -> 0**: Agent is myopic, only cares about immediate reward. Won't plan long paths to the goal.
- **gamma -> 1**: Agent is far-sighted, plans optimally over long horizons. But convergence slows down since value differences between states shrink.

## 3. Effect of slip probability

- **slip=0.0**: Shortest path along the cliff edge. No risk, so minimize steps.
- **slip=0.01**: Nearly the same -- small risk doesn't outweigh shorter path.
- **slip=0.2**: Agent detours via the top row, staying far from the cliff.

**Why more conservative?** Each step near the cliff risks a -100 penalty with probability proportional to slip_chance. At slip=0.2, the expected cost of walking near the cliff (~-20/step) far exceeds the extra -1/step penalties of the longer safe path.
