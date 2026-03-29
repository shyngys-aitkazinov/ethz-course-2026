# HW2 — Theoretical Question Answers

## Exercise 1: Inverse Kinematics

**Q1: If you increase the width of the Lemniscate (increasing `a`), what issue can happen with the robot performing IK?**

The keypoints can move outside the robot's reachable workspace or push the arm near singularities, causing IK to fail.

**Q2: What can happen if you change the `dt` parameter in IK?**

If `dt` is too large - joint updates overshoot and IK oscillates or diverges; if `dt` is too small convergence becomes very slow and hit the iteration limit.

**Q3: We implemented a simple numerical IK solver. What are the advantages and disadvantages compared to an analytical IK solver?**

Numerical IK is general-purpose and works for any robot geometry, but it is slower and not guaranteed to find a solution, whereas analytical IK is fast and exact but only exists for specific robot structures.

**Q4: What are the limits of our IK solver compared to state-of-the-art IK solvers?**

Our solver ignores joint limits, handles only position (not orientation), uses a fixed damping factor, and has no redundancy resolution compared to SOTA solvers.

---

## Exercise 2: PID Control

**Q1: If you keep increasing Kp, what issue arises when tracking the waypoints?**

The robot begins to overshoot and oscillate around the target waypoints.

**Q2: How does Kd mitigate the effect you saw above when increasing Kp?**

Kd adds damping by opposing the rate of change of the error, which slows the robot down as it approaches the target and reduces overshoot and oscillation. (Like a string).

**Q3: In what scenarios is a non-zero Ki needed for the controller to perform well?**

Ki is needed when there is a persistent steady-state error that P and D alone cannot eliminate, such as when gravity or friction creates a constant offset from the target.

---

## Exercise 3: RL Policy (Bonus)

**Q (Bonus): What difference can you observe when the RL policy tracks the Lemniscate curve compared to PID?**

The PID controller tracks the lemniscate smoothly because it was designed for this specific trajectory pipeline (IK + waypoints), while the RL policy was only trained on random static targets and struggles with continuous trajectory tracking — it overshoots, lags behind, or oscillates between keypoints because it never learned smooth sequential motion.

**Q (Bonus): What changes can improve the RL policy?**

Possible improvements:

- **Reward shaping:** Add penalties for large joint velocities/accelerations to encourage smoother motion
- **Richer observations:** Include joint velocities (`qvel`) and/or the direction to the target in the observation vector