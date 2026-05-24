## 2024-05-24 - Frontend rendering optimization
**Learning:** O(N) array lookups inside rendering loops scale poorly with large segment lists. Direct DOM append operations in loops cause reflow overhead.
**Action:** Always use Set for repeated inclusion checks (O(1)) and DocumentFragment for batching DOM insertions to minimize browser layout recalculations.
