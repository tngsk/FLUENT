## 2024-05-15 - [Array includes lookup in render loop]
**Learning:** Checking an Array with `.includes()` inside a rendering loop (like iterating through `segments` to compute their status) results in O(n*m) complexity and causes unnecessary UI lag as the dataset grows.
**Action:** Use a `Set` for state variables that require frequent existence checks (e.g., `trainedIds.has()`) to ensure O(1) lookups during DOM rendering, making the overall render O(n).
