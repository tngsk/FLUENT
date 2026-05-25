## 2025-05-25 - Frontend Rendering and State Lookup Optimization
**Learning:** Vanilla JS rendering of long lists directly to the DOM triggers repeated reflows, and Array.includes() for state checks in rendering loops creates O(N) bottlenecks.
**Action:** Always batch DOM insertions using DocumentFragment and use Set.has() for O(1) state lookups in tight rendering loops.
