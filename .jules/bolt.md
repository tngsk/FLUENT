## 2026-05-27 - Optimize render loop
**Learning:** O(N^2) render loop and DOM thrashing cause performance issues
**Action:** Use Set for lookups and DocumentFragment for DOM insertions
