# Channel Model — notes for §6.5

**Formal protocol:** [`tro_6_5_formal_protocol.md`](tro_6_5_formal_protocol.md)  
**Status:** parameters + claim hygiene frozen.

| Factor | Levels |
|--------|--------|
| Packet loss \(p_\ell\) | 0, 0.1, 0.3, 0.5 |
| Delay \(\tau\) | 0, 20, 50, 100 ms → steps 0, 1, 2, 4 |
| Bandwidth / \(B_t\) | \(B/B_{\mathrm{full}}\in\{0.1,0.2,0.4,0.8\}\) |

Keep RF/Shannon theory out of scientific claims; use channel as **stress test** for adaptive topology.
