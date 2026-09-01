# Baselines — notes for §6.6

**Parent design:** [`tro_experimental_design.md`](tro_experimental_design.md)  
**Formal protocol:** [`tro_6_6_formal_protocol.md`](tro_6_6_formal_protocol.md)  
**Status:** Phase A taxonomy locked; Class C deferred.

| Class | Methods (Phase A) | Role |
|-------|-------------------|------|
| A No comm learning | MAPPO | necessity of communication |
| B Fixed / non-budgeted graph | GAT-MAPPO, DSGF | necessity of budgeted dynamic topology |
| G2 Ceiling | Full Attention | cost–performance upper reference |
| Ours | AC-DSGF (+ hard \(K\) slices) | topology decision + \(\Pi_{B_t}\) |
| C Learned communication | TarMAC, IC3Net, ATOC, MAGIC | **Phase B — not yet implemented** |

Always report joint \((J,C)\) / identical-budget slices.
