# Cover Letter (draft)

**To:** Editor-in-Chief, [IEEE RA-L / T-RO / TASE / TIV]  

**Manuscript title:**  
Adaptive Communication-Constrained Dynamic Spatial Graph Fusion for Multi-UAV Cooperative Navigation

**Dear Editor,**

Please find enclosed our manuscript presenting **AC-DSGF**, a communication-constrained multi-agent reinforcement learning framework for cooperative UAV navigation.

Existing MARL guidance methods typically assume dense or distance-fixed communication topologies. In airborne networks, however, bandwidth, energy, and latency are first-class constraints. AC-DSGF treats the **communication topology as a learnable decision variable**, jointly optimizing task return and communication cost via a soft gate, Top-$K$ budgeting, and residual spatiotemporal guidance.

Empirically, on a 16-UAV continuous navigation benchmark with five seeds, AC-DSGF **maintains cooperative success comparable to DSGF** while reducing communication intensity by approximately **90×**, supported by budget sweeps, silence-collapse ablations, trigger analysis, and gate-stability diagnostics.

We believe this contribution aligns with the journal’s interest in scalable multi-robot / UAV systems under realistic communication limits.

Thank you for your consideration.

Sincerely,  
[Authors]

---

## Suggested keywords
Multi-UAV systems; multi-agent reinforcement learning; communication-efficient cooperation; dynamic graphs; residual guidance
