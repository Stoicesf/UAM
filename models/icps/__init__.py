"""ICPS (Integrated Communication, Positioning, Sensing) — parallel branch package.

Default-off: nothing here is imported by AC-DSGF main training unless
`quality_source=icps_channel` or explicit ICPS experiment scripts enable it.

ponytail: abstract dual-channel + synthetic perception; AAS is SNR calibration only.
"""

from models.icps.channels import ChannelConfig, DualChannelModel
from models.icps.link_quality import sinr_to_quality, soft_packet_loss_prob

__all__ = [
    "ChannelConfig",
    "DualChannelModel",
    "sinr_to_quality",
    "soft_packet_loss_prob",
]
