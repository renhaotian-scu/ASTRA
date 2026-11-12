
# See LICENSE and THIRD_PARTY_NOTICES.md for source attribution.

from .layers import BasicBlock
from .sampling import AdaptiveDilatedConv

class BasicBlock_FADC(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)

        self.branch2b = AdaptiveDilatedConv(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1)
