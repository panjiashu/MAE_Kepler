from torch import nn as nn

from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert

class PatchEmbed(nn.Module):
    """ 1D light curve to Patch Embedding
    """
    def __init__(
            self,
            img_size=4000,
            patch_size=40,
            in_chans=1,
            embed_dim=128,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = img_size
        patch_size = patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size
        self.flatten = flatten

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        _assert(L == self.img_size, f"Input light curve length ({L}) doesn't match model ({self.img_size}).")
        x = self.proj(x)
        x = x.transpose(0, 1)  # BCL -> BLC
        x = self.norm(x)
        return x