"""Speedups for SmolVLA inference, applied as patches on a loaded policy."""
import torch


def batch_vision(policy):
    """Encode every camera in one vision-encoder pass instead of one pass per camera.

    embed_prefix loops over cameras and calls embed_image once each. Here we run all
    images through embed_image as a single batch first, then hand the slices back to
    the loop in order, so the rest of embed_prefix is untouched.
    """
    model = policy.model
    vlm = model.vlm_with_expert
    orig_embed_prefix = model.embed_prefix
    orig_embed_image = vlm.embed_image

    def embed_prefix(images, img_masks, *a, **k):
        bsize = images[0].shape[0]
        embs = orig_embed_image(torch.cat(images, dim=0)).split(bsize, dim=0)
        it = iter(embs)
        vlm.embed_image = lambda _img: next(it)
        try:
            return orig_embed_prefix(images, img_masks, *a, **k)
        finally:
            vlm.embed_image = orig_embed_image

    model.embed_prefix = embed_prefix
    return policy
