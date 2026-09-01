"""Encode demo frames (PNG sequence) into mp4 / gif."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def frames_to_mp4(frame_dir: Path, out_mp4: Path, fps: int = 10) -> Path:
    """Write mp4 from sorted PNG frames. Prefers imageio, falls back to matplotlib."""
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio

        imgs = [imageio.imread(f) for f in frames]
        imageio.mimsave(str(out_mp4), imgs, fps=fps)
        return out_mp4
    except Exception:
        pass

    try:
        import cv2

        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_mp4),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for f in frames:
            writer.write(cv2.imread(str(f)))
        writer.release()
        return out_mp4
    except Exception:
        pass

    # Fallback: animated GIF (always available via matplotlib)
    out_gif = out_mp4.with_suffix(".gif")
    frames_to_gif(frame_dir, out_gif, fps=fps)
    return out_gif


def frames_to_gif(frame_dir: Path, out_gif: Path, fps: int = 8) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from PIL import Image

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    imgs = [np.asarray(Image.open(f).convert("RGB")) for f in frames]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(imgs[0])

    def _update(k):
        im.set_data(imgs[k])
        return (im,)

    anim = animation.FuncAnimation(fig, _update, frames=len(imgs), interval=1000 / fps, blit=True)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer="pillow", fps=fps)
    plt.close(fig)
    return out_gif
