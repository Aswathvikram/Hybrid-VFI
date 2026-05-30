import math
import subprocess
from pathlib import Path


def interpolate_video(video_path, factor=2):

    if factor not in [2, 4, 8, 16]:
        raise ValueError(
            "factor must be one of: 2, 4, 8, 16"
        )

    video_path = Path(video_path)

    exp = int(math.log2(factor))

    rife_dir = (
        Path(__file__)
        .resolve()
        .parent.parent
        / "third_party"
        / "RIFE"
    )

    cmd = [
        "python",
        "inference_video.py",
        f"--video={video_path}",
        f"--exp={exp}"
    ]

    subprocess.run(
        cmd,
        cwd=rife_dir,
        check=True
    )

    fps = int(factor * 5)  # matches your current naming pattern example

    output_video = (
        video_path.parent
        / f"{video_path.stem}_{factor}X_{fps}fps.mp4"
    )

    return str(output_video)