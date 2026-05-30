import cv2
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


def interpolate_video(video_path):

    video_path = Path(video_path)

    output_video = (
        video_path.parent.parent
        / "outputs"
        / f"{video_path.stem}_sepconv.mp4"
    )

    # Locate SepConv
    sepconv_dir = (
        Path(__file__)
        .resolve()
        .parent.parent
        / "third_party"
        / "SepConv"
    )

    sys.path.append(str(sepconv_dir))

    old_argv = sys.argv
    sys.argv = [""]

    import run

    sys.argv = old_argv

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps * 2,
        (width, height)
    )

    ret, prev_frame = cap.read()

    if not ret:
        raise ValueError(
            "Could not read video."
        )

    pbar = tqdm(
        total=total_frames - 1,
        desc="SepConv"
    )

    while True:

        ret, curr_frame = cap.read()

        if not ret:
            break

        ten1 = torch.FloatTensor(
            cv2.cvtColor(
                prev_frame,
                cv2.COLOR_BGR2RGB
            ).transpose(2, 0, 1)
        ) / 255.0

        ten2 = torch.FloatTensor(
            cv2.cvtColor(
                curr_frame,
                cv2.COLOR_BGR2RGB
            ).transpose(2, 0, 1)
        ) / 255.0

        middle = run.estimate(
            ten1,
            ten2
        )

        middle_frame = (
            middle
            .clamp(0, 1)
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            * 255.0
        ).astype(np.uint8)

        middle_frame = cv2.cvtColor(
            middle_frame,
            cv2.COLOR_RGB2BGR
        )

        writer.write(prev_frame)
        writer.write(middle_frame)

        prev_frame = curr_frame

        pbar.update(1)

    writer.write(prev_frame)

    pbar.close()

    cap.release()
    writer.release()

    return str(output_video)