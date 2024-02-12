import argparse
from video import Video


def main():
    parser = argparse.ArgumentParser(
        description="traffilyzer"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to YOLOv8 model",
        type=str
    )

    parser.add_argument(
        "--conf",
        required=False,
        default=0.3,
        help="Model confidence threshold",
        type=float
    )

    parser.add_argument(
        "--iou",
        required=False,
        default=0.7,
        help="Model IOU threshold",
        type=float
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to source video file",
        type=str
    )

    args = parser.parse_args()
    video = Video(
        model=args.model,
        video_file=args.video,
        conf_threshold=args.conf,
        iou=args.iou
    )
    video.process()


if __name__ == "__main__":
    main()
