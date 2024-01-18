from video import Video


def main():
    video = Video("./footage/footage1.mp4", conf_threshold=0.15)
    video.process()


if __name__ == "__main__":
    main()
