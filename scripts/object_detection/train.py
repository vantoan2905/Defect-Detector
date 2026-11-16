"""Train an object detector."""

from scripts._runner import train
from scripts.object_detection.models import DEFAULT_WEIGHTS, TASK


def main(argv: list[str] | None = None) -> None:
    train(TASK, DEFAULT_WEIGHTS, argv)


if __name__ == "__main__":
    main()
