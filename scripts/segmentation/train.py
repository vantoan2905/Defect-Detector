"""Train an instance segmentation model."""

from scripts._runner import train
from scripts.segmentation.models import DEFAULT_WEIGHTS, TASK


def main(argv: list[str] | None = None) -> None:
    train(TASK, DEFAULT_WEIGHTS, argv)


if __name__ == "__main__":
    main()
