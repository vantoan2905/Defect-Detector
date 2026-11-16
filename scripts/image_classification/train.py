"""Train an image classifier."""

from scripts._runner import train
from scripts.image_classification.models import DEFAULT_WEIGHTS, TASK


def main(argv: list[str] | None = None) -> None:
    train(TASK, DEFAULT_WEIGHTS, argv, no_augment=True)


if __name__ == "__main__":
    main()
