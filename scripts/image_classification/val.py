"""Validate an image classifier."""

from scripts._runner import evaluate
from scripts.image_classification.models import TASK


def main(argv: list[str] | None = None) -> None:
    evaluate(TASK, "val", argv)


if __name__ == "__main__":
    main()
