"""Validate an object detector."""

from scripts._runner import evaluate
from scripts.object_detection.models import TASK


def main(argv: list[str] | None = None) -> None:
    evaluate(TASK, "val", argv)


if __name__ == "__main__":
    main()
