"""Backward-compatible entry point for MixedWM38 classification labels."""

if __package__:
    from scripts.image_classification.label import main, prepare
else:
    from image_classification.label import main, prepare

__all__ = ["main", "prepare"]


if __name__ == "__main__":
    main()
