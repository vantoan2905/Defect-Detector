# Local data

Everything here except this README is ignored by Git. For the general workflow, place image and sibling Labelme JSON pairs under `raw/<project>/`, then run `defectdet prepare` to create `processed/<project>/`.

```text
data/
├── raw/my_product/          # image_001.png + image_001.json, etc.
├── processed/defects/       # generated YOLO dataset, including dataset.yaml
├── interim/                 # optional working files
└── archives/                # original data archives
```

The capacitor example uses `raw/` batch folders, `interim/` cell CSVs and tiles, and `processed/{region_detector,capacitor_detector}/`. See [the capacitor example](../docs/capacitor-example.md).
