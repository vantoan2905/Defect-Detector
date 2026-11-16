# DeepStream deployment

## Live RTSP camera (general defect detector)

This full-frame flow accepts an RTSP camera and any YOLO detection model trained with `defectdet`. It uses GPU decoding and TensorRT inference, then writes one JSON line per frame. It does not need the capacitor tray ROIs.

```text
RTSP camera → nvurisrcbin → nvstreammux → nvinfer → metadata probe → fakesink
                                                        ↓
                                                detections.jsonl
```

On a machine with NVIDIA DeepStream, its Python bindings and the trained `best.pt`:

```sh
CUDA_VER=12.6 deploy/deepstream/setup.sh runs/defect_detector/train/weights/best.pt 1024 defect_detector
python3 deploy/deepstream/gen_rtsp_config.py --model-name defect_detector
python3 deploy/deepstream/rtsp.py 'rtsp://user:password@camera-host/stream' \
  --mux-width 1920 --mux-height 1080 --source-name line-1 \
  --out runs/deepstream/rtsp_detections.jsonl
```

Set `CUDA_VER` to the CUDA version included with the installed DeepStream release. The setup command exports ONNX, builds the custom YOLO parser and writes model-specific class labels. The config generator reads those labels to set the class count. The first inference run builds a TensorRT engine. Output records contain `source`, `frame`, `pts_ns` and `detections` (class ID, label, score and `[x1,y1,x2,y2]` box). `pts_ns` is the stream presentation timestamp, not a wall-clock time. If you use a model name other than `defect_detector`, pass its generated config and label file through `--infer-config` and `--labels`.

`--mux-width` and `--mux-height` set the resolution sent into inference. For tiny defects in large frames, full-frame resizing may hide details; use the tiled `nvdspreprocess` pipeline below after adapting its ROIs to the camera and training tiles. The RTSP flow has no renderer and writes detections only.

The RTSP pipeline requires `nvurisrcbin`, `nvstreammux`, `nvinfer` and the DeepStream Python bindings (`gi`, `pyds`). Run it in the DeepStream environment; these bindings are not installed by the Python package extras.

## Capacitor tray images

Runs the capacitor detector with NVIDIA DeepStream / TensorRT on full 5472x3648 frames.
Each frame is cut into the same 9 overlapping tiles used for training (`tiling.ratio_overlap_tiles`),
inferred as one batch, and duplicates from overlapping tiles are merged with NMS.

```
appsrc (images) → nvvideoconvert → nvstreammux → nvdspreprocess (9 ROIs) → nvinfer (YOLOv8)
                → probe: cross-tile NMS, JSON lines → fakesink
```

## Files

| File | Purpose |
| --- | --- |
| `configs/config_preprocess.txt` | ROIs (tiles), network input size and normalisation |
| `configs/config_infer_capacitor.txt` | nvinfer settings, model/engine paths, thresholds |
| `gen_rois.py` | rewrite the ROIs/batch size from `tiling.py` when frame size or tiling changes |
| `setup.sh` | build the DeepStream-Yolo parser, export `best.pt` to ONNX + `labels.txt` |
| `app.py` | the pipeline; writes `{"image", "detections": [{label, score, box}]}` per line |

`DeepStream-Yolo/` and `models/` are created by `setup.sh` and are git-ignored.

## Requirements

- NVIDIA GPU (dGPU or Jetson) with DeepStream 7.x, e.g. the `nvcr.io/nvidia/deepstream` container
- DeepStream Python bindings (`pyds`) from [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
- This package installed in the same environment: `pip install -e ".[yolo]"`

## Run

```sh
# 1. export the trained model (CUDA_VER = CUDA version of your DeepStream release)
CUDA_VER=12.6 deploy/deepstream/setup.sh runs/capacitor_detector/train/weights/best.pt 1344

# 2. run; the first run builds the TensorRT engine (several minutes)
python3 deploy/deepstream/app.py data/interim/test_images --out runs/deepstream/detections.jsonl
```

## Keep in sync with training

- **Input size**: `network-input-shape` / `processing-*` (1344) must equal the export size, which is the
  training `imgsz` rounded up to a multiple of 32.
- **Tiles**: after changing frame size or overlap, run `python deploy/deepstream/gen_rois.py --image-size WxH`.
- **Classes**: `labels.txt` comes from the exported model; `num-detected-classes` must match its line count.

## Not covered yet

- Region detector (OBB): `nvinfer` only outputs axis-aligned boxes. Either retrain it with `task: detect`
  and add it as the primary model, or keep the fixed ROIs (fine while the camera and tray do not move).
- Live camera input: replace `appsrc` with the camera source (e.g. `aravissrc` for GigE Vision).
- INT8: needs a calibration set; validate mAP against the `.pt` model before switching from FP16.
