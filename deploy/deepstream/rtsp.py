"""Run full-frame YOLO detection on one RTSP stream with NVIDIA DeepStream."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gi
import pyds

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst


def _make(factory: str, name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(factory, name)
    if element is None:
        raise RuntimeError(f"missing GStreamer element: {factory}")
    return element


def build_pipeline(uri: str, infer_config: Path, width: int, height: int, latency: int, reconnect: int):
    pipeline = Gst.Pipeline.new("defect-detector-rtsp")
    source = _make("nvurisrcbin", "source")
    mux = _make("nvstreammux", "mux")
    infer = _make("nvinfer", "infer")
    sink = _make("fakesink", "sink")
    source.set_property("uri", uri)
    source.set_property("latency", latency)
    source.set_property("rtsp-reconnect-interval", reconnect)
    mux.set_property("live-source", 1)
    mux.set_property("batch-size", 1)
    mux.set_property("width", width)
    mux.set_property("height", height)
    mux.set_property("batched-push-timeout", 40000)
    infer.set_property("config-file-path", str(infer_config.resolve()))
    sink.set_property("sync", False)
    for element in (source, mux, infer, sink):
        pipeline.add(element)
    mux_sink = mux.request_pad_simple("sink_0")
    if mux_sink is None:
        raise RuntimeError("cannot request nvstreammux sink_0")

    def link_source(_element, pad):
        if mux_sink.is_linked():
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and caps.to_string().startswith("audio/"):
            return
        if pad.link(mux_sink) != Gst.PadLinkReturn.OK:
            raise RuntimeError("cannot link RTSP decoder to nvstreammux")

    src_pad = source.get_static_pad("src")
    if src_pad is not None:
        link_source(source, src_pad)
    else:
        source.connect("pad-added", link_source)
    if not mux.link(infer) or not infer.link(sink):
        raise RuntimeError("cannot link nvstreammux -> nvinfer -> fakesink")
    return pipeline, infer


def _objects(frame_meta):
    item = frame_meta.obj_meta_list
    while item is not None:
        yield pyds.NvDsObjectMeta.cast(item.data)
        item = item.next


def make_probe(labels: list[str], output, source_name: str):
    def probe(_pad, info):
        buffer = info.get_buffer()
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch is None:
            return Gst.PadProbeReturn.OK
        item = batch.frame_meta_list
        while item is not None:
            frame = pyds.NvDsFrameMeta.cast(item.data)
            detections = []
            for obj in _objects(frame):
                rect = obj.rect_params
                detections.append({
                    "class_id": obj.class_id,
                    "label": labels[obj.class_id] if 0 <= obj.class_id < len(labels) else str(obj.class_id),
                    "score": round(float(obj.confidence), 4),
                    "box": [round(float(v), 1) for v in (
                        rect.left, rect.top, rect.left + rect.width, rect.top + rect.height
                    )],
                })
            output.write(json.dumps({
                "source": source_name,
                "frame": int(frame.frame_num),
                "pts_ns": int(frame.buf_pts),
                "detections": detections,
            }) + "\n")
            output.flush()
            item = item.next
        return Gst.PadProbeReturn.OK

    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uri", help="RTSP URL of the camera")
    parser.add_argument("--infer-config", type=Path, default=Path(__file__).parent / "models/defect_detector_rtsp_infer.txt")
    parser.add_argument("--labels", type=Path, default=Path(__file__).parent / "models/defect_detector_labels.txt")
    parser.add_argument("--out", type=Path, default=Path("runs/deepstream/rtsp_detections.jsonl"))
    parser.add_argument("--source-name", default="camera-0", help="camera identifier in JSONL output")
    parser.add_argument("--mux-width", type=int, default=1920)
    parser.add_argument("--mux-height", type=int, default=1080)
    parser.add_argument("--latency", type=int, default=200, help="RTSP jitter buffer in milliseconds")
    parser.add_argument("--reconnect", type=int, default=10, help="reconnect after this many seconds without RTSP data")
    args = parser.parse_args()
    if not args.uri.startswith(("rtsp://", "rtsps://")):
        parser.error("uri must be an RTSP URL")
    if args.mux_width <= 0 or args.mux_height <= 0 or args.latency <= 0 or args.reconnect < 0:
        parser.error("mux dimensions and latency must be positive; reconnect must be nonnegative")
    if not args.infer_config.is_file() or not args.labels.is_file():
        parser.error("inference config or labels missing; run setup.sh and gen_rtsp_config.py first")
    labels = [line.strip() for line in args.labels.read_text().splitlines() if line.strip()]
    if not labels:
        parser.error("labels file is empty")
    Gst.init(None)
    pipeline, infer = build_pipeline(args.uri, args.infer_config, args.mux_width, args.mux_height, args.latency, args.reconnect)
    loop = GLib.MainLoop()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failure = None
    with args.out.open("w") as output:
        infer.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, make_probe(labels, output, args.source_name))
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_message(_bus, message):
            nonlocal failure
            if message.type == Gst.MessageType.EOS:
                loop.quit()
            elif message.type == Gst.MessageType.ERROR:
                failure, debug = message.parse_error()
                print(f"GStreamer error: {failure}: {debug}", file=sys.stderr)
                loop.quit()

        bus.connect("message", on_message)
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("could not start DeepStream pipeline")
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            pipeline.set_state(Gst.State.NULL)
            bus.remove_signal_watch()
    if failure is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
