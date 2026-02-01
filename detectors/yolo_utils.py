from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    # xywh: [...,4], x,y are center
    x, y, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)

def nms(dets: List[Dict[str, Any]], iou_thres: float) -> List[Dict[str, Any]]:
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep: List[Dict[str, Any]] = []
    for d in dets:
        dbox = np.array(d["xyxy"], dtype=np.float32)
        ok = True
        for k in keep:
            if d["class_id"] != k["class_id"]:
                continue
            kbox = np.array(k["xyxy"], dtype=np.float32)
            if iou_xyxy(dbox, kbox) > iou_thres:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep

def _scale_xyxy(dets: List[Dict[str, Any]], input_size: int, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
    sx = orig_w / float(input_size)
    sy = orig_h / float(input_size)
    out: List[Dict[str, Any]] = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        out.append({**d, "xyxy": [float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)]})
    return out

def parse_yolo_output(
    outputs: List[np.ndarray],
    input_size: int,
    orig_w: int,
    orig_h: int,
    conf_thres: float,
    iou_thres: float,
    num_classes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Parse common Ultralytics-style ONNX outputs into detections.

    Supported:
    - Already decoded: (n,6) => x1,y1,x2,y2,conf,class
    - YOLOv5: (1, N, 5+nc) => xywh,obj,cls...
    - YOLOv8: (1, N, 4+nc) OR (1, 4+nc, N) => xywh,cls... (no obj)

    Returns list of dicts:
    {xyxy:[x1,y1,x2,y2], conf:float, class_id:int}
    """
    if not outputs:
        return []
    out = outputs[0]
    if out is None:
        return []

    out = np.array(out)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    # (n,6)
    if out.ndim == 2 and out.shape[1] == 6:
        dets = []
        for row in out:
            conf = float(row[4])
            if conf < conf_thres:
                continue
            dets.append({"xyxy": row[:4].astype(np.float32).tolist(), "conf": conf, "class_id": int(row[5])})
        return nms(_scale_xyxy(dets, input_size, orig_w, orig_h), iou_thres)

    # YOLOv8 (4+nc, N) -> transpose
    if out.ndim == 2 and out.shape[0] in (4, 5) and out.shape[1] > 1000:
        out = out.T  # (N, 4+nc)

    if out.ndim != 2 or out.shape[1] < 6:
        return []

    # infer nc
    if num_classes is None:
        # If looks like YOLOv5: 5+nc with obj at index 4.
        # Heuristic: if (shape[1] - 5) is small-ish, assume YOLOv5.
        if out.shape[1] >= 7:
            nc_v5 = out.shape[1] - 5
            nc_v8 = out.shape[1] - 4
            # prefer YOLOv5 if nc_v5 in 1..200
            if 1 <= nc_v5 <= 200:
                nc = nc_v5
                is_v5 = True
            else:
                nc = nc_v8
                is_v5 = False
        else:
            nc = out.shape[1] - 4
            is_v5 = False
    else:
        # if num_classes explicitly set, assume v5 if 5+nc matches
        is_v5 = (out.shape[1] == 5 + num_classes)
        nc = num_classes

    dets: List[Dict[str, Any]] = []
    if is_v5 and out.shape[1] == 5 + nc:
        xywh = out[:, :4]
        obj = out[:, 4]
        cls_scores = out[:, 5:5+nc]
        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
        conf = obj * cls_conf
        keep = conf >= conf_thres
        xywh = xywh[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]
        xyxy = xywh_to_xyxy(xywh)
        for i in range(xyxy.shape[0]):
            dets.append({"xyxy": xyxy[i].tolist(), "conf": float(conf[i]), "class_id": int(cls_id[i])})
    else:
        # YOLOv8: xywh + class probs (no obj)
        xywh = out[:, :4]
        cls_scores = out[:, 4:4+nc]
        cls_id = np.argmax(cls_scores, axis=1)
        conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
        keep = conf >= conf_thres
        xywh = xywh[keep]
        conf = conf[keep]
        cls_id = cls_id[keep]
        xyxy = xywh_to_xyxy(xywh)
        for i in range(xyxy.shape[0]):
            dets.append({"xyxy": xyxy[i].tolist(), "conf": float(conf[i]), "class_id": int(cls_id[i])})

    return nms(_scale_xyxy(dets, input_size, orig_w, orig_h), iou_thres)
