#!/usr/bin/env python3
"""多目标自动跟随脚本（红底数字目标）。

能力说明：
1. 检测红色背景、固定尺寸的目标（数字内容可变化）。
2. 多目标同时出现时，支持“离鼠标最近锁定”和“循环切换目标”。
3. 目标移动或短暂丢失时，使用速度预测进行重捕获。
4. 鼠标脱离目标后，可自动重新拉回到锁定目标中心。

热键：
- F8  : 开/关自动跟随
- F9  : 锁定离当前鼠标最近的目标
- F10 : 在可见目标之间循环切换
- F11 : 取消当前锁定
- ESC : 退出程序
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
import pyautogui
from pynput import keyboard

pyautogui.FAILSAFE = False


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


@dataclass
class Detection:
    bbox: BBox
    center: Point
    score: float


@dataclass
class Track:
    track_id: int
    bbox: BBox
    center: Point
    velocity: Point
    missing: int = 0
    last_seen_frame: int = 0


class RedTargetDetector:
    """通过 HSV 红色阈值检测目标。"""

    def __init__(
        self,
        min_area: int,
        max_area: int,
        min_aspect: float,
        max_aspect: float,
        template_path: Optional[str] = None,
        template_iou_threshold: float = 0.55,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.template_iou_threshold = template_iou_threshold
        self.template_red_mask = self._load_template_mask(template_path)

        self.lower_red_1 = np.array([0, 90, 70], dtype=np.uint8)
        self.upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        self.lower_red_2 = np.array([160, 90, 70], dtype=np.uint8)
        self.upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        red_mask = self._build_red_mask(hsv)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        frame_h, frame_w = frame_bgr.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(h, 1)
            if area < self.min_area or area > self.max_area:
                continue
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame_w, x + w + pad)
            y2 = min(frame_h, y + h + pad)
            candidate = frame_bgr[y1:y2, x1:x2]
            if candidate.size == 0:
                continue

            iou_score = self._template_iou(candidate)
            if iou_score < self.template_iou_threshold:
                continue

            bbox = (x1, y1, x2, y2)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            detections.append(Detection(bbox=bbox, center=center, score=iou_score))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections, red_mask

    def _build_red_mask(self, hsv: np.ndarray) -> np.ndarray:
        mask1 = cv2.inRange(hsv, self.lower_red_1, self.upper_red_1)
        mask2 = cv2.inRange(hsv, self.lower_red_2, self.upper_red_2)
        merged = cv2.bitwise_or(mask1, mask2)
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, self.kernel, iterations=1)
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return merged

    def _load_template_mask(self, template_path: Optional[str]) -> Optional[np.ndarray]:
        if not template_path:
            return None
        template = cv2.imread(template_path)
        if template is None:
            raise FileNotFoundError(f"模板图读取失败: {template_path}")
        hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        mask = self._build_red_mask(hsv)
        if np.count_nonzero(mask) == 0:
            raise ValueError("模板图中没有检测到红色背景，请检查模板")
        return cv2.resize(mask, (40, 40), interpolation=cv2.INTER_NEAREST)

    def _template_iou(self, candidate_bgr: np.ndarray) -> float:
        if self.template_red_mask is None:
            return 1.0
        hsv = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2HSV)
        mask = self._build_red_mask(hsv)
        mask = cv2.resize(mask, self.template_red_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
        intersection = np.logical_and(mask > 0, self.template_red_mask > 0).sum()
        union = np.logical_or(mask > 0, self.template_red_mask > 0).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)


class MultiObjectTracker:
    """简易多目标跟踪：基于距离关联和速度预测。"""

    def __init__(self, max_match_distance: float, max_missing_frames: int) -> None:
        self.max_match_distance = max_match_distance
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 1
        self.frame_idx = 0
        self.tracks: Dict[int, Track] = {}
        self.locked_track_id: Optional[int] = None

    def update(self, detections: List[Detection]) -> None:
        self.frame_idx += 1
        unmatched_track_ids = set(self.tracks.keys())
        matched_track_ids = set()

        for det in detections:
            best_track_id = None
            best_dist = float("inf")
            for track_id in unmatched_track_ids:
                dist = _distance(self.tracks[track_id].center, det.center)
                if dist < best_dist and dist <= self.max_match_distance:
                    best_track_id = track_id
                    best_dist = dist

            if best_track_id is None:
                self.tracks[self.next_track_id] = Track(
                    track_id=self.next_track_id,
                    bbox=det.bbox,
                    center=det.center,
                    velocity=(0.0, 0.0),
                    missing=0,
                    last_seen_frame=self.frame_idx,
                )
                self.next_track_id += 1
            else:
                track = self.tracks[best_track_id]
                vx = det.center[0] - track.center[0]
                vy = det.center[1] - track.center[1]
                track.velocity = (vx, vy)
                track.center = det.center
                track.bbox = det.bbox
                track.missing = 0
                track.last_seen_frame = self.frame_idx
                unmatched_track_ids.remove(best_track_id)
                matched_track_ids.add(best_track_id)

        for track_id in list(self.tracks.keys()):
            if track_id in matched_track_ids:
                continue
            track = self.tracks[track_id]
            track.missing += 1
            if track.missing > self.max_missing_frames:
                del self.tracks[track_id]

        if self.locked_track_id is not None and self.locked_track_id not in self.tracks:
            self.locked_track_id = None

    def visible_track_ids(self) -> List[int]:
        ids = [t.track_id for t in self.tracks.values() if t.missing == 0]
        ids.sort()
        return ids

    def lock_nearest(self, point: Point) -> Optional[int]:
        nearest_id = None
        nearest_dist = float("inf")
        for track in self.tracks.values():
            if track.missing != 0:
                continue
            dist = _distance(track.center, point)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = track.track_id
        self.locked_track_id = nearest_id
        return nearest_id

    def cycle_lock(self) -> Optional[int]:
        visible_ids = self.visible_track_ids()
        if not visible_ids:
            self.locked_track_id = None
            return None
        if self.locked_track_id not in visible_ids:
            self.locked_track_id = visible_ids[0]
            return self.locked_track_id
        curr_idx = visible_ids.index(self.locked_track_id)
        self.locked_track_id = visible_ids[(curr_idx + 1) % len(visible_ids)]
        return self.locked_track_id

    def clear_lock(self) -> None:
        self.locked_track_id = None

    def get_locked_position(self) -> Optional[Point]:
        if self.locked_track_id is None:
            return None
        track = self.tracks.get(self.locked_track_id)
        if track is None:
            self.locked_track_id = None
            return None
        predict_frames = min(track.missing, 3)
        return (
            track.center[0] + track.velocity[0] * predict_frames,
            track.center[1] + track.velocity[1] * predict_frames,
        )

    def auto_lock_if_close(self, point: Point, distance_threshold: float) -> Optional[int]:
        if self.locked_track_id is not None:
            return self.locked_track_id
        nearest = self.lock_nearest(point)
        if nearest is None:
            return None
        dist = _distance(self.tracks[nearest].center, point)
        if dist <= distance_threshold:
            return nearest
        self.locked_track_id = None
        return None


class HotkeyController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.follow_enabled = True
        self.running = True
        self.pending_command: Optional[str] = None

    def on_press(self, key: keyboard.KeyCode) -> Optional[bool]:
        with self._lock:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            if key == keyboard.Key.f8:
                self.follow_enabled = not self.follow_enabled
                print(f"[HOTKEY] 自动跟随: {'开启' if self.follow_enabled else '关闭'}")
            elif key == keyboard.Key.f9:
                self.pending_command = "lock_nearest"
            elif key == keyboard.Key.f10:
                self.pending_command = "cycle"
            elif key == keyboard.Key.f11:
                self.pending_command = "clear"
        return None

    def consume_command(self) -> Optional[str]:
        with self._lock:
            cmd = self.pending_command
            self.pending_command = None
            return cmd

    def is_running(self) -> bool:
        with self._lock:
            return self.running


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _parse_region(region_text: Optional[str], monitor: Dict[str, int]) -> Dict[str, int]:
    if not region_text:
        return dict(monitor)
    raw = [p.strip() for p in region_text.split(",")]
    if len(raw) != 4:
        raise ValueError("region 参数格式应为: left,top,width,height")
    left, top, width, height = [int(v) for v in raw]
    return {"left": left, "top": top, "width": width, "height": height}


def _draw_debug_view(frame: np.ndarray, tracker: MultiObjectTracker) -> np.ndarray:
    vis = frame.copy()
    for track in tracker.tracks.values():
        x1, y1, x2, y2 = track.bbox
        is_locked = track.track_id == tracker.locked_track_id
        color = (0, 255, 0) if is_locked else (255, 255, 0)
        if track.missing > 0:
            color = (128, 128, 128)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id}"
        if is_locked:
            label += " [LOCK]"
        cv2.putText(vis, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def run(args: argparse.Namespace) -> None:
    detector = RedTargetDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
        template_path=args.template,
        template_iou_threshold=args.template_iou_threshold,
    )
    tracker = MultiObjectTracker(
        max_match_distance=args.max_match_distance,
        max_missing_frames=args.max_missing_frames,
    )
    hotkeys = HotkeyController()

    print("脚本已启动，热键：F8 跟随开关 | F9 锁最近 | F10 切换目标 | F11 清锁定 | ESC 退出")

    with mss.mss() as sct:
        if args.monitor < 1 or args.monitor >= len(sct.monitors):
            raise ValueError(f"monitor 编号无效，可用范围: 1~{len(sct.monitors) - 1}")
        capture_region = _parse_region(args.region, sct.monitors[args.monitor])
        fps_interval = 1.0 / max(args.fps, 1.0)

        listener = keyboard.Listener(on_press=hotkeys.on_press)
        listener.start()

        try:
            while hotkeys.is_running():
                t0 = time.time()

                screen = np.array(sct.grab(capture_region))
                frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                detections, _ = detector.detect(frame)
                tracker.update(detections)

                cmd = hotkeys.consume_command()
                mouse_abs = pyautogui.position()
                mouse_rel = (
                    mouse_abs.x - capture_region["left"],
                    mouse_abs.y - capture_region["top"],
                )

                if cmd == "lock_nearest":
                    locked = tracker.lock_nearest(mouse_rel)
                    print(f"[LOCK] 锁定目标: {locked}")
                elif cmd == "cycle":
                    locked = tracker.cycle_lock()
                    print(f"[LOCK] 切换目标: {locked}")
                elif cmd == "clear":
                    tracker.clear_lock()
                    print("[LOCK] 已取消锁定")

                if args.auto_lock_distance > 0:
                    tracker.auto_lock_if_close(mouse_rel, args.auto_lock_distance)

                target_rel = tracker.get_locked_position()
                if target_rel and hotkeys.follow_enabled:
                    target_abs = (
                        capture_region["left"] + target_rel[0],
                        capture_region["top"] + target_rel[1],
                    )
                    mx, my = pyautogui.position()
                    dx = target_abs[0] - mx
                    dy = target_abs[1] - my
                    dist = math.hypot(dx, dy)
                    if dist > args.stop_distance:
                        step = min(1.0, args.smooth_factor)
                        new_x = mx + dx * step
                        new_y = my + dy * step
                        pyautogui.moveTo(int(new_x), int(new_y), duration=0)

                if args.show_window:
                    vis = _draw_debug_view(frame, tracker)
                    cv2.imshow("moving_target_tracker", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                wait = fps_interval - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
        finally:
            listener.stop()
            if args.show_window:
                cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="红底数字目标自动跟随脚本")
    parser.add_argument("--monitor", type=int, default=1, help="mss 监视器编号，通常主屏是 1")
    parser.add_argument("--region", type=str, default=None, help="捕获区域 left,top,width,height")
    parser.add_argument("--fps", type=float, default=30.0, help="检测与跟随帧率")
    parser.add_argument("--template", type=str, default=None, help="红底模板图路径（可选）")
    parser.add_argument(
        "--template-iou-threshold",
        type=float,
        default=0.55,
        help="使用模板时的红色区域 IoU 阈值",
    )
    parser.add_argument("--min-area", type=int, default=500, help="检测框最小面积")
    parser.add_argument("--max-area", type=int, default=20000, help="检测框最大面积")
    parser.add_argument("--min-aspect", type=float, default=0.65, help="宽高比下限")
    parser.add_argument("--max-aspect", type=float, default=1.35, help="宽高比上限")
    parser.add_argument("--max-match-distance", type=float, default=110.0, help="目标关联最大距离")
    parser.add_argument("--max-missing-frames", type=int, default=10, help="最大允许丢失帧数")
    parser.add_argument(
        "--auto-lock-distance",
        type=float,
        default=90.0,
        help="鼠标离目标小于该距离时自动锁定，<=0 表示关闭",
    )
    parser.add_argument("--smooth-factor", type=float, default=0.35, help="鼠标移动平滑系数(0,1]")
    parser.add_argument("--stop-distance", type=float, default=2.0, help="距离小于该值停止微调")
    parser.add_argument("--show-window", action="store_true", help="显示调试窗口")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
