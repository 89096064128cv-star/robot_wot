"""Offline 2D smoothing demo using synthetic points only."""

import argparse
import math
from pathlib import Path
from typing import Callable, NamedTuple


class Point(NamedTuple):
    x: float
    y: float


class Sample(NamedTuple):
    time: float
    target: Point
    follower: Point


def target_at(time: float) -> Point:
    """Return a synthetic target following a smooth elliptical path."""
    return Point(320.0 + 130.0 * math.cos(0.65 * time),
                 220.0 + 90.0 * math.sin(0.65 * time))


def simulate(
    target: Callable[[float], Point] = target_at,
    initial: Point = Point(80.0, 80.0),
    max_speed: float = 260.0,
    smoothness: float = 7.5,
    duration: float = 12.0,
    time_step: float = 1.0 / 60.0,
) -> list[Sample]:
    """Simulate a speed-limited point following a supplied synthetic target."""
    if max_speed <= 0 or duration <= 0 or time_step <= 0 or smoothness < 0:
        raise ValueError("speed, duration and time_step must be positive; smoothness cannot be negative")

    follower = initial
    samples = [Sample(0.0, target(0.0), follower)]
    elapsed = 0.0
    while elapsed < duration:
        step = min(time_step, duration - elapsed)
        next_time = elapsed + step
        destination = target(next_time)
        dx = destination.x - follower.x
        dy = destination.y - follower.y
        distance = math.hypot(dx, dy)

        if distance:
            smoothed_distance = distance * (1.0 - math.exp(-smoothness * step))
            travel = min(smoothed_distance, max_speed * step, distance)
            follower = Point(
                follower.x + dx * travel / distance,
                follower.y + dy * travel / distance,
            )

        elapsed = next_time
        samples.append(Sample(elapsed, destination, follower))
    return samples


def write_svg(samples: list[Sample], output: Path) -> None:
    """Write a self-contained SVG showing the synthetic target and follower paths."""
    width, height, padding = 640, 440, 24
    points = [point for sample in samples for point in (sample.target, sample.follower)]
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)

    def project(point: Point) -> tuple[float, float]:
        x = padding + (point.x - min_x) * scale
        y = height - padding - (point.y - min_y) * scale
        return x, y

    def path_for(path_points: list[Point]) -> str:
        return " ".join(
            ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
            for index, (x, y) in enumerate(map(project, path_points))
        )

    target_path = path_for([sample.target for sample in samples])
    follower_path = path_for([sample.follower for sample in samples])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#101820"/>
<text x="{padding}" y="20" fill="#ffffff" font-family="sans-serif" font-size="14">Офлайн-симуляция: синтетические траектории</text>
<path d="{target_path}" fill="none" stroke="#ffb000" stroke-width="2.5"/>
<path d="{follower_path}" fill="none" stroke="#00c8ff" stroke-width="2.5"/>
<circle cx="{project(samples[-1].target)[0]:.1f}" cy="{project(samples[-1].target)[1]:.1f}" r="5" fill="#ffb000"/>
<circle cx="{project(samples[-1].follower)[0]:.1f}" cy="{project(samples[-1].follower)[1]:.1f}" r="5" fill="#00c8ff"/>
<text x="{padding}" y="{height - 8}" fill="#ffb000" font-family="sans-serif" font-size="13">Синтетическая цель</text>
<text x="{padding + 180}" y="{height - 8}" fill="#00c8ff" font-family="sans-serif" font-size="13">Сглаживаемая точка</text>
</svg>
"""
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Симуляция плавного следования за синтетической 2D-точкой.")
    parser.add_argument("--speed", type=float, default=260.0, help="Ограничение скорости точки (единиц/с).")
    parser.add_argument("--smoothness", type=float, default=7.5, help="Скорость сглаживания (1/с).")
    parser.add_argument("--duration", type=float, default=12.0, help="Длительность симуляции в секундах.")
    parser.add_argument("--output", type=Path, default=Path("trajectory.svg"), help="Путь для SVG-траектории.")
    args = parser.parse_args()

    samples = simulate(max_speed=args.speed, smoothness=args.smoothness, duration=args.duration)
    write_svg(samples, args.output)
    last = samples[-1]
    error = math.hypot(last.target.x - last.follower.x, last.target.y - last.follower.y)
    print(f"Готово: {args.output} (конечное расстояние до синтетической цели: {error:.1f})")


if __name__ == "__main__":
    main()
