"""Draw the docs site's problem figure, logo and favicon from a real cbls solve.

Run it from the repository root after the gate's build fence, which puts the
compiled module in build/python:

    PYTHONPATH=build/python .venv/bin/python docs/figures/make_routes.py

It solves INSTANCE through the cbls Python API, the CVRP model of
benchmarks/cvrp/cvrp_model.h, and writes, next to itself, `routes-light.svg`
and `routes-dark.svg` (the landing page's figure, one per Furo colour mode),
`logo-light.svg` and `logo-dark.svg` (the sidebar logo: the routes cut out of
the figure), `favicon.svg`, and `routes-card.png` (the light figure on a
white card, the image a link to the site shows on LinkedIn and elsewhere,
which do not render SVG). The docs build uses the committed files and never
runs this script.
"""

import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import cbls  # noqa: E402

from benchmarks.cvrp.verify_solution import (  # noqa: E402
    VrpInstance,
    distance_matrix,
    read_vrp_file,
)

# A small X instance with a central depot: 114 customers, at least 10 routes.
INSTANCE = ROOT / "benchmarks" / "instances" / "cvrp" / "X" / "X-n115-k10.vrp"
# CVRPLIB's best-known cost (the .sol beside it), for the printed gap only.
BEST_KNOWN = 12747
# The fleet: X's minimum `k` plus 3, as default_fleet() gives for k = 10.
FLEET = 13
# One thread and an iteration cap, not a time limit, so that the routes are the
# same on every run of one build and platform. Seconds of CPU with LNS.
ITERATIONS = 500_000
SEED = 1
LNS_FRACTION = 0.2
NEIGHBOURS = 20

FIGURE_WIDTH = 640.0
MARGIN = 16.0
# The link-preview card: LinkedIn's 1.91:1 image, drawn SUPERSAMPLE times
# larger and scaled down, so that its edges are smooth.
CARD_SIZE = (1200, 627)
CARD_MARGIN = 40.0
CARD_BACKGROUND = "#ffffff"
SUPERSAMPLE = 4


@dataclass(frozen=True)
class Palette:
    """Colours of one Furo mode; `accent` is conf.py's color-brand-primary."""

    accent: str
    depot: str


LIGHT = Palette(accent="#b45309", depot="#111827")
DARK = Palette(accent="#f59e0b", depot="#f3f4f6")
# A browser tab may be light or dark: an amber between the two modes'.
FAVICON_ACCENT = "#d97706"

Point = tuple[float, float]


@dataclass(frozen=True)
class Frame:
    """Maps instance coordinates onto an SVG canvas, y pointing up."""

    x0: float
    y1: float
    scale: float
    width: float
    height: float
    left: float
    top: float

    @staticmethod
    def fit(points: list[Point], size: float, margin: float, square: bool = False) -> "Frame":
        """The points' bounding box, its longer side `size` with the margins.

        The canvas hugs the box, or with `square` is `size` by `size` with
        the box centred in it.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        scale = (size - 2 * margin) / max(span_x, span_y)
        width = span_x * scale + 2 * margin
        height = span_y * scale + 2 * margin
        left = top = margin
        if square:
            left += (size - width) / 2
            top += (size - height) / 2
            width = height = size
        return Frame(min(xs), max(ys), scale, width, height, left, top)

    def __call__(self, p: Point) -> Point:
        return (
            self.left + (p[0] - self.x0) * self.scale,
            self.top + (self.y1 - p[1]) * self.scale,
        )


def svg(frame: Frame, title: str, body: list[str]) -> str:
    """A standalone SVG document with a transparent background."""
    w, h = f"{frame.width:.0f}", f"{frame.height:.0f}"
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">',
            f"<title>{title}</title>",
            *body,
            "</svg>",
            "",
        ]
    )


def closed_route(route: list[int]) -> list[int]:
    """A route's nodes from the depot (node 0) back to it."""
    return [0, *route, 0]


def route_path(
    frame: Frame, coords: list[Point], route: list[int], color: str, width: float
) -> str:
    points = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (frame(coords[i]) for i in closed_route(route))
    )
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def depot_square(frame: Frame, p: Point, side: float, color: str) -> str:
    x, y = frame(p)
    return (
        f'<rect x="{x - side / 2:.1f}" y="{y - side / 2:.1f}" '
        f'width="{side:.1f}" height="{side:.1f}" fill="{color}"/>'
    )


def circle(frame: Frame, p: Point, r: float, color: str) -> str:
    x, y = frame(p)
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"/>'


# The figure's marks, on its FIGURE_WIDTH canvas; the card scales them.
ROUTE_WIDTH = 2.0
DEPOT_SIDE = 12.0


def demand_radii(demands: list[int]) -> list[float]:
    """Each node's circle radius on the figure's canvas, by its demand."""
    top = max(demands) or 1
    return [2.0 + 4.0 * (d / top) ** 0.5 for d in demands]


def draw_figure(inst: VrpInstance, routes: list[list[int]], pal: Palette) -> str:
    """The routes from the depot, customers sized by demand."""
    frame = Frame.fit(inst.coords, FIGURE_WIDTH, MARGIN)
    radius = demand_radii(inst.demands)
    body = [route_path(frame, inst.coords, r, pal.accent, ROUTE_WIDTH) for r in routes]
    body += [circle(frame, inst.coords[i], radius[i], pal.accent) for r in routes for i in r]
    body.append(depot_square(frame, inst.coords[0], DEPOT_SIDE, pal.depot))
    return svg(frame, f"cbls's routes for the CVRP instance {inst.name}", body)


def draw_card(inst: VrpInstance, routes: list[list[int]], pal: Palette) -> Image.Image:
    """The figure as a CARD_SIZE PNG on CARD_BACKGROUND, centred, for link previews."""
    width, height = (side * SUPERSAMPLE for side in CARD_SIZE)
    # A square as tall as the card, its marks scaled as the figure's are to
    # FIGURE_WIDTH, then moved to the card's middle.
    square = Frame.fit(inst.coords, height, CARD_MARGIN * SUPERSAMPLE, square=True)
    frame = replace(square, left=square.left + (width - height) / 2, width=width)
    unit = (height - 2 * CARD_MARGIN * SUPERSAMPLE) / (FIGURE_WIDTH - 2 * MARGIN)

    image = Image.new("RGB", (width, height), CARD_BACKGROUND)
    draw = ImageDraw.Draw(image)
    accent = ImageColor.getrgb(pal.accent)
    radius = demand_radii(inst.demands)
    for r in routes:
        draw.line(
            [frame(inst.coords[i]) for i in closed_route(r)],
            fill=accent,
            width=round(ROUTE_WIDTH * unit),
            joint="curve",
        )
    for r in routes:
        for i in r:
            x, y = frame(inst.coords[i])
            rad = radius[i] * unit
            draw.ellipse((x - rad, y - rad, x + rad, y + rad), fill=accent)
    x, y = frame(inst.coords[0])
    half = DEPOT_SIDE * unit / 2
    draw.rectangle((x - half, y - half, x + half, y + half), fill=pal.depot)
    return image.resize(CARD_SIZE, Image.Resampling.LANCZOS)


def draw_logo(
    inst: VrpInstance, routes: list[list[int]], colors: tuple[str, str], size: float, icon: bool
) -> str:
    """The routes alone, without the customers: lines from the depot.

    `colors` are the routes' and the depot's. An `icon` is square, with strokes
    heavy enough to read at 16 pixels.
    """
    accent, depot_color = colors
    frame = Frame.fit(inst.coords, size, size / (16 if icon else 20), square=icon)
    line, side = (size / 24, size / 5) if icon else (size / 48, size / 9)
    body = [route_path(frame, inst.coords, r, accent, line) for r in routes]
    body.append(depot_square(frame, inst.coords[0], side, depot_color))
    return svg(frame, "cbls", body)


def solve(inst: VrpInstance) -> tuple[float, list[list[int]]]:
    """cvrp_model.h's model: FLEET Lists partitioning the customers exactly,
    each within capacity, the sum of route lengths minimised. Returns the cost
    and the non-empty routes as node numbers (customer element e is node e + 1).
    """
    dist = np.array(distance_matrix(inst), dtype=np.float64)
    between = np.ascontiguousarray(dist[1:, 1:])
    n = inst.customers
    model = cbls.Model()
    routes = [model.list_var(n, 0, n, cbls.ListInit.Empty, f"route{r}") for r in range(FLEET)]
    model.add_list_partition(routes, cbls.Cover.Exact)
    demand = np.array(inst.demands[1:], dtype=np.float64)
    capacity = model.constant(float(inst.capacity))
    lengths = []
    for route in routes:
        model.add_constraint(model.leq(model.lambda_table_sum(route, demand), capacity))
        lengths.append(
            model.pair_table_sum(
                route,
                between,
                head=np.ascontiguousarray(dist[0, 1:]),
                tail=np.ascontiguousarray(dist[1:, 0]),
            )
        )
    model.minimize(model.sum(lengths))
    model.close()

    config = cbls.SearchConfig()
    config.max_iterations = ITERATIONS
    config.structural_neighbours = cbls.nearest_neighbours(
        n, NEIGHBOURS, lambda a, b: float(between[a, b])
    )
    # The time limit only backs the iteration cap up.
    result = cbls.solve(
        model, time_limit=600.0, seed=SEED, threads=1, lns=LNS_FRACTION, config=config
    )
    if not result.feasible or result.termination != cbls.TerminationReason.IterationLimit:
        raise SystemExit(f"{inst.name}: {result.termination}, feasible {result.feasible}")
    nodes = [[e + 1 for e in result.elements[-(h + 1)]] for h in routes]
    return result.objective, [r for r in nodes if r]


def main() -> None:
    inst = read_vrp_file(INSTANCE)
    start = time.process_time()
    cost, routes = solve(inst)
    seconds = time.process_time() - start
    for mode, pal in (("light", LIGHT), ("dark", DARK)):
        (HERE / f"routes-{mode}.svg").write_text(draw_figure(inst, routes, pal))
        (HERE / f"logo-{mode}.svg").write_text(
            draw_logo(inst, routes, (pal.accent, pal.depot), 128.0, icon=False)
        )
    (HERE / "favicon.svg").write_text(
        draw_logo(inst, routes, (FAVICON_ACCENT, FAVICON_ACCENT), 64.0, icon=True)
    )
    draw_card(inst, routes, LIGHT).save(HERE / "routes-card.png", optimize=True)
    gap = 100.0 * (cost - BEST_KNOWN) / BEST_KNOWN
    print(
        f"{inst.name}: cost {cost:.0f} ({gap:.1f}% above the best known {BEST_KNOWN}), "
        f"{len(routes)} routes, {seconds:.1f} s CPU"
    )


if __name__ == "__main__":
    main()
