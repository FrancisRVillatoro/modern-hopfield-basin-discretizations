"""Interval enclosure of the critical points of the planar Hopfield energy.

Run: python certify_planar_critical_points.py
Uses mpmath's directed-rounding interval context. All inputs to the
verification are exact rational data; decimal strings denote rationals.
"""
import json
from pathlib import Path
from collections import deque

import mpmath as mp

mp.iv.dps = 35
iv = mp.iv
PATTERNS = tuple(tuple(iv.mpf(x) for x in row)
                 for row in (("2", "0"), ("-1", "1.5"), ("-0.5", "-1.8")))
BETA = iv.mpf("1.5")
CENTERS = (
    ("-0.497076272255", "-1.797168306609"),
    ("-0.998707919574", "1.498488348053"),
    ("1.998233731262", "-0.000817575796"),
    ("-0.679272916886", "-0.088498064225"),
    ("0.651499906767", "-0.960557311514"),
    ("0.349281048573", "0.780616371879"),
    ("-0.111097064525", "0.029662574439"),
)
RADIUS = iv.mpf("0.0000001")


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def subtract(x, y):
    return tuple(a - b for a, b in zip(x, y))


def interior(x, y):
    return all(u.a > v.a and u.b < v.b for u, v in zip(x, y))


def disjoint(x, y):
    return any(u.b < v.a or u.a > v.b for u, v in zip(x, y))


def contains_zero(x):
    return not (x.a > 0 or x.b < 0)


def g_and_h(x):
    p = [
        1 / sum(iv.exp(BETA * dot(subtract(other, pattern), x)) for other in PATTERNS)
        for pattern in PATTERNS
    ]
    g = tuple(x[i] - sum(weight * pattern[i] for weight, pattern in zip(p, PATTERNS))
              for i in range(2))
    h = [[iv.mpf(int(i == j)) for j in range(2)] for i in range(2)]
    for a, pattern in enumerate(PATTERNS):
        for b in range(a):
            d = subtract(pattern, PATTERNS[b])
            factor = BETA * p[a] * p[b]
            for i in range(2):
                for j in range(2):
                    h[i][j] -= factor * d[i] * d[j]
    return g, h


def determinant(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def midpoint_inverse(matrix):
    a, b, c, d = (entry.mid for row in matrix for entry in row)
    det = a * d - b * c
    if contains_zero(det):
        return None
    # Any nonsingular point preconditioner is allowed.
    inverse = tuple(tuple((entry / det).mid for entry in row) for row in ((d, -b), (-c, a)))
    return inverse if not contains_zero(determinant(inverse)) else None


def krawczyk(box, h):
    center = tuple(x.mid for x in box)
    g_center, h_center = g_and_h(center)
    inverse = midpoint_inverse(h_center)
    if inverse is None:
        return None, None
    offset = subtract(box, center)
    residual = [
        [int(i == j) - sum(inverse[i][k] * h[k][j] for k in range(2))
         for j in range(2)]
        for i in range(2)
    ]
    image = tuple(center[i] - dot(inverse[i], g_center) + dot(residual[i], offset)
                  for i in range(2))
    norm = max(sum(abs(entry).b for entry in row).b for row in residual)
    return image, norm


def classify(h):
    det = determinant(h)
    if det.b < 0:
        return 1
    if det.a > 0 and h[0][0].a > 0:
        return 0
    if det.a > 0 and h[0][0].b < 0:
        return 2
    raise ValueError("Interval Hessian has no certified index")


def energy(x):
    return dot(x, x) / 2 - iv.log(sum(iv.exp(BETA * dot(pattern, x))
                                     for pattern in PATTERNS)) / BETA


def certify_descending_chord(saddle, minimum):
    """Verify E(s + t(m-s)) < E(s) for 0 < t <= 1."""
    direction = subtract(minimum, saddle)
    near_saddle = iv.mpf([0, "0.03125"])
    nearby = tuple(x + near_saddle * v for x, v in zip(saddle, direction))
    _, h = g_and_h(nearby)
    curvature = sum(direction[i] * h[i][j] * direction[j]
                    for i in range(2) for j in range(2))
    if curvature.b >= 0:
        raise RuntimeError("Initial chord curvature is not strictly negative")

    exit_energy = energy(saddle).a
    todo = [iv.mpf(["0.03125", 1])]
    tested = 0
    while todo:
        parameter = todo.pop()
        tested += 1
        if tested > 100_000:
            raise RuntimeError("Chord subdivision limit exceeded")
        points = tuple(x + parameter * v for x, v in zip(saddle, direction))
        if energy(points).b < exit_energy:
            continue
        midpoint = parameter.mid
        todo.extend((iv.mpf([parameter.a, midpoint]), iv.mpf([midpoint, parameter.b])))
    return tested


def main():
    known = []
    records = []
    root_intervals = []
    for center in CENTERS:
        # RADIUS and decimal inputs are enclosed by directed rounding.
        c = tuple(iv.mpf(x) for x in center)
        box = tuple(iv.mpf([(x - RADIUS).a, (x + RADIUS).b]) for x in c)
        _, h = g_and_h(box)
        image, norm = krawczyk(box, h)
        if image is None or not interior(image, box) or norm >= 1:
            raise RuntimeError("Could not verify unique root around " + repr(center))
        if any(not disjoint(box, old_box) for old_box in known):
            raise RuntimeError("Non-disjoint root boxes")
        known.append(box)
        root_intervals.append(image)
        records.append({
            "center_decimal": list(center),
            "radius_decimal": "0.0000001",
            "morse_index": classify(h),
            "critical_energy_enclosure": str(energy(image)),
        })

    # Every critical point is a strictly positive convex combination of patterns.
    todo = deque([(iv.mpf(["-1", "2"]), iv.mpf(["-1.8", "1.5"]))])
    checked = 0
    while todo:
        box = todo.pop()
        checked += 1
        if checked > 100_000:
            raise RuntimeError("Subdivision limit exceeded")
        if any(interior(box, root_box) for root_box in known):
            continue
        g, h = g_and_h(box)
        if not all(contains_zero(entry) for entry in g):
            continue
        image, _ = krawczyk(box, h)
        if image is not None:
            if disjoint(image, box):
                continue
            # Every root in box must lie in its Krawczyk image.
            if any(interior(image, root_box) for root_box in known):
                continue
        index = int(box[1].delta > box[0].delta)
        left, right = list(box), list(box)
        x = box[index]
        midpoint = x.mid
        left[index] = iv.mpf([x.a, midpoint])
        right[index] = iv.mpf([midpoint, x.b])
        todo.extend((tuple(left), tuple(right)))

    # Confirm the Morse-index pattern and order relevant to the topology.
    assert [item["morse_index"] for item in records] == [0, 0, 0, 1, 1, 1, 2]
    ordered = [energy(root_intervals[i]) for i in (2, 0, 1, 4, 5, 3, 6)]
    assert all(a.b < b.a for a, b in zip(ordered, ordered[1:]))
    chords = [
        {
            "saddle_root_index": saddle,
            "minimum_root_index": minimum,
            "tested_parameter_intervals": certify_descending_chord(
                root_intervals[saddle], root_intervals[minimum],
            ),
        }
        for saddle, minimum in ((4, 0), (4, 2), (5, 1), (5, 2))
    ]
    path = Path(__file__).with_name("planar_critical_intervals.json")
    result = {
        "checked_boxes": checked,
        "decimal_precision": iv.dps,
        "roots": records,
        "saddle_to_minimum_chords": chords,
        "escape_saddle_root_by_minimum_root": {"0": 4, "1": 5, "2": 4},
    }
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
