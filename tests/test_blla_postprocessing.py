# -*- coding: utf-8 -*-
import unittest

from unittest.mock import patch

import numpy as np
import shapely.geometry as geom
import torch
from shapely.ops import nearest_points, unary_union
from skimage import filters
from skimage.measure import label, regionprops

from kraken.lib import segmentation
from kraken.lib.vgsl import spred


def baseline_map(h=900, w=300, seed=0):
    """
    Wavy ridges plus noise, shaped like the baseline channel of a segmentation heatmap.
    """
    rng = np.random.default_rng(seed)
    im = np.zeros((h, w), dtype=np.float32)
    xs = np.arange(w)
    for y in range(40, h - 40, 36):
        ys = (y + 5 * np.sin(xs / 40)).astype(int)
        for dy in range(-2, 3):
            im[ys + dy, xs] = 1.0
    return im + rng.random((h, w), dtype=np.float32) * 0.05


def moore_neighborhood(current, backtrack):
    operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])
    neighbors = (current + operations).astype(int)
    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0


def reference_boundary_tracing(region):
    """
    The numpy based tracing kraken used before it was rewritten on plain ints.
    """
    coords = region.coords
    mins = np.amin(coords, axis=0)
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] - mins[0] + 3, maxs[1] - mins[1] + 3))
    y = coords[:, 0] - mins[0] + 1
    x = coords[:, 1] - mins[1] + 1
    binary[tuple([y, x])] = 1
    idx_start = 0
    while True:
        start = [y[idx_start], x[idx_start]]
        if np.sum(binary[start[0] - 1:start[0] + 2, start[1] - 1:start[1] + 2]) > 1:
            break
        idx_start += 1
    if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]
    current, backtrack, boundary = start, backtrack_start, []
    while True:
        neighbors_current = moore_neighborhood(current, backtrack)
        idx = np.argmax(binary[tuple([neighbors_current[:, 0], neighbors_current[:, 1]])])
        boundary.append(current)
        backtrack = neighbors_current[idx - 1]
        current = neighbors_current[idx]
        if np.all(current == start) and np.all(backtrack == backtrack_start):
            break
    return np.array(boundary) + [mins[0] - 1, mins[1] - 1]


def reference_calc_roi(line, bounds, baselines, suppl_obj, p_dir):
    """
    The one-shapely-call-at-a-time implementation kraken used before _calc_roi was vectorized.
    """
    ls = geom.LineString(line)
    ip_line = [line[0]]
    dist = 10
    while dist < ls.length:
        ip_line.append(np.array(ls.interpolate(dist).coords[0]))
        dist += 10
    ip_line.append(line[-1])
    ip_line = np.array(ip_line)
    upper_bounds_intersects = []
    bottom_bounds_intersects = []
    for point in ip_line:
        upper_bounds_intersects.append(segmentation._ray_intersect_boundaries(point, (p_dir * (-1, 1))[::-1], bounds + 1).astype('int'))
        bottom_bounds_intersects.append(segmentation._ray_intersect_boundaries(point, (p_dir * (1, -1))[::-1], bounds + 1).astype('int'))
    upper_polygon = geom.Polygon(ip_line.tolist() + upper_bounds_intersects)
    bottom_polygon = geom.Polygon(ip_line.tolist() + bottom_bounds_intersects)
    side_a = [geom.LineString(upper_bounds_intersects)]
    side_b = [geom.LineString(bottom_bounds_intersects)]
    for adj_line in baselines + suppl_obj:
        adj_line = geom.LineString(adj_line)
        if upper_polygon.intersects(adj_line):
            side_a.append(adj_line)
        elif bottom_polygon.intersects(adj_line):
            side_b.append(adj_line)
    side_a = unary_union(side_a).buffer(1).boundary
    side_b = unary_union(side_b).buffer(1).boundary

    def closest(pt, intersects):
        spt = geom.Point(pt)
        if intersects.is_empty:
            raise Exception('No intersection with boundaries.')
        if intersects.geom_type == 'MultiPoint':
            return min([p for p in intersects.geoms], key=lambda x: spt.distance(x))
        elif intersects.geom_type == 'Point':
            return intersects
        elif intersects.geom_type == 'GeometryCollection' and len(intersects.geoms) > 0:
            t = min([p for p in intersects.geoms], key=lambda x: spt.distance(x))
            return nearest_points(spt, t)[1]
        raise Exception('No intersection with boundaries.')

    env_up = []
    env_bottom = []
    for point, upper_bounds_intersect, bottom_bounds_intersect in zip(ip_line, upper_bounds_intersects, bottom_bounds_intersects):
        env_up.append(closest(point, geom.LineString([point, upper_bounds_intersect]).intersection(side_a)).coords[0])
        env_bottom.append(closest(point, geom.LineString([point, bottom_bounds_intersect]).intersection(side_b)).coords[0])
    return np.array(env_up, dtype='uint'), np.array(env_bottom, dtype='uint')


def wavy_baselines(n, spacing, width=900):
    xs = np.linspace(60, width, 14)
    return [np.stack([xs, 120 + k * spacing + 12 * np.sin(xs / 90 + k)], axis=1) for k in range(n)]


class TestBllaPostprocessing(unittest.TestCase):

    """
    Tests that the faster post-processing of the baseline segmenter gives the same results as the straightforward implementations
    """
    def check_sato(self, sato, device):
        im = baseline_map()
        ref = filters.sato(im, black_ridges=False, mode='constant')
        res = sato(im, device)
        self.assertEqual(res.dtype, ref.dtype)
        np.testing.assert_allclose(res, ref, atol=1e-5)
        np.testing.assert_array_equal(res > 0.17, ref > 0.17)

    def test_sato_cpu(self):
        self.check_sato(segmentation._sato, torch.device('cpu'))

    def test_sato_torch_cpu(self):
        self.check_sato(segmentation._sato_torch, torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_sato_cuda(self):
        self.check_sato(segmentation._sato, torch.device('cuda'))

    def test_boundary_tracing(self):
        blobs = label(baseline_map(300, 200) > 0.5)
        regions = [r for r in regionprops(blobs) if r.area >= 6]
        self.assertGreater(len(regions), 3)
        for region in regions:
            np.testing.assert_array_equal(segmentation.boundary_tracing(region), reference_boundary_tracing(region))

    def test_calc_roi(self):
        for spacing in (70, 520):  # the second one has neighbours farther away than the short ray used first
            lines = wavy_baselines(4, spacing)
            bounds = np.array([999., 120 + 4 * spacing + 200.])
            extra = [np.array([[30., 10.], [30., bounds[1] - 10]])]
            for idx, line in enumerate(lines):
                others = [x.tolist() for k, x in enumerate(lines) if k != idx]
                lengths = np.linalg.norm(np.diff(line.T), axis=0)
                p_dir = np.mean(np.diff(line.T) * lengths / lengths.sum(), axis=1)
                p_dir = p_dir / np.sqrt(np.sum(p_dir ** 2))
                for suppl in ([], [x.tolist() for x in extra]):
                    ref = reference_calc_roi(line, bounds, others, suppl, p_dir)
                    res = segmentation._calc_roi(line, bounds, others, suppl, p_dir)
                    np.testing.assert_array_equal(res[0], ref[0])
                    np.testing.assert_array_equal(res[1], ref[1])

    def test_vec_lines_workers(self):
        h, w = 900, 300
        heatmap = np.zeros((3, h, w), dtype=np.float32)
        heatmap[2] = baseline_map(h, w)
        kwargs = dict(cls_map={'aux': {'_start_separator': 0, '_end_separator': 1}, 'baselines': {'default': 2}},
                      scale=1.0,
                      regions=[],
                      scal_im=np.zeros((h, w)))

        def fake_polygonal_environment(baselines, **kwargs):
            x, y = np.array(baselines[0]).T
            if y.mean() > 400:  # lines the polygonizer fails on
                return [None]
            return [[[x.min(), y.min() - 10], [x.max(), y.min() - 10], [x.max(), y.max() + 10], [x.min(), y.max() + 10]]]

        def reset_pool():
            if spred._pool is not None:
                spred._pool.terminate()
            spred._pool, spred._pool_size = None, 0

        self.addCleanup(reset_pool)
        with patch.object(spred, 'calculate_polygonal_environment', fake_polygonal_environment):
            serial = spred.vec_lines(heatmap, num_workers=1, **kwargs)
            parallel = spred.vec_lines(heatmap, num_workers=3, **kwargs)
        self.assertGreater(len(serial), 3)
        self.assertEqual(repr(serial), repr(parallel))
