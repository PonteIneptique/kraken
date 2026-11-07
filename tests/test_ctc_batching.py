# -*- coding: utf-8 -*-
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from kraken.containers import BaselineLine, BaselineOCRRecord, Segmentation
from kraken.models import ctc
from kraken.models.ctc import CTCRecognitionInferenceMixin


class FakePool:
    """
    Hands out line extractions in a fixed order like imap_unordered does in completion order.
    """
    def __init__(self, order):
        self.order = order

    def imap_unordered(self, func, iterable):
        return (func(i) for i in self.order)


class FakeModel(CTCRecognitionInferenceMixin):
    input = (1, 3, 48, 0)
    use_legacy_polygons = False
    _m_dtype = torch.float32

    def __init__(self, batch_size, order):
        self._inf_config = SimpleNamespace(padding=0, batch_size=batch_size, no_legacy_polygons=False)
        self._fabric = SimpleNamespace(init_tensor=nullcontext)
        self._line_extraction_pool = FakePool(order)

    def _recognize_baseline_lines(self, lines, segmentation):
        for _, _, idx in lines:
            yield BaselineOCRRecord('text', [], [], segmentation.lines[idx]), idx


class TestCTCBatching(unittest.TestCase):

    """
    Tests that the batched recognition returns a record for every line
    """
    def test_empty_line_extracted_last(self):
        n = 7
        lines = [BaselineLine(id=f'l{i}', baseline=[[0, 10 * i], [50, 10 * i]],
                              boundary=[[0, 10 * i - 5], [50, 10 * i - 5], [50, 10 * i + 5], [0, 10 * i + 5]]) for i in range(n)]
        seg = Segmentation(type='baselines', imagename='', text_direction='horizontal-lr', script_detection=False, lines=lines)
        textured = Image.linear_gradient('L').resize((40, 20))
        blank = Image.new('L', (40, 20), 255)

        def fake_extract(im, segmentation, line_idx, legacy=False):
            return (blank if line_idx == n - 1 else textured), line_idx

        for batch_size in (1, 4, 16):
            with self.subTest(batch_size=batch_size), patch.object(ctc, '_extract_line', fake_extract):
                records = list(FakeModel(batch_size, list(range(n)))._recognition_pred(textured, seg))
                self.assertEqual([r.id for r in records], [line.id for line in lines])
