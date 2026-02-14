"""Tests for layer manager."""

import numpy as np

from manga_translator.core.layer_manager import Layer, LayerStack, GimpLayerAdapter


class TestLayer:
    def test_create_layer(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        layer = Layer(name="test", image=img)
        assert layer.name == "test"
        assert layer.visible is True
        assert layer.opacity == 1.0
        assert layer.offset_x == 0
        assert layer.offset_y == 0


class TestLayerStack:
    def test_add_and_get_layer(self):
        stack = LayerStack(width=100, height=100)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        stack.add_layer("bg", img)
        layer = stack.get_layer("bg")
        assert layer is not None
        assert layer.name == "bg"

    def test_get_nonexistent_layer(self):
        stack = LayerStack()
        assert stack.get_layer("missing") is None

    def test_remove_layer(self):
        stack = LayerStack(width=100, height=100)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        stack.add_layer("temp", img)
        assert stack.remove_layer("temp") is True
        assert stack.get_layer("temp") is None

    def test_remove_nonexistent(self):
        stack = LayerStack()
        assert stack.remove_layer("nope") is False

    def test_flatten_empty(self):
        stack = LayerStack(width=50, height=50)
        result = stack.flatten()
        assert result.shape == (50, 50, 3)

    def test_flatten_single_layer(self):
        stack = LayerStack(width=100, height=100)
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        stack.add_layer("bg", img)
        flat = stack.flatten()
        assert np.array_equal(flat, img)

    def test_flatten_skips_invisible(self):
        stack = LayerStack(width=100, height=100)
        bg = np.full((100, 100, 3), 0, dtype=np.uint8)
        fg = np.full((100, 100, 3), 255, dtype=np.uint8)
        stack.add_layer("bg", bg)
        layer = stack.add_layer("fg", fg)
        layer.visible = False
        flat = stack.flatten()
        assert np.array_equal(flat, bg)

    def test_flatten_with_opacity(self):
        stack = LayerStack(width=100, height=100)
        bg = np.full((100, 100, 3), 0, dtype=np.uint8)
        fg = np.full((100, 100, 3), 200, dtype=np.uint8)
        stack.add_layer("bg", bg)
        layer = stack.add_layer("fg", fg)
        layer.opacity = 0.5
        flat = stack.flatten()
        # Should be blended ~100
        assert 90 <= flat[50, 50, 0] <= 110

    def test_flatten_with_alpha(self):
        stack = LayerStack(width=100, height=100)
        bg = np.full((100, 100, 3), 0, dtype=np.uint8)
        # BGRA image with semi-transparent
        fg = np.full((100, 100, 4), 200, dtype=np.uint8)
        fg[:, :, 3] = 128  # ~50% alpha
        stack.add_layer("bg", bg)
        stack.add_layer("fg", fg)
        flat = stack.flatten()
        assert 90 <= flat[50, 50, 0] <= 110

    def test_flatten_with_offset(self):
        stack = LayerStack(width=100, height=100)
        bg = np.full((100, 100, 3), 0, dtype=np.uint8)
        fg = np.full((50, 50, 3), 255, dtype=np.uint8)
        stack.add_layer("bg", bg)
        layer = stack.add_layer("fg", fg)
        layer.offset_x = 25
        layer.offset_y = 25
        flat = stack.flatten()
        assert flat[0, 0, 0] == 0  # top-left = bg
        assert flat[50, 50, 0] == 255  # center = fg


class TestGimpLayerAdapter:
    def test_not_in_gimp(self):
        adapter = GimpLayerAdapter()
        assert adapter.is_gimp_available is False

    def test_create_layer_outside_gimp(self):
        adapter = GimpLayerAdapter()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.create_layer("test", img)
        assert result is None
