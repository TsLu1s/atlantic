"""Tests for atlantic.feature_selection.h2o_selector that don't require
booting an H2O cluster — they exercise the shutdown contract via mocks.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from atlantic.feature_selection.h2o_selector import H2OFeatureSelector


class TestH2OShutdownContract:
    """Regression tests for the cluster-cleanup bug.

    History of this code path:
      * v1 called ``h2o.shutdown(prompt=False)`` — kills the entire cluster.
      * v2 called ``h2o.remove_all()`` — drops EVERY key in the shared
        cluster, including frames/models owned by other user code.
      * current: removes ONLY the keys this selector created during fit
        (tracked in ``self._h2o_keys``) via ``h2o.remove(key)``, leaving all
        other H2O state intact.
    """

    def test_shutdown_removes_only_own_keys(self):
        selector = H2OFeatureSelector(target="t", relevance=0.99, max_models=1)
        selector._h2o_keys = ["my_frame", "model_a", "model_b"]

        stub = _StubH2O()
        with patch.dict("sys.modules", {"h2o": stub}):
            selector._shutdown_h2o()

        assert stub.shutdown_called is False, (
            "h2o.shutdown must NOT be called — it would tear down the cluster"
        )
        assert stub.remove_all_called is False, (
            "h2o.remove_all must NOT be called — it drops other users' keys too"
        )
        assert stub.removed == ["my_frame", "model_a", "model_b"], (
            "only this selector's own tracked keys should be removed"
        )
        assert selector._h2o_keys == [], "tracked keys should be cleared after cleanup"

    def test_shutdown_with_no_keys_is_noop(self):
        selector = H2OFeatureSelector(target="t", relevance=0.99, max_models=1)
        stub = _StubH2O()
        with patch.dict("sys.modules", {"h2o": stub}):
            selector._shutdown_h2o()
        assert stub.removed == []
        assert stub.remove_all_called is False

    def test_shutdown_swallows_errors(self):
        """Cleanup must remain best-effort: a per-key removal failure must
        not propagate (frames may already be gone, cluster down, etc.)."""
        selector = H2OFeatureSelector(target="t", relevance=0.99, max_models=1)
        selector._h2o_keys = ["k1", "k2"]
        with patch.dict("sys.modules", {"h2o": _RaisingStubH2O()}):
            # Should not raise.
            selector._shutdown_h2o()
        assert selector._h2o_keys == []


class _StubH2O:
    """Minimal stub of the h2o module — records which calls happened."""

    def __init__(self):
        self.shutdown_called = False
        self.remove_all_called = False
        self.removed = []

    def shutdown(self, prompt=False):  # pragma: no cover - must not be called
        self.shutdown_called = True

    def remove_all(self):  # pragma: no cover - must not be called
        self.remove_all_called = True

    def remove(self, key):
        self.removed.append(key)


class _RaisingStubH2O:
    def shutdown(self, prompt=False):  # pragma: no cover
        raise RuntimeError("must not be called")

    def remove_all(self):  # pragma: no cover
        raise RuntimeError("must not be called")

    def remove(self, key):
        raise RuntimeError("simulated cleanup failure")
