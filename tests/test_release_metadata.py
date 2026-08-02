import importlib.metadata

import refgauge


def test_runtime_and_distribution_versions_match():
    assert refgauge.__version__ == importlib.metadata.version("refgauge")
