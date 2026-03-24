from __future__ import annotations

import os


# Keep project-local test runs isolated from globally installed pytest plugins.
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
