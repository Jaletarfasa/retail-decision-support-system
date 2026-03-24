from __future__ import annotations

from app.explainers import get_explainer_path, list_explainers, load_explainer_markup


def test_explainer_assets_exist_and_load() -> None:
    explainers = list_explainers()

    assert len(explainers) == 5
    for explainer in explainers:
        path = get_explainer_path(explainer)
        assert path.exists()
        markup = load_explainer_markup(explainer)
        assert "<svg" in markup
