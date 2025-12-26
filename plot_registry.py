from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PlotArtifact:
    plot_id: str
    kind: str
    figure: Any  # plotly.graph_objects.Figure
    summary: Dict[str, Any]


_lock = threading.Lock()
_plots: Dict[str, PlotArtifact] = {}


def store_plot(kind: str, figure: Any, summary: Dict[str, Any]) -> str:
    plot_id = uuid.uuid4().hex
    artifact = PlotArtifact(plot_id=plot_id, kind=kind, figure=figure, summary=summary)
    with _lock:
        _plots[plot_id] = artifact
    return plot_id


def get_plot(plot_id: str) -> Optional[PlotArtifact]:
    with _lock:
        return _plots.get(plot_id)


def clear_plots() -> None:
    with _lock:
        _plots.clear()


