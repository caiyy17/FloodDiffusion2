"""MEI-138 T2M metrics: feats -> rotation-only joints -> 263 -> 263-D evaluator."""
from ..tools.t2m_metrics import T2MMetricsBase
from .to_humanml3d import to_humanml3d


class T2MMetrics(T2MMetricsBase):
    to_humanml3d_single = staticmethod(to_humanml3d)
