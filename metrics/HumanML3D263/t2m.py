"""HumanML3D-263 T2M metrics: native 263 feats (identity funnel) -> 263-D evaluator.

Pass rebuild_from_joints=True (e.g. via the metrics params in the yaml config)
to instead re-encode the ric joint positions through the shared joints->263
funnel — evaluating 263 in exactly the same mode as the other representations.
"""
from ..tools.t2m_metrics import T2MMetricsBase
from .to_humanml3d import to_humanml3d, to_humanml3d_via_joints


class T2MMetrics(T2MMetricsBase):
    to_humanml3d_single = staticmethod(to_humanml3d)

    def __init__(self, *args, rebuild_from_joints: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # metric params arrive via the instantiate cfg, not constructor kwargs
        self.rebuild_from_joints = bool(self.cfg.get("rebuild_from_joints", rebuild_from_joints))
        if self.rebuild_from_joints:
            # instance attribute shadows the class-level identity funnel
            self.to_humanml3d_single = to_humanml3d_via_joints
