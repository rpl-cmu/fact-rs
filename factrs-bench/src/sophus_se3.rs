use sophus_lie::{Isometry3, Isometry3F64};
use sophus_opt::{nlls::EvaluatedCostTerm, prelude::*, robust_kernel, variables::VarKind};

/// Pose graph term
#[derive(Debug, Clone)]
pub struct PoseGraph3CostTerm {
    /// 2d relative pose constraint
    pub pose_m_from_pose_n: Isometry3F64,
    /// ids of the two poses
    pub entity_indices: [usize; 2],
}

impl PoseGraph3CostTerm {
    /// Compute the residual of the pose graph term
    ///
    /// `g(ʷTₘ, ʷTₙ) = log[ (ʷTₘ)⁻¹ ∙ ʷTₙ ∙ (ᵐTₙ)⁻¹ ]`
    ///
    /// Note that `ʷTₘ:= world_from_pose_m`, `ʷTₙ:= world_from_pose_n` and
    /// `ᵐTₙ:= pose_m_from_pose_n` are of type `Isometry2F64`.
    pub fn residual<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        world_from_pose_m: Isometry3<S, 1, DM, DN>,
        world_from_pose_n: Isometry3<S, 1, DM, DN>,
        pose_m_from_pose_n: Isometry3<S, 1, DM, DN>,
    ) -> S::Vector<6> {
        (world_from_pose_m.inverse() * world_from_pose_n * pose_m_from_pose_n.inverse()).log()
    }
}

impl HasResidualFn<12, 2, (), (Isometry3F64, Isometry3F64)> for PoseGraph3CostTerm {
    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        world_from_pose_x: (Isometry3F64, Isometry3F64),
        var_kinds: [VarKind; 2],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<12, 2> {
        let world_from_pose_m = world_from_pose_x.0;
        let world_from_pose_n = world_from_pose_x.1;

        let residual = Self::residual(
            world_from_pose_m,
            world_from_pose_n,
            self.pose_m_from_pose_n,
        );

        (
            || {
                -Isometry3::dx_log_a_exp_x_b_at_0(
                    world_from_pose_m.inverse(),
                    world_from_pose_n * self.pose_m_from_pose_n.inverse(),
                )
            },
            || {
                Isometry3::dx_log_a_exp_x_b_at_0(
                    world_from_pose_m.inverse(),
                    world_from_pose_n * self.pose_m_from_pose_n.inverse(),
                )
            },
        )
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
