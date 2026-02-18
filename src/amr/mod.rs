pub mod hierarchy;
pub mod interp;
pub mod patch;
pub mod rect;
pub mod stepper;

pub mod clustering;
pub mod indicator;
pub mod regrid;

pub use hierarchy::AmrHierarchy2D;
pub use patch::Patch2D;
pub use rect::Rect2i;
pub use stepper::{AmrStepperRK4, PatchRK4Scratch};

pub use clustering::{
    ClusterPolicy, ClusterStats, Connectivity, compute_patch_rects_clustered_from_indicator,
};
pub use indicator::{IndicatorStats, compute_patch_bbox_from_indicator, indicator_grad2_forward};
pub use regrid::{RegridPolicy, maybe_regrid_multi_patch, maybe_regrid_single_patch};
