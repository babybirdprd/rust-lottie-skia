// lottie-core: Interpolation logic and Scene Graph
pub mod renderer;

pub use renderer::{
    BlendMode, DashPattern, Effect, Fill, FillRule, Gradient, GradientKind, GradientStop, LineCap,
    LineJoin, Mask, MaskMode, Matte, MatteMode, NodeContent, Paint, RenderNode, RenderTree, Shape,
    Stroke,
};

pub struct LottiePlayer;

impl LottiePlayer {
    pub fn new() -> Self {
        Self
    }
}
