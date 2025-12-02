// lottie-core: Interpolation logic and Scene Graph
pub mod renderer;

pub use renderer::{
    BlendMode, ColorChannel, DashPattern, Effect, Fill, FillRule, Gradient, GradientKind,
    GradientStop, Image, Justification, LineCap, LineJoin, Mask, MaskMode, Matte, MatteMode,
    NodeContent, Paint, RenderNode, RenderTree, Shape, Stroke, Text,
};

pub struct LottiePlayer;

impl LottiePlayer {
    pub fn new() -> Self {
        Self
    }
}
