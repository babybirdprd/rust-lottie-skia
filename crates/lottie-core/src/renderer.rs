use glam::{Mat3, Vec2, Vec4};
use kurbo::BezPath;

pub struct RenderTree {
    pub width: f32,
    pub height: f32,
    pub root: RenderNode,
}

impl RenderTree {
    /// Returns a mock render tree for testing purposes.
    pub fn mock_sample() -> Self {
        // Create a red rectangle
        let mut rect_path = BezPath::new();
        rect_path.move_to((100.0, 100.0));
        rect_path.line_to((300.0, 100.0));
        rect_path.line_to((300.0, 300.0));
        rect_path.line_to((100.0, 300.0));
        rect_path.close_path();

        let rect_shape = Shape {
            geometry: rect_path,
            fill: Some(Fill {
                paint: Paint::Solid(Vec4::new(1.0, 0.0, 0.0, 1.0)), // Red
                opacity: 1.0,
                rule: FillRule::NonZero,
            }),
            stroke: Some(Stroke {
                paint: Paint::Solid(Vec4::new(0.0, 0.0, 0.0, 1.0)), // Black
                width: 5.0,
                opacity: 1.0,
                cap: LineCap::Round,
                join: LineJoin::Round,
                miter_limit: None,
                dash: None,
            }),
        };

        let root = RenderNode {
            transform: Mat3::IDENTITY,
            alpha: 1.0,
            blend_mode: BlendMode::Normal,
            content: NodeContent::Shape(rect_shape),
            masks: vec![],
            matte: None,
            effects: vec![],
        };

        RenderTree {
            width: 500.0,
            height: 500.0,
            root,
        }
    }
}

pub struct RenderNode {
    pub transform: Mat3,
    pub alpha: f32,
    pub blend_mode: BlendMode,
    pub content: NodeContent,
    pub masks: Vec<Mask>,
    pub matte: Option<Box<Matte>>,
    pub effects: Vec<Effect>,
}

pub enum NodeContent {
    Group(Vec<RenderNode>),
    Shape(Shape),
}

pub struct Shape {
    pub geometry: BezPath,
    pub fill: Option<Fill>,
    pub stroke: Option<Stroke>,
}

pub struct Fill {
    pub paint: Paint,
    pub opacity: f32,
    pub rule: FillRule,
}

pub struct Stroke {
    pub paint: Paint,
    pub width: f32,
    pub opacity: f32,
    pub cap: LineCap,
    pub join: LineJoin,
    pub miter_limit: Option<f32>,
    pub dash: Option<DashPattern>,
}

pub enum Paint {
    Solid(Vec4), // R, G, B, A
    Gradient(Gradient),
}

pub struct Gradient {
    pub kind: GradientKind,
    pub stops: Vec<GradientStop>,
    // Coordinates are handled by the gradient shader construction.
    // Lottie gradients usually have start/end points.
    pub start: Vec2,
    pub end: Vec2,
}

pub enum GradientKind {
    Linear,
    Radial,
}

pub struct GradientStop {
    pub offset: f32,
    pub color: Vec4,
}

pub struct DashPattern {
    pub array: Vec<f32>,
    pub offset: f32,
}

pub struct Mask {
    pub mode: MaskMode,
    pub geometry: BezPath,
    pub opacity: f32,
}

pub struct Matte {
    pub mode: MatteMode,
    pub node: RenderNode,
}

pub enum Effect {
    GaussianBlur {
        sigma: f32,
    },
    DropShadow {
        color: Vec4,
        offset: Vec2,
        blur: f32,
    },
}

// Enums
#[derive(Clone, Copy, Debug)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
    // Add others as needed
}

#[derive(Clone, Copy, Debug)]
pub enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy, Debug)]
pub enum LineCap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug)]
pub enum LineJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Copy, Debug)]
pub enum MaskMode {
    Add,
    Subtract,
    Intersect,
    Lighten,
    Darken,
    Difference,
}

#[derive(Clone, Copy, Debug)]
pub enum MatteMode {
    Alpha,
    AlphaInverted,
    Luma,
    LumaInverted,
}
