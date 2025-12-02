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
    Text(Text),
    Image(Image),
}

pub struct Shape {
    pub geometry: BezPath,
    pub fill: Option<Fill>,
    pub stroke: Option<Stroke>,
}

pub struct Text {
    pub content: String,
    // Simple font handling: family name
    pub font_family: String,
    pub size: f32,
    pub justify: Justification,
    pub tracking: f32,
    pub line_height: f32,
    pub fill: Option<Fill>,
    pub stroke: Option<Stroke>,
}

pub struct Image {
    // Encoded image data (e.g. PNG, JPEG)
    pub data: Option<Vec<u8>>,
    pub width: u32,
    pub height: u32,
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
    ColorMatrix {
        matrix: [f32; 20],
    },
    DisplacementMap {
        // Defines the displacement map source.
        // Ideally this should be a reference to another layer or an image,
        // but for now we assume it's implicit or handled by the renderer via input chaining
        // if the Lottie structure implies it. However, Lottie displacement maps use a specific layer.
        // Here we'll assume the map is provided as a separate RenderNode or similar.
        // To simplify for this task and match typical Skia filter inputs (which take an input filter),
        // we might need to know *what* to use as the displacement map.
        // BUT, since we are defining the *data* struct, we should hold the data.
        // Let's assume the displacement map layer's content is rendered and passed here?
        // Or maybe just the parameters?
        // The spec says: "Displacement Map: image_filters::displacement_map."
        // We'll stick to parameters.
        scale: f32,
        x_channel: ColorChannel,
        y_channel: ColorChannel,
    },
}

// Enums
#[derive(Clone, Copy, Debug)]
pub enum ColorChannel {
    R,
    G,
    B,
    A,
}

#[derive(Clone, Copy, Debug)]
pub enum Justification {
    Left,
    Right,
    Center,
}

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
