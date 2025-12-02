use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LottieJson {
    pub v: Option<String>,
    pub ip: f32,
    pub op: f32,
    pub fr: f32,
    pub w: u32,
    pub h: u32,
    pub layers: Vec<Layer>,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Layer {
    // Common
    #[serde(default)]
    pub ty: u8, // 0..5
    #[serde(default)]
    pub ind: Option<u32>,
    #[serde(default)]
    pub parent: Option<u32>,
    #[serde(default)]
    pub nm: Option<String>,
    #[serde(default)]
    pub ip: f32,
    #[serde(default)]
    pub op: f32,
    #[serde(default = "default_one")]
    pub st: f32,
    #[serde(default)]
    pub ks: Transform,
    #[serde(default)]
    pub ao: Option<u32>,

    #[serde(default, rename = "masksProperties")]
    pub masks_properties: Option<Vec<MaskProperties>>,
    #[serde(default)]
    pub tt: Option<u8>,

    // Type specific (flattened manually as optional fields)
    #[serde(default, rename = "refId")]
    pub ref_id: Option<String>, // PreComp, Image
    #[serde(default)]
    pub w: Option<u32>, // PreComp
    #[serde(default)]
    pub h: Option<u32>, // PreComp
    #[serde(default, rename = "sc")]
    pub color: Option<String>, // Solid color
    #[serde(default)]
    pub sw: Option<u32>, // Solid width
    #[serde(default)]
    pub sh: Option<u32>, // Solid height
    #[serde(default)]
    pub shapes: Option<Vec<Shape>>, // Shape Layer
    #[serde(default)]
    pub t: Option<TextData>, // Text Layer
}

fn default_one() -> f32 { 1.0 }

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MaskProperties {
    #[serde(default)]
    pub inv: bool,
    #[serde(default)]
    pub mode: Option<String>,
    pub pt: Property<BezierPath>,
    pub o: Property<f32>,
    #[serde(default)]
    pub nm: Option<String>,
}

// Shapes

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "ty")]
pub enum Shape {
    #[serde(rename = "gr")]
    Group(GroupShape),
    #[serde(rename = "rc")]
    Rect(RectShape),
    #[serde(rename = "el")]
    Ellipse(EllipseShape),
    #[serde(rename = "fl")]
    Fill(FillShape),
    #[serde(rename = "st")]
    Stroke(StrokeShape),
    #[serde(rename = "gf")]
    GradientFill(GradientFillShape),
    #[serde(rename = "gs")]
    GradientStroke(GradientStrokeShape),
    #[serde(rename = "tr")]
    Transform(TransformShape),
    #[serde(rename = "sh")]
    Path(PathShape),
    #[serde(rename = "tm")]
    Trim(TrimShape),
    #[serde(rename = "sr")]
    Polystar(PolystarShape),
    #[serde(rename = "rp")]
    Repeater(RepeaterShape),
    #[serde(rename = "rd")]
    RoundCorners(RoundCornersShape),
    // Use other for unsupported shapes to prevent failure?
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PolystarShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub p: PositionProperty,
    pub or: Property<f32>,
    #[serde(default)]
    pub os: Property<f32>,
    pub r: Property<f32>,
    pub pt: Property<f32>,
    #[serde(default)]
    pub sy: u8, // 1=star, 2=polygon
    #[serde(default)]
    pub ir: Option<Property<f32>>,
    #[serde(default)]
    pub is: Option<Property<f32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RepeaterShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub c: Property<f32>, // Copies
    pub o: Property<f32>, // Offset
    #[serde(default)]
    pub m: u8, // Composite
    pub tr: RepeaterTransform,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RepeaterTransform {
    #[serde(flatten)]
    pub t: Transform,
    #[serde(default)]
    pub so: Property<f32>, // Start Opacity
    #[serde(default)]
    pub eo: Property<f32>, // End Opacity
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RoundCornersShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub r: Property<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GroupShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub it: Vec<Shape>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RectShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub s: Property<Vec2>,
    pub p: Property<Vec2>,
    pub r: Property<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EllipseShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub s: Property<Vec2>,
    pub p: Property<Vec2>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FillShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub c: Property<Vec4>,
    pub o: Property<f32>,
    #[serde(default)]
    pub r: Option<u8>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StrokeShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub c: Property<Vec4>,
    pub w: Property<f32>,
    pub o: Property<f32>,
    #[serde(default)]
    pub lc: u8,
    #[serde(default)]
    pub lj: u8,
    #[serde(default)]
    pub ml: Option<f32>,
    #[serde(default)]
    pub d: Vec<DashProperty>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DashProperty {
    pub n: Option<String>,
    pub v: Property<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GradientFillShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub o: Property<f32>,
    pub s: Property<Vec2>,
    pub e: Property<Vec2>,
    pub t: u8,
    pub g: GradientColors,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GradientStrokeShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub o: Property<f32>,
    pub w: Property<f32>,
    pub s: Property<Vec2>,
    pub e: Property<Vec2>,
    pub t: u8,
    pub g: GradientColors,
    #[serde(default)]
    pub lc: u8,
    #[serde(default)]
    pub lj: u8,
    #[serde(default)]
    pub ml: Option<f32>,
    #[serde(default)]
    pub d: Vec<DashProperty>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GradientColors {
    pub p: u32,
    pub k: Property<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PathShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub ks: Property<BezierPath>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrimShape {
    #[serde(default)]
    pub nm: Option<String>,
    pub s: Property<f32>,
    pub e: Property<f32>,
    pub o: Property<f32>,
    #[serde(default)]
    pub m: u8,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransformShape {
    #[serde(flatten)]
    pub t: Transform,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Transform {
    #[serde(default)]
    pub a: Property<Vec2>,
    #[serde(default)]
    pub p: PositionProperty,
    #[serde(default)]
    pub s: Property<Vec2>,
    #[serde(default)]
    pub r: Property<f32>,
    #[serde(default)]
    pub o: Property<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum PositionProperty {
    Unified(Property<Vec2>),
    Split {
        x: Property<f32>,
        y: Property<f32>,
    },
}

impl Default for PositionProperty {
    fn default() -> Self {
        PositionProperty::Unified(Property::default())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Property<T> {
    #[serde(default)]
    pub a: u8,
    #[serde(default)]
    pub k: Value<T>,
    #[serde(default)]
    pub ix: Option<u32>,
}

impl<T> Default for Property<T> {
    fn default() -> Self {
        Property {
            a: 0,
            k: Value::Default,
            ix: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum Value<T> {
    Default,
    Static(T),
    Animated(Vec<Keyframe<T>>),
}

impl<T> Default for Value<T> {
    fn default() -> Self {
        Value::Default
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Keyframe<T> {
    pub t: f32,
    pub s: Option<T>,
    pub e: Option<T>,
    pub i: Option<Vec2>,
    pub o: Option<Vec2>,
    pub to: Option<Vec<f32>>,
    pub ti: Option<Vec<f32>>,
    pub h: Option<u8>,
}

pub type Vec2 = [f32; 2];
pub type Vec4 = [f32; 4];

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BezierPath {
    #[serde(default)]
    pub c: bool,
    #[serde(default)]
    pub i: Vec<Vec2>,
    #[serde(default)]
    pub o: Vec<Vec2>,
    #[serde(default)]
    pub v: Vec<Vec2>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Asset {
    pub id: String,
    #[serde(default)]
    pub nm: Option<String>,
    #[serde(default)]
    pub layers: Option<Vec<Layer>>,
    #[serde(default)]
    pub w: Option<u32>,
    #[serde(default)]
    pub h: Option<u32>,
    #[serde(default)]
    pub u: Option<String>,
    #[serde(default)]
    pub p: Option<String>,
    #[serde(default)]
    pub e: Option<u8>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TextData {
    pub d: Property<TextDocument>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TextDocument {
    #[serde(default)]
    pub t: String,
    #[serde(default)]
    pub f: String,
    #[serde(default)]
    pub s: f32,
    #[serde(default)]
    pub j: u8,
    #[serde(default)]
    pub tr: f32,
    #[serde(default)]
    pub lh: f32,
    #[serde(default)]
    pub ls: Option<f32>,
    #[serde(default)]
    pub fc: Vec4,
    #[serde(default)]
    pub sc: Option<Vec4>,
    #[serde(default)]
    pub sw: Option<f32>,
    #[serde(default)]
    pub of: Option<bool>,
}
