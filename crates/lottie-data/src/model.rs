use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct LottieJson {
    pub v: String, // Version
    pub ip: f32,   // In Point
    pub op: f32,   // Out Point
    pub fr: f32,   // Frame Rate
    pub w: u32,    // Width
    pub h: u32,    // Height
    pub layers: Vec<serde_json::Value>,
}
