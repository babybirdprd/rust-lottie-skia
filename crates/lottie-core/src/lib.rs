pub mod animatable;
pub mod modifiers;
pub mod renderer;

use animatable::Animator;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use glam::{Mat3, Vec2, Vec4};
use kurbo::{BezPath, Point, Shape as _};
use lottie_data::model::{self as data, LottieJson};
use modifiers::{
    GeometryModifier, OffsetPathModifier, PuckerBloatModifier, TwistModifier, WiggleModifier,
    ZigZagModifier,
};
pub use renderer::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

#[derive(Clone)]
struct PendingGeometry {
    kind: GeometryKind,
    transform: Mat3,
}

#[derive(Clone)]
enum GeometryKind {
    Path(BezPath),
    Rect { size: Vec2, pos: Vec2, radius: f32 },
    Polystar(PolystarParams),
    Ellipse { size: Vec2, pos: Vec2 },
    Merge(Vec<PendingGeometry>, MergeMode),
}

impl PendingGeometry {
    fn to_shape_geometry(&self, builder: &SceneGraphBuilder) -> ShapeGeometry {
        match &self.kind {
            GeometryKind::Merge(geoms, mode) => {
                let shapes = geoms.iter().map(|g| g.to_shape_geometry(builder)).collect();
                ShapeGeometry::Boolean {
                    mode: *mode,
                    shapes,
                }
            }
            _ => ShapeGeometry::Path(self.to_path(builder)),
        }
    }

    fn to_path(&self, builder: &SceneGraphBuilder) -> BezPath {
        let mut path = match &self.kind {
            GeometryKind::Path(p) => p.clone(),
            GeometryKind::Merge(geoms, _) => {
                let mut p = BezPath::new();
                for g in geoms {
                    p.extend(g.to_path(builder));
                }
                p
            }
            GeometryKind::Rect { size, pos, radius } => {
                let half = *size / 2.0;
                let rect = kurbo::Rect::new(
                    (pos.x - half.x) as f64,
                    (pos.y - half.y) as f64,
                    (pos.x + half.x) as f64,
                    (pos.y + half.y) as f64,
                );
                if *radius > 0.0 {
                    rect.to_rounded_rect(*radius as f64).to_path(0.1)
                } else {
                    rect.to_path(0.1)
                }
            }
            GeometryKind::Ellipse { size, pos } => {
                let half = *size / 2.0;
                let ellipse = kurbo::Ellipse::new(
                    (pos.x as f64, pos.y as f64),
                    (half.x as f64, half.y as f64),
                    0.0,
                );
                ellipse.to_path(0.1)
            }
            GeometryKind::Polystar(params) => builder.generate_polystar_path(params),
        };

        let m = self.transform.to_cols_array();
        let affine = kurbo::Affine::new([
            m[0] as f64,
            m[1] as f64,
            m[3] as f64,
            m[4] as f64,
            m[6] as f64,
            m[7] as f64,
        ]);
        path.apply_affine(affine);
        path
    }
}

#[derive(Clone, Copy)]
struct PolystarParams {
    pos: Vec2,
    outer_radius: f32,
    inner_radius: f32,
    outer_roundness: f32,
    inner_roundness: f32,
    rotation: f32,
    points: f32,
    kind: u8,           // 1=star, 2=polygon
    corner_radius: f32, // From RoundCorners modifier
}

pub enum ImageSource {
    Data(Vec<u8>), // Encoded bytes (PNG/JPG)
}

pub trait TextMeasurer: Send + Sync {
    /// Returns the width of the text string for the given font and size.
    fn measure(&self, text: &str, font_family: &str, size: f32) -> f32;
}

pub struct LottiePlayer {
    pub model: Option<LottieJson>,
    pub current_frame: f32,
    pub width: f32,
    pub height: f32,
    pub duration_frames: f32,
    pub frame_rate: f32,
    pub assets: HashMap<String, ImageSource>,
    pub text_measurer: Option<Box<dyn TextMeasurer>>,
}

impl LottiePlayer {
    pub fn new() -> Self {
        Self {
            model: None,
            current_frame: 0.0,
            width: 500.0,
            height: 500.0,
            duration_frames: 60.0,
            frame_rate: 60.0,
            assets: HashMap::new(),
            text_measurer: None,
        }
    }

    pub fn set_text_measurer(&mut self, measurer: Box<dyn TextMeasurer>) {
        self.text_measurer = Some(measurer);
    }

    pub fn set_asset(&mut self, id: String, data: Vec<u8>) {
        self.assets.insert(id, ImageSource::Data(data));
    }

    pub fn load(&mut self, data: LottieJson) {
        self.width = data.w as f32;
        self.height = data.h as f32;
        self.frame_rate = data.fr;
        self.duration_frames = data.op - data.ip;
        self.current_frame = data.ip; // Start at in-point
        self.model = Some(data);
    }

    pub fn advance(&mut self, dt: f32) {
        if self.model.is_none() {
            return;
        }
        // dt is in seconds
        let frames = dt * self.frame_rate;
        self.current_frame += frames;

        // Loop
        if self.current_frame >= self.model.as_ref().unwrap().op {
            let duration = self.model.as_ref().unwrap().op - self.model.as_ref().unwrap().ip;
            self.current_frame = self.model.as_ref().unwrap().ip
                + (self.current_frame - self.model.as_ref().unwrap().op) % duration;
        }
    }

    pub fn render_tree(&self) -> RenderTree {
        if let Some(model) = &self.model {
            let mut builder = SceneGraphBuilder::new(
                model,
                self.current_frame,
                &self.assets,
                self.text_measurer.as_deref(),
            );
            builder.build()
        } else {
            // Return empty tree
            RenderTree {
                width: self.width,
                height: self.height,
                root: RenderNode {
                    transform: Mat3::IDENTITY,
                    alpha: 1.0,
                    blend_mode: BlendMode::Normal,
                    content: NodeContent::Group(vec![]),
                    masks: vec![], styles: vec![],
                    matte: None,
                    effects: vec![],
                    is_adjustment_layer: false,
                },
            }
        }
    }
}

struct SceneGraphBuilder<'a> {
    model: &'a LottieJson,
    frame: f32,
    assets: HashMap<String, &'a data::Asset>,
    external_assets: &'a HashMap<String, ImageSource>,
    text_measurer: Option<&'a dyn TextMeasurer>,
}

impl<'a> SceneGraphBuilder<'a> {
    fn new(
        model: &'a LottieJson,
        frame: f32,
        external_assets: &'a HashMap<String, ImageSource>,
        text_measurer: Option<&'a dyn TextMeasurer>,
    ) -> Self {
        let mut assets = HashMap::new();
        for asset in &model.assets {
            assets.insert(asset.id.clone(), asset);
        }
        Self {
            model,
            frame,
            assets,
            external_assets,
            text_measurer,
        }
    }

    fn build(&mut self) -> RenderTree {
        let root_node = self.build_composition(&self.model.layers);
        RenderTree {
            width: self.model.w as f32,
            height: self.model.h as f32,
            root: root_node,
        }
    }

    fn build_composition(&mut self, layers: &'a [data::Layer]) -> RenderNode {
        // Iterate in reverse for painter's algorithm (bottom layers first)
        // Lottie: index 0 is top.
        // We want to draw bottom layer first. So iterate layers.len()-1 down to 0.

        let mut nodes = Vec::new();

        // Map ind -> Layer for parenting lookup
        let mut layer_map = HashMap::new();
        for layer in layers {
            if let Some(ind) = layer.ind {
                layer_map.insert(ind, layer);
            }
        }

        let mut consumed_indices = HashSet::new();
        let len = layers.len();

        // Render order: Bottom (last in list) to Top (first in list)
        for i in (0..len).rev() {
            if consumed_indices.contains(&i) {
                continue;
            }

            let layer = &layers[i];

            // Track Mattes
            // If this layer has 'tt', it uses the layer ABOVE it (index i - 1) as the matte.
            // We must process that matte layer and attach it to this layer's node.
            // Then mark the matte layer as consumed so it isn't rendered separately.
            if let Some(tt) = layer.tt {
                if i > 0 {
                    let matte_idx = i - 1;
                    // Ensure matte hasn't been consumed (unlikely in this loop unless recursive mattes?)
                    if !consumed_indices.contains(&matte_idx) {
                        consumed_indices.insert(matte_idx);
                        let matte_layer = &layers[matte_idx];

                        if let Some(mut content_node) = self.process_layer(layer, &layer_map) {
                            if let Some(matte_node) = self.process_layer(matte_layer, &layer_map) {
                                let mode = match tt {
                                    1 => MatteMode::Alpha,
                                    2 => MatteMode::AlphaInverted,
                                    3 => MatteMode::Luma,
                                    4 => MatteMode::LumaInverted,
                                    _ => MatteMode::Alpha,
                                };
                                content_node.matte = Some(Box::new(Matte {
                                    mode,
                                    node: matte_node,
                                }));
                            }
                            nodes.push(content_node);
                        }
                        continue;
                    }
                }
            }

            if let Some(node) = self.process_layer(layer, &layer_map) {
                nodes.push(node);
            }
        }

        RenderNode {
            transform: Mat3::IDENTITY,
            alpha: 1.0,
            blend_mode: BlendMode::Normal,
            content: NodeContent::Group(nodes),
            masks: vec![], styles: vec![],
            matte: None,
            effects: vec![],
            is_adjustment_layer: false,
        }
    }

    fn process_layer(
        &mut self,
        layer: &'a data::Layer,
        layer_map: &HashMap<u32, &'a data::Layer>,
    ) -> Option<RenderNode> {
        let is_adjustment_layer = layer.ao == Some(1);

        // Visibility check (in/out points)
        // Note: frame is global time. Layer.st is start time offset.
        // Layer exists from ip to op.
        // But ip/op are usually relative to composition start?
        // Usually ip/op are absolute frame numbers in the parent timeline.
        if self.frame < layer.ip || self.frame >= layer.op {
            return None;
        }

        // Calculate Transform
        // We need full transform chain relative to this composition.
        // But wait, if we build the tree hierarchically, we only need local transform?
        // I decided earlier to resolve global transform because parenting doesn't imply nesting in RenderTree.
        // Let's resolve global transform.

        let transform = self.resolve_transform(layer, layer_map);

        // Opacity
        let opacity = Animator::resolve(&layer.ks.o, self.frame - layer.st, |v| *v / 100.0, 1.0);

        // Content
        let content = if let Some(shapes) = &layer.shapes {
            // Shape Layer
            let shape_nodes = self.process_shapes(shapes, self.frame - layer.st);
            NodeContent::Group(shape_nodes)
        } else if let Some(text_data) = &layer.t {
            // Text Layer
            let doc = Animator::resolve(
                &text_data.d,
                self.frame - layer.st,
                |v| v.clone(),
                data::TextDocument::default(),
            );

            let base_fill_color = Vec4::new(doc.fc[0], doc.fc[1], doc.fc[2], 1.0);
            let base_stroke_color = if let Some(sc) = &doc.sc {
                Some(Vec4::new(sc[0], sc[1], sc[2], 1.0))
            } else {
                None
            };

            let chars: Vec<char> = doc.t.chars().collect();
            let char_count = chars.len();

            let mut glyphs = Vec::with_capacity(char_count);

            for &c in &chars {
                let g = RenderGlyph {
                    character: c,
                    pos: Vec2::ZERO,
                    scale: Vec2::ONE,
                    rotation: 0.0,
                    tracking: 0.0,
                    alpha: 1.0,
                    fill: Some(Fill {
                        paint: Paint::Solid(base_fill_color),
                        opacity: 1.0,
                        rule: FillRule::NonZero,
                    }),
                    stroke: if let Some(col) = base_stroke_color {
                        Some(Stroke {
                            paint: Paint::Solid(col),
                            width: doc.sw.unwrap_or(1.0),
                            opacity: 1.0,
                            cap: LineCap::Round,
                            join: LineJoin::Round,
                            miter_limit: None,
                            dash: None,
                        })
                    } else {
                        None
                    },
                };
                glyphs.push(g);
            }

            // Animators
            if let Some(animators) = &text_data.a {
                for animator in animators {
                    // 1. Selector
                    let sel = &animator.s;
                    // Resolve Start, End, Offset
                    let start_val = Animator::resolve(
                        sel.s.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| *v,
                        0.0,
                    );
                    let end_val = Animator::resolve(
                        sel.e.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| *v,
                        100.0,
                    );
                    let offset_val = Animator::resolve(
                        sel.o.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| *v,
                        0.0,
                    );

                    let start_idx = char_count as f32 * start_val / 100.0;
                    let end_idx = char_count as f32 * end_val / 100.0;
                    let offset_idx = char_count as f32 * offset_val / 100.0;

                    // 2. Style
                    let style = &animator.a;
                    let p_delta = Animator::resolve(
                        style.p.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| Vec2::from_slice(v),
                        Vec2::ZERO,
                    );
                    let s_val = Animator::resolve(
                        style.s.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| Vec2::from_slice(v),
                        Vec2::new(100.0, 100.0),
                    );
                    let o_val = Animator::resolve(
                        style.o.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| *v,
                        100.0,
                    );
                    let r_val = Animator::resolve(
                        style.r.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| *v,
                        0.0,
                    );
                    let t_val = Animator::resolve(
                        style.t.as_ref().unwrap_or(&data::Property::default()),
                        self.frame - layer.st,
                        |v| *v,
                        0.0,
                    );

                    let fc_val = if let Some(fc_prop) = &style.fc {
                        Some(Animator::resolve(
                            fc_prop,
                            self.frame - layer.st,
                            |v| Vec4::from_slice(v),
                            Vec4::ONE,
                        ))
                    } else {
                        None
                    };

                    let sc_val = if let Some(sc_prop) = &style.sc {
                        Some(Animator::resolve(
                            sc_prop,
                            self.frame - layer.st,
                            |v| Vec4::from_slice(v),
                            Vec4::ONE,
                        ))
                    } else {
                        None
                    };

                    for (i, glyph) in glyphs.iter_mut().enumerate() {
                        let idx = i as f32;
                        let effective_start = start_idx + offset_idx;
                        let effective_end = end_idx + offset_idx;

                        let overlap_start = idx.max(effective_start);
                        let overlap_end = (idx + 1.0).min(effective_end);

                        let factor = (overlap_end - overlap_start).max(0.0).min(1.0);

                        if factor > 0.0 {
                            // Position (Additive)
                            glyph.pos += p_delta * factor;

                            // Scale
                            let scale_factor = Vec2::ONE + ((s_val / 100.0) - Vec2::ONE) * factor;
                            glyph.scale *= scale_factor;

                            // Rotation (Additive)
                            glyph.rotation += r_val * factor;

                            // Tracking (Additive)
                            glyph.tracking += t_val * factor;

                            // Opacity (Multiplicative)
                            let target_alpha = o_val / 100.0;
                            // Lerp alpha multiplier: 1.0 -> target_alpha based on factor
                            let alpha_mult = 1.0 + (target_alpha - 1.0) * factor;
                            glyph.alpha *= alpha_mult;

                            // Fill Color (Mix)
                            if let Some(target_color) = fc_val {
                                if let Some(fill) = &mut glyph.fill {
                                    if let Paint::Solid(current_color) = &mut fill.paint {
                                        *current_color = current_color.lerp(target_color, factor);
                                    }
                                }
                            }

                            // Stroke Color (Mix)
                            if let Some(target_color) = sc_val {
                                if let Some(stroke) = &mut glyph.stroke {
                                    if let Paint::Solid(current_color) = &mut stroke.paint {
                                        *current_color = current_color.lerp(target_color, factor);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Layout & Wrapping
            if let Some(measurer) = self.text_measurer {
                let box_size = doc.sz.map(|v| Vec2::from_slice(&v));
                let box_pos = doc.ps.map(|v| Vec2::from_slice(&v)).unwrap_or(Vec2::ZERO);
                let tracking_val = doc.tr; // Usually in 1/1000s of em or px? Lottie Skia treats as px additive.

                if let Some(sz) = box_size {
                    // --- Box Text ---
                    let box_width = sz.x;
                    let mut lines: Vec<Vec<usize>> = Vec::new(); // Indices of glyphs
                    let mut current_line: Vec<usize> = Vec::new();
                    let mut current_line_width = 0.0;

                    // Simple word tokenizer
                    // We iterate glyphs. We group them into words.
                    // A "word" is a sequence of non-space characters followed by spaces.

                    let mut i = 0;
                    while i < glyphs.len() {
                        // Find word start
                        let start = i;
                        // Find next space or end
                        let mut end = i;
                        let mut word_width = 0.0;

                        while end < glyphs.len() {
                            let g = &glyphs[end];
                            // Measure glyph
                            let char_str = g.character.to_string();
                            let w = measurer.measure(&char_str, &doc.f, doc.s);
                            let advance = w + tracking_val + g.tracking;

                            word_width += advance;

                            let is_space = g.character == ' ';
                            let is_newline = g.character == '\n';

                            end += 1;

                            if is_space || is_newline {
                                break;
                            }
                        }

                        // Check if word fits
                        // If current_line is empty, we must add it (even if overflow)
                        // If current_line not empty, and current + word > box, wrap.

                        // Note: word includes the trailing space.
                        // If wrapping, we might drop the space? Lottie usually wraps at space.
                        // The space should stay on the line it belongs to, or disappear?
                        // Usually trailing space at EOL is invisible/ignored for width but conceptually there.

                        let is_newline = if end > 0 { glyphs[end-1].character == '\n' } else { false };

                        if is_newline {
                             // Force break
                             // Add word to current line
                             for k in start..end {
                                 current_line.push(k);
                             }
                             lines.push(current_line);
                             current_line = Vec::new();
                             current_line_width = 0.0;
                        } else {
                            if !current_line.is_empty() && current_line_width + word_width > box_width {
                                // Wrap
                                lines.push(current_line);
                                current_line = Vec::new();
                                current_line_width = 0.0;
                            }

                            // Add word
                            for k in start..end {
                                current_line.push(k);
                            }
                            current_line_width += word_width;
                        }

                        i = end;
                    }
                    if !current_line.is_empty() {
                        lines.push(current_line);
                    }

                    // Vertical Alignment (Top)
                    let mut current_y = box_pos.y;
                    // Note: If drawing from baseline, we might need to add ascent?
                    // User said: "current_y = ps.y".
                    // But also "The `ps` property defines the center or top-left".
                    // Let's assume text starts at ps.y.

                    for line_indices in lines {
                         // Measure line width (excluding trailing spaces for justification?)
                         let mut line_width = 0.0;
                         // Also calculate advances for positioning
                         let mut advances = Vec::new();

                         // Determine visible width (trim trailing whitespace for alignment)
                         // But we need to calculate positions for all.

                         for &idx in &line_indices {
                             let g = &glyphs[idx];
                             let w = measurer.measure(&g.character.to_string(), &doc.f, doc.s);
                             let advance = w + tracking_val + g.tracking;
                             advances.push(advance);
                             line_width += advance;
                         }

                         // Calculate Start X based on Justification
                         // Justify relative to Box Width
                         let align_width = line_width; // Or trim trailing space?
                         // For simplicity, use full line width calculated.

                         let start_x = match doc.j {
                             1 => box_width - align_width, // Right (relative to box)
                             2 => (box_width - align_width) / 2.0, // Center
                             _ => 0.0, // Left
                         };

                         let mut x = box_pos.x + start_x;
                         for (k, &idx) in line_indices.iter().enumerate() {
                             let g = &mut glyphs[idx];
                             // Apply calculated position (plus any animator position delta)
                             // g.pos already has animator delta. We need to ADD the layout position.
                             // Wait, g.pos currently stores the animator delta (p_delta).
                             // We should add the layout pos to it.
                             g.pos += Vec2::new(x, current_y);

                             x += advances[k];
                         }

                         current_y += doc.lh;
                    }

                } else {
                    // --- Point Text (with Measurer) ---
                    // Handle newlines, but no wrapping width
                    let mut current_y = 0.0;

                    // For Justification in Point Text, we need to know line width first.
                    // Split into lines by '\n'
                    let mut lines: Vec<Vec<usize>> = Vec::new();
                    let mut current_line = Vec::new();

                    for (i, g) in glyphs.iter().enumerate() {
                        if g.character == '\n' {
                            lines.push(current_line);
                            current_line = Vec::new();
                        } else {
                            current_line.push(i);
                        }
                    }
                    lines.push(current_line);

                    for line_indices in lines {
                        let mut line_width = 0.0;
                        let mut advances = Vec::new();
                        for &idx in &line_indices {
                             let g = &glyphs[idx];
                             let w = measurer.measure(&g.character.to_string(), &doc.f, doc.s);
                             let advance = w + tracking_val + g.tracking;
                             advances.push(advance);
                             line_width += advance;
                        }

                        // Point Text Justification (Anchor point is usually origin)
                        // Left: start at 0.
                        // Center: start at -width/2.
                        // Right: start at -width.

                        let start_x = match doc.j {
                            1 => -line_width,
                            2 => -line_width / 2.0,
                            _ => 0.0,
                        };

                        let mut x = start_x;
                        for (k, &idx) in line_indices.iter().enumerate() {
                             let g = &mut glyphs[idx];
                             g.pos += Vec2::new(x, current_y);
                             x += advances[k];
                        }
                        current_y += doc.lh;
                    }
                }
            } else {
                 // --- Fallback (No Measurer) ---
                 // If we have no measurer, we can't calculate layout.
                 // RenderGlyphs will have positions relative to (0,0) (animator deltas only).
                 // The Renderer will have to deal with it, OR we emit warnings?
                 // Since the task requirement is to implement wrapping in Core,
                 // and we added the Trait, we assume the host provides it.
                 // For now, leave as is (positions are just animator deltas).
                 // Point text will likely render all on top of each other in the new renderer
                 // if we remove layout logic there.
                 // But typically, simple examples might not set the measurer.
                 // I should probably implement a naive fallback (fixed width)?
                 // Let's assume 10px width per char as fallback to prevent 0-overlap pileup.

                 let fixed_width = 10.0;
                 let mut x = 0.0;
                 let mut y = 0.0;
                 for g in &mut glyphs {
                     if g.character == '\n' {
                         x = 0.0;
                         y += doc.lh;
                     } else {
                         g.pos += Vec2::new(x, y);
                         x += fixed_width + doc.tr + g.tracking;
                     }
                 }
            }

            NodeContent::Text(Text {
                glyphs,
                font_family: doc.f,
                size: doc.s,
                justify: match doc.j {
                    1 => Justification::Right,
                    2 => Justification::Center,
                    _ => Justification::Left,
                },
                tracking: doc.tr,
                line_height: doc.lh,
            })
        } else if let Some(ref_id) = &layer.ref_id {
            // Image or Precomp
            if let Some(asset) = self.assets.get(ref_id) {
                if let Some(layers) = &asset.layers {
                    // Precomp
                    let local_frame = if let Some(tm_prop) = &layer.tm {
                        let tm_sec = Animator::resolve(tm_prop, self.frame, |v| *v, 0.0);
                        tm_sec * self.model.fr
                    } else {
                        self.frame - layer.st
                    };

                    // We need to recursively build the composition.
                    let mut sub_builder = SceneGraphBuilder::new(
                        self.model,
                        local_frame,
                        self.external_assets,
                        self.text_measurer,
                    );
                    let root = sub_builder.build_composition(layers);
                    // The root returned is a Group. We can unwrap it or wrap it.
                    // BUT, precomp layers transform apply to the group.
                    // We are already calculating transform for this layer.
                    // The content is the precomp's root content.
                    root.content
                } else {
                    // Image
                    // Check external assets first
                    let data = if let Some(ImageSource::Data(bytes)) =
                        self.external_assets.get(&asset.id)
                    {
                        Some(bytes.clone())
                    } else if let Some(p) = &asset.p {
                        if p.starts_with("data:image/") && p.contains(";base64,") {
                            // Base64
                            let split: Vec<&str> = p.splitn(2, ',').collect();
                            if split.len() > 1 {
                                match BASE64_STANDARD.decode(split[1]) {
                                    Ok(bytes) => Some(bytes),
                                    Err(e) => {
                                        eprintln!("Lottie Error: Failed to decode embedded image: {}", e);
                                        None
                                    }
                                }
                            } else {
                                None
                            }
                        } else {
                            // Path. Not implemented yet (needs file IO).
                            // User recommended "Embedded and Local Files".
                            // I will attempt to read file relative to execution?
                            // Unsafe in library code usually, but okay for this task.
                            if let Ok(bytes) = std::fs::read(p) {
                                Some(bytes)
                            } else {
                                None
                            }
                        }
                    } else {
                        None
                    };

                    NodeContent::Image(Image {
                        data,
                        width: asset.w.unwrap_or(100),
                        height: asset.h.unwrap_or(100),
                    })
                }
            } else {
                NodeContent::Group(vec![])
            }
        } else if let Some(color) = &layer.color {
            // Solid
            // Parse hex color?
            // "sc": "#rrggbb"
            // Simple solid rect
            let w = layer.sw.unwrap_or(100) as f64;
            let h = layer.sh.unwrap_or(100) as f64;
            let mut path = BezPath::new();
            path.move_to((0.0, 0.0));
            path.line_to((w, 0.0));
            path.line_to((w, h));
            path.line_to((0.0, h));
            path.close_path();

            // Hex parse
            let c_str = color.trim_start_matches('#');
            let r = u8::from_str_radix(&c_str[0..2], 16).unwrap_or(0) as f32 / 255.0;
            let g = u8::from_str_radix(&c_str[2..4], 16).unwrap_or(0) as f32 / 255.0;
            let b = u8::from_str_radix(&c_str[4..6], 16).unwrap_or(0) as f32 / 255.0;

            NodeContent::Shape(renderer::Shape {
                geometry: renderer::ShapeGeometry::Path(path),
                fill: Some(Fill {
                    paint: Paint::Solid(Vec4::new(r, g, b, 1.0)),
                    opacity: 1.0,
                    rule: FillRule::NonZero,
                }),
                stroke: None,
                trim: None,
            })
        } else {
            NodeContent::Group(vec![])
        };

        // Masks
        let masks = if let Some(props) = &layer.masks_properties {
            self.process_masks(props, self.frame - layer.st)
        } else {
            vec![]
        };

        // Effects
        let effects = self.process_effects(layer);
        let styles = self.process_layer_styles(layer);

        Some(RenderNode {
            transform,
            alpha: opacity,
            blend_mode: BlendMode::Normal, // Map blend mode
            content,
            masks,
            matte: None,
            effects,
            styles,
            is_adjustment_layer,
        })
    }

    fn process_layer_styles(&self, layer: &data::Layer) -> Vec<LayerStyle> {
        let mut styles = Vec::new();
        if let Some(sy_list) = &layer.sy {
            for sy in sy_list {
                let ty = sy.ty.unwrap_or(8);
                let mut kind = None;
                if ty == 0 { kind = Some("DropShadow"); }
                else if ty == 1 { kind = Some("InnerShadow"); }
                else if ty == 2 { kind = Some("OuterGlow"); }
                else if let Some(nm) = &sy.nm {
                    if nm.contains("Stroke") { kind = Some("Stroke"); }
                }

                if kind.is_none() {
                    if ty == 3 || ty == 8 {
                        kind = Some("Stroke");
                    }
                }

                if let Some(k) = kind {
                    match k {
                        "DropShadow" => {
                             let color = self.resolve_json_vec4_arr(&sy.c, self.frame - layer.st);
                             let opacity = Animator::resolve(&sy.o, self.frame - layer.st, |v| *v / 100.0, 1.0);
                             let angle = Animator::resolve(&sy.a, self.frame - layer.st, |v| *v, 0.0);
                             let distance = Animator::resolve(&sy.d, self.frame - layer.st, |v| *v, 0.0);
                             let size = Animator::resolve(&sy.s, self.frame - layer.st, |v| *v, 0.0);
                             let spread = Animator::resolve(&sy.ch, self.frame - layer.st, |v| *v, 0.0);
                             styles.push(LayerStyle::DropShadow {
                                 color, opacity, angle, distance, size, spread
                             });
                        },
                        "InnerShadow" => {
                             let color = self.resolve_json_vec4_arr(&sy.c, self.frame - layer.st);
                             let opacity = Animator::resolve(&sy.o, self.frame - layer.st, |v| *v / 100.0, 1.0);
                             let angle = Animator::resolve(&sy.a, self.frame - layer.st, |v| *v, 0.0);
                             let distance = Animator::resolve(&sy.d, self.frame - layer.st, |v| *v, 0.0);
                             let size = Animator::resolve(&sy.s, self.frame - layer.st, |v| *v, 0.0);
                             let choke = Animator::resolve(&sy.ch, self.frame - layer.st, |v| *v, 0.0);
                             styles.push(LayerStyle::InnerShadow {
                                 color, opacity, angle, distance, size, choke
                             });
                        },
                        "OuterGlow" => {
                             let color = self.resolve_json_vec4_arr(&sy.c, self.frame - layer.st);
                             let opacity = Animator::resolve(&sy.o, self.frame - layer.st, |v| *v / 100.0, 1.0);
                             let size = Animator::resolve(&sy.s, self.frame - layer.st, |v| *v, 0.0);
                             let range = Animator::resolve(&sy.ch, self.frame - layer.st, |v| *v, 0.0);
                             styles.push(LayerStyle::OuterGlow {
                                 color, opacity, size, range
                             });
                        },
                        "Stroke" => {
                             let color = self.resolve_json_vec4_arr(&sy.c, self.frame - layer.st);
                             let opacity = Animator::resolve(&sy.o, self.frame - layer.st, |v| *v / 100.0, 1.0);
                             let width = Animator::resolve(&sy.s, self.frame - layer.st, |v| *v, 0.0);
                             styles.push(LayerStyle::Stroke {
                                 color, width, opacity
                             });
                        },
                        _ => {}
                    }
                }
            }
        }
        styles
    }

    fn process_effects(&self, layer: &data::Layer) -> Vec<Effect> {
        let mut effects = Vec::new();
        if let Some(ef_list) = &layer.ef {
            for ef in ef_list {
                if let Some(en) = ef.en {
                    if en == 0 {
                        continue;
                    }
                }
                let ty = ef.ty.unwrap_or(0);
                let values = if let Some(vals) = &ef.ef {
                    vals
                } else {
                    continue;
                };

                match ty {
                    20 => {
                        // Tint
                        // 0: Black, 1: White, 2: Amount
                        let black = self.find_effect_vec4(values, 0, "Black", layer);
                        let white = self.find_effect_vec4(values, 1, "White", layer);
                        let amount = self.find_effect_scalar(values, 2, "Intensity", layer) / 100.0;
                        effects.push(Effect::Tint {
                            black,
                            white,
                            amount,
                        });
                    }
                    21 => {
                        // Fill
                        // 2: Color, 6: Opacity
                        let color = self.find_effect_vec4(values, 2, "Color", layer);
                        let opacity = self.find_effect_scalar(values, 6, "Opacity", layer) / 100.0;
                        effects.push(Effect::Fill { color, opacity });
                    }
                    22 => {
                        // Stroke
                        // 3: Color, 4: Size, 6: Opacity
                        let color = self.find_effect_vec4(values, 3, "Color", layer);
                        let width = self.find_effect_scalar(values, 4, "Brush Size", layer);
                        let opacity = self.find_effect_scalar(values, 6, "Opacity", layer) / 100.0;

                        // Use 9999 to force name search since indices vary
                        let all_masks_val =
                            self.find_effect_scalar(values, 9999, "All Masks", layer);
                        let all_masks = all_masks_val > 0.5;

                        let mut mask_idx_val = self.find_effect_scalar(values, 9999, "Path", layer);
                        if mask_idx_val < 0.5 {
                            mask_idx_val = self.find_effect_scalar(values, 9999, "Mask", layer);
                        }

                        let mask_index = if mask_idx_val >= 0.5 {
                            Some(mask_idx_val.round() as usize)
                        } else {
                            None
                        };

                        effects.push(Effect::Stroke {
                            color,
                            width,
                            opacity,
                            mask_index,
                            all_masks,
                        });
                    }
                    23 => {
                        // Tritone
                        // 0: Highlights (bright), 1: Midtones (mid), 2: Shadows (dark)
                        let highlights = self.find_effect_vec4(values, 0, "bright", layer);
                        let midtones = self.find_effect_vec4(values, 1, "mid", layer);
                        let shadows = self.find_effect_vec4(values, 2, "dark", layer);
                        effects.push(Effect::Tritone {
                            highlights,
                            midtones,
                            shadows,
                        });
                    }
                    24 => {
                        // Levels (Pro Levels)
                        // 3: In Black, 4: In White, 5: Gamma, 6: Out Black, 7: Out White
                        let in_black = self.find_effect_scalar(values, 3, "inblack", layer);
                        let in_white = self.find_effect_scalar(values, 4, "inwhite", layer);
                        let gamma = self.find_effect_scalar(values, 5, "gamma", layer);
                        let out_black = self.find_effect_scalar(values, 6, "outblack", layer);
                        let out_white = self.find_effect_scalar(values, 7, "outwhite", layer);
                        effects.push(Effect::Levels {
                            in_black,
                            in_white,
                            gamma,
                            out_black,
                            out_white,
                        });
                    }
                    _ => {}
                }
            }
        }
        effects
    }

    fn find_effect_scalar(
        &self,
        values: &[data::EffectValue],
        index: usize,
        name_hint: &str,
        layer: &data::Layer,
    ) -> f32 {
        // Try index first
        if let Some(v) = values.get(index) {
            if let Some(prop) = &v.v {
                // Check if it looks valid or if we should fallback?
                // Assuming index is reliable for standard effects.
                return self.resolve_json_scalar(prop, self.frame - layer.st);
            }
        }
        // Try name
        for v in values {
            if let Some(nm) = &v.nm {
                if nm.contains(name_hint) {
                    if let Some(prop) = &v.v {
                        return self.resolve_json_scalar(prop, self.frame - layer.st);
                    }
                }
            }
        }
        0.0
    }

    fn find_effect_vec4(
        &self,
        values: &[data::EffectValue],
        index: usize,
        name_hint: &str,
        layer: &data::Layer,
    ) -> Vec4 {
        // Try index
        if let Some(v) = values.get(index) {
            if let Some(prop) = &v.v {
                return self.resolve_json_vec4(prop, self.frame - layer.st);
            }
        }
        // Try name
        for v in values {
            if let Some(nm) = &v.nm {
                if nm.contains(name_hint) {
                    if let Some(prop) = &v.v {
                        return self.resolve_json_vec4(prop, self.frame - layer.st);
                    }
                }
            }
        }
        Vec4::ZERO
    }

    fn resolve_json_scalar(&self, prop: &data::Property<serde_json::Value>, frame: f32) -> f32 {
        Animator::resolve(
            prop,
            frame,
            |v| {
                if let Some(n) = v.as_f64() {
                    n as f32
                } else if let Some(arr) = v.as_array() {
                    arr.get(0).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
                } else {
                    0.0
                }
            },
            0.0,
        )
    }

    fn resolve_json_vec4(&self, prop: &data::Property<serde_json::Value>, frame: f32) -> Vec4 {
        Animator::resolve(
            prop,
            frame,
            |v| {
                if let Some(arr) = v.as_array() {
                    let r = arr.get(0).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    let g = arr.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    let b = arr.get(2).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
                    let a = arr.get(3).and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
                    Vec4::new(r, g, b, a)
                } else {
                    Vec4::ZERO
                }
            },
            Vec4::ZERO,
        )
    }

    fn resolve_json_vec4_arr(&self, prop: &data::Property<Vec<f32>>, frame: f32) -> Vec4 {
        Animator::resolve(
            prop,
            frame,
            |v| {
                if v.len() >= 4 {
                    Vec4::new(v[0], v[1], v[2], v[3])
                } else if v.len() >= 3 {
                    Vec4::new(v[0], v[1], v[2], 1.0)
                } else {
                    Vec4::ZERO
                }
            },
            Vec4::ONE,
        )
    }

    fn resolve_transform(&self, layer: &data::Layer, map: &HashMap<u32, &data::Layer>) -> Mat3 {
        let local = self.get_local_transform(layer);
        if let Some(parent_ind) = layer.parent {
            if let Some(parent) = map.get(&parent_ind) {
                return self.resolve_transform(parent, map) * local;
            }
        }
        local
    }

    fn get_local_transform(&self, layer: &data::Layer) -> Mat3 {
        // Calculate T, R, S, A
        let t_frame = self.frame - layer.st;
        let ks = &layer.ks;

        let anchor = Animator::resolve(&ks.a, t_frame, |v| Vec2::from_slice(v), Vec2::ZERO);
        let pos = match &ks.p {
            data::PositionProperty::Unified(p) => {
                Animator::resolve(p, t_frame, |v| Vec2::from_slice(v), Vec2::ZERO)
            }
            data::PositionProperty::Split { x, y } => {
                let px = Animator::resolve(x, t_frame, |v| *v, 0.0);
                let py = Animator::resolve(y, t_frame, |v| *v, 0.0);
                Vec2::new(px, py)
            }
        };
        let scale = Animator::resolve(&ks.s, t_frame, |v| Vec2::from_slice(v) / 100.0, Vec2::ONE);
        let rotation = Animator::resolve(&ks.r, t_frame, |v| v.to_radians(), 0.0);

        // Matrix: T * R * S * -A
        // Translate to P
        let mat_t = Mat3::from_translation(pos);
        // Rotate
        let mat_r = Mat3::from_rotation_z(rotation);
        // Scale
        let mat_s = Mat3::from_scale(scale);
        // Translate by -Anchor
        let mat_a = Mat3::from_translation(-anchor);

        mat_t * mat_r * mat_s * mat_a
    }

    fn process_shapes(&self, shapes: &'a [data::Shape], frame: f32) -> Vec<RenderNode> {
        let mut processed_nodes = Vec::new();
        let mut active_geometries: Vec<PendingGeometry> = Vec::new();

        let mut trim: Option<Trim> = None;
        for item in shapes {
            if let data::Shape::Trim(t) = item {
                let s = Animator::resolve(&t.s, frame, |v| *v / 100.0, 0.0);
                let e = Animator::resolve(&t.e, frame, |v| *v / 100.0, 1.0);
                let o = Animator::resolve(&t.o, frame, |v| *v / 360.0, 0.0);
                trim = Some(Trim {
                    start: s,
                    end: e,
                    offset: o,
                });
            }
        }

        for item in shapes {
            match item {
                data::Shape::MergePaths(mp) => {
                    // Consume all active geometries into a single merged geometry
                    if !active_geometries.is_empty() {
                        let mode = match mp.mm {
                            1 => MergeMode::Merge,
                            2 => MergeMode::Add,
                            3 => MergeMode::Subtract,
                            4 => MergeMode::Intersect,
                            5 => MergeMode::Exclude,
                            _ => MergeMode::Merge,
                        };
                        let merged = PendingGeometry {
                            kind: GeometryKind::Merge(active_geometries.clone(), mode),
                            transform: Mat3::IDENTITY,
                        };
                        active_geometries.clear();
                        active_geometries.push(merged);
                    }
                }
                data::Shape::Path(p) => {
                    let path = self.convert_path(p, frame);
                    active_geometries.push(PendingGeometry {
                        kind: GeometryKind::Path(path),
                        transform: Mat3::IDENTITY,
                    });
                }
                data::Shape::Rect(r) => {
                    let size = Animator::resolve(&r.s, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let pos = Animator::resolve(&r.p, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let radius = Animator::resolve(&r.r, frame, |v| *v, 0.0);
                    active_geometries.push(PendingGeometry {
                        kind: GeometryKind::Rect { size, pos, radius },
                        transform: Mat3::IDENTITY,
                    });
                }
                data::Shape::Ellipse(e) => {
                    let size = Animator::resolve(&e.s, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let pos = Animator::resolve(&e.p, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    active_geometries.push(PendingGeometry {
                        kind: GeometryKind::Ellipse { size, pos },
                        transform: Mat3::IDENTITY,
                    });
                }
                data::Shape::Polystar(sr) => {
                    let pos = match &sr.p {
                        data::PositionProperty::Unified(p) => {
                            Animator::resolve(p, 0.0, |v| Vec2::from_slice(v), Vec2::ZERO)
                        }
                        data::PositionProperty::Split { x, y } => {
                            let px = Animator::resolve(x, 0.0, |v| *v, 0.0);
                            let py = Animator::resolve(y, 0.0, |v| *v, 0.0);
                            Vec2::new(px, py)
                        }
                    };
                    let or = Animator::resolve(&sr.or, frame, |v| *v, 0.0);
                    let os = Animator::resolve(&sr.os, frame, |v| *v, 0.0);
                    let r = Animator::resolve(&sr.r, frame, |v| *v, 0.0);
                    let pt = Animator::resolve(&sr.pt, frame, |v| *v, 5.0);
                    let ir = if let Some(prop) = &sr.ir {
                        Animator::resolve(prop, 0.0, |v| *v, 0.0)
                    } else {
                        0.0
                    };
                    let is = if let Some(prop) = &sr.is {
                        Animator::resolve(prop, 0.0, |v| *v, 0.0)
                    } else {
                        0.0
                    };

                    active_geometries.push(PendingGeometry {
                        kind: GeometryKind::Polystar(PolystarParams {
                            pos,
                            outer_radius: or,
                            inner_radius: ir,
                            outer_roundness: os,
                            inner_roundness: is,
                            rotation: r,
                            points: pt,
                            kind: sr.sy,
                            corner_radius: 0.0,
                        }),
                        transform: Mat3::IDENTITY,
                    });
                }
                data::Shape::RoundCorners(rd) => {
                    let r = Animator::resolve(&rd.r, frame, |v| *v, 0.0);
                    if r > 0.0 {
                        for geom in &mut active_geometries {
                            match &mut geom.kind {
                                GeometryKind::Rect { radius, .. } => *radius += r,
                                GeometryKind::Polystar(p) => p.corner_radius += r,
                                _ => {}
                            }
                        }
                    }
                }
                data::Shape::ZigZag(zz) => {
                    let ridges = Animator::resolve(&zz.r, frame, |v| *v, 0.0);
                    let size = Animator::resolve(&zz.s, frame, |v| *v, 0.0);
                    let pt = Animator::resolve(&zz.pt, frame, |v| *v, 1.0); // 1 = Corner, 2 = Smooth

                    let modifier = ZigZagModifier {
                        ridges,
                        size,
                        smooth: pt > 1.5,
                    };

                    self.apply_modifier_to_active(&mut active_geometries, &modifier);
                }
                data::Shape::PuckerBloat(pb) => {
                    let amount = Animator::resolve(&pb.a, frame, |v| *v, 0.0);
                    // Center? PuckerBloat usually doesn't have explicit center in struct?
                    // I defined struct with only 'a'.
                    // Usually it operates on the center of the shape.
                    // But active_geometries has multiple shapes.
                    // I will assume (0,0) is center or bounds center?
                    // Lottie PuckerBloat applies to the shape relative to its own transform.
                    // Since I bake geometries which include transform, I should use (0,0) if transform is correct?
                    // Actually, PendingGeometry.transform applies to the shape.
                    // If I bake, I bake the transform. So center is affected.
                    // I'll use Vec2::ZERO as default center if not provided?
                    // Wait, `PuckerBloatShape` struct I added has `nm` and `a`. No `c`.
                    // So Center must be implicit (0,0).

                    let modifier = PuckerBloatModifier {
                        amount,
                        center: Vec2::ZERO, // Implicit center
                    };
                    self.apply_modifier_to_active(&mut active_geometries, &modifier);
                }
                data::Shape::Twist(tw) => {
                    let angle = Animator::resolve(&tw.a, frame, |v| *v, 0.0);
                    let center =
                        Animator::resolve(&tw.c, frame, |v| Vec2::from_slice(v), Vec2::ZERO);

                    let modifier = TwistModifier { angle, center };
                    self.apply_modifier_to_active(&mut active_geometries, &modifier);
                }
                data::Shape::OffsetPath(op) => {
                    let amount = Animator::resolve(&op.a, frame, |v| *v, 0.0);
                    let miter_limit = op.ml.unwrap_or(4.0);
                    let line_join = op.lj;

                    let modifier = OffsetPathModifier {
                        amount,
                        miter_limit,
                        line_join,
                    };
                    self.apply_modifier_to_active(&mut active_geometries, &modifier);
                }
                data::Shape::WigglePath(wg) => {
                    let speed = Animator::resolve(&wg.s, frame, |v| *v, 0.0);
                    let size = Animator::resolve(&wg.w, frame, |v| *v, 0.0);
                    let correlation = Animator::resolve(&wg.r, frame, |v| *v, 0.0);
                    let seed_prop = Animator::resolve(&wg.sh, frame, |v| *v, 0.0);

                    // Seed from struct or generic?
                    // Use seed_prop as base.

                    let modifier = WiggleModifier {
                        seed: seed_prop,
                        time: frame / 60.0, // Frame is usually frames. Convert to seconds?
                        // LottiePlayer has frame_rate. SceneGraphBuilder has 'frame'.
                        // Assuming frame is frame number. time = frame / fps.
                        // I don't have fps here easily?
                        // SceneGraphBuilder doesn't store fps.
                        // But I have `frame` (current time in frames).
                        // Speed is "wiggles per second".
                        // Lottie defaults 30fps or 60fps?
                        // I'll assume 60.0 or pass it?
                        // `LottieJson` has `fr`.
                        speed: speed / self.model.fr, // Convert wiggles/sec to wiggles/frame?
                        // Wait, modifier logic uses `time * speed`.
                        // If I pass `frame` as time, then speed should be wiggles/frame.
                        // speed is wiggles/sec. frame is frames.
                        // t_sec = frame / fps.
                        // input = t_sec * speed = (frame/fps) * speed.
                        // I'll pass calculated value.
                        amount: size,
                        correlation,
                    };

                    // Fix speed calculation inside apply?
                    // I will instantiate struct with correct time/speed combo.
                    // Actually, I can just pass `frame` and `speed_per_frame`.
                    // speed_per_frame = speed / model.fr

                    let mut mod_clone = modifier; // copy
                    mod_clone.time = frame;
                    mod_clone.speed = speed / self.model.fr;

                    self.apply_modifier_to_active(&mut active_geometries, &mod_clone);
                }
                data::Shape::Repeater(rp) => {
                    let copies = Animator::resolve(&rp.c, frame, |v| *v, 0.0);
                    let offset = Animator::resolve(&rp.o, frame, |v| *v, 0.0);

                    let t_anchor =
                        Animator::resolve(&rp.tr.t.a, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let t_pos = match &rp.tr.t.p {
                        data::PositionProperty::Unified(p) => {
                            Animator::resolve(p, 0.0, |v| Vec2::from_slice(v), Vec2::ZERO)
                        }
                        data::PositionProperty::Split { x, y } => {
                            let px = Animator::resolve(x, 0.0, |v| *v, 0.0);
                            let py = Animator::resolve(y, 0.0, |v| *v, 0.0);
                            Vec2::new(px, py)
                        }
                    };
                    let t_scale = Animator::resolve(
                        &rp.tr.t.s,
                        0.0,
                        |v| Vec2::from_slice(v) / 100.0,
                        Vec2::ONE,
                    );
                    let t_rot = Animator::resolve(&rp.tr.t.r, frame, |v| *v, 0.0);

                    let start_opacity = Animator::resolve(&rp.tr.so, frame, |v| *v / 100.0, 1.0);
                    let end_opacity = Animator::resolve(&rp.tr.eo, frame, |v| *v / 100.0, 1.0);

                    self.apply_repeater(
                        copies,
                        offset,
                        t_anchor,
                        t_pos,
                        t_scale,
                        t_rot,
                        start_opacity,
                        end_opacity,
                        &mut active_geometries,
                        &mut processed_nodes,
                    );
                }
                data::Shape::Fill(f) => {
                    let color = Animator::resolve(&f.c, frame, |v| Vec4::from_slice(v), Vec4::ONE);
                    let opacity = Animator::resolve(&f.o, frame, |v| *v / 100.0, 1.0);
                    for geom in &active_geometries {
                        let path = self.convert_geometry(geom);
                        processed_nodes.push(RenderNode {
                            transform: Mat3::IDENTITY,
                            alpha: 1.0,
                            blend_mode: BlendMode::Normal,
                            content: NodeContent::Shape(renderer::Shape {
                                geometry: path,
                                fill: Some(Fill {
                                    paint: Paint::Solid(color),
                                    opacity,
                                    rule: FillRule::NonZero,
                                }),
                                stroke: None,
                                trim: trim.clone(),
                            }),
                            masks: vec![], styles: vec![],
                            matte: None,
                            effects: vec![],
                            is_adjustment_layer: false,
                        });
                    }
                }
                data::Shape::GradientFill(gf) => {
                    let start =
                        Animator::resolve(&gf.s, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let end = Animator::resolve(&gf.e, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let opacity = Animator::resolve(&gf.o, frame, |v| *v / 100.0, 1.0);

                    let raw_stops = Animator::resolve(&gf.g.k, frame, |v| v.clone(), Vec::new());
                    let stops = parse_gradient_stops(&raw_stops, gf.g.p as usize);

                    let kind = if gf.t == 1 {
                        GradientKind::Linear
                    } else {
                        GradientKind::Radial
                    };

                    for geom in &active_geometries {
                        let path = self.convert_geometry(geom);
                        processed_nodes.push(RenderNode {
                            transform: Mat3::IDENTITY,
                            alpha: 1.0,
                            blend_mode: BlendMode::Normal,
                            content: NodeContent::Shape(renderer::Shape {
                                geometry: path,
                                fill: Some(Fill {
                                    paint: Paint::Gradient(Gradient {
                                        kind,
                                        stops: stops.clone(),
                                        start,
                                        end,
                                    }),
                                    opacity,
                                    rule: FillRule::NonZero,
                                }),
                                stroke: None,
                                trim: trim.clone(),
                            }),
                            masks: vec![], styles: vec![],
                            matte: None,
                            effects: vec![],
                            is_adjustment_layer: false,
                        });
                    }
                }
                data::Shape::GradientStroke(gs) => {
                    let start =
                        Animator::resolve(&gs.s, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let end = Animator::resolve(&gs.e, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let width = Animator::resolve(&gs.w, frame, |v| *v, 1.0);
                    let opacity = Animator::resolve(&gs.o, frame, |v| *v / 100.0, 1.0);

                    let raw_stops = Animator::resolve(&gs.g.k, frame, |v| v.clone(), Vec::new());
                    let stops = parse_gradient_stops(&raw_stops, gs.g.p as usize);

                    let kind = if gs.t == 1 {
                        GradientKind::Linear
                    } else {
                        GradientKind::Radial
                    };

                    let mut dash = None;
                    if !gs.d.is_empty() {
                        let mut array = Vec::new();
                        let mut offset = 0.0;
                        for prop in &gs.d {
                            match prop.n.as_deref() {
                                Some("o") => {
                                    offset = Animator::resolve(&prop.v, frame, |v| *v, 0.0)
                                }
                                Some("d") | Some("v") | Some("g") => {
                                    array.push(Animator::resolve(&prop.v, frame, |v| *v, 0.0))
                                }
                                _ => {}
                            }
                        }
                        if !array.is_empty() {
                            if array.len() % 2 != 0 {
                                let clone = array.clone();
                                array.extend(clone);
                            }

                            // Normalize Offset
                            let total_length: f32 = array.iter().sum();
                            if total_length > 0.0 {
                                offset = (offset % total_length + total_length) % total_length;
                            } else {
                                offset = 0.0;
                            }

                            dash = Some(DashPattern { array, offset });
                        }
                    }

                    for geom in &active_geometries {
                        let path = self.convert_geometry(geom);
                        processed_nodes.push(RenderNode {
                            transform: Mat3::IDENTITY,
                            alpha: 1.0,
                            blend_mode: BlendMode::Normal,
                            content: NodeContent::Shape(renderer::Shape {
                                geometry: path,
                                fill: None,
                                stroke: Some(Stroke {
                                    paint: Paint::Gradient(Gradient {
                                        kind,
                                        stops: stops.clone(),
                                        start,
                                        end,
                                    }),
                                    width,
                                    opacity,
                                    cap: match gs.lc {
                                        1 => LineCap::Butt,
                                        3 => LineCap::Square,
                                        _ => LineCap::Round,
                                    },
                                    join: match gs.lj {
                                        1 => LineJoin::Miter,
                                        3 => LineJoin::Bevel,
                                        _ => LineJoin::Round,
                                    },
                                    miter_limit: gs.ml,
                                    dash: dash.clone(),
                                }),
                                trim: trim.clone(),
                            }),
                            masks: vec![], styles: vec![],
                            matte: None,
                            effects: vec![],
                            is_adjustment_layer: false,
                        });
                    }
                }
                data::Shape::Stroke(s) => {
                    let color = Animator::resolve(&s.c, frame, |v| Vec4::from_slice(v), Vec4::ONE);
                    let width = Animator::resolve(&s.w, frame, |v| *v, 1.0);
                    let opacity = Animator::resolve(&s.o, frame, |v| *v / 100.0, 1.0);

                    let mut dash = None;
                    if !s.d.is_empty() {
                        let mut array = Vec::new();
                        let mut offset = 0.0;
                        for prop in &s.d {
                            match prop.n.as_deref() {
                                Some("o") => {
                                    offset = Animator::resolve(&prop.v, frame, |v| *v, 0.0);
                                }
                                Some("d") | Some("v") | Some("g") => {
                                    array.push(Animator::resolve(&prop.v, frame, |v| *v, 0.0));
                                }
                                _ => {}
                            }
                        }
                        if !array.is_empty() {
                            if array.len() % 2 != 0 {
                                let clone = array.clone();
                                array.extend(clone);
                            }

                            // Normalize Offset
                            let total_length: f32 = array.iter().sum();
                            if total_length > 0.0 {
                                offset = (offset % total_length + total_length) % total_length;
                            } else {
                                offset = 0.0;
                            }

                            dash = Some(DashPattern { array, offset });
                        }
                    }

                    for geom in &active_geometries {
                        let path = self.convert_geometry(geom);
                        processed_nodes.push(RenderNode {
                            transform: Mat3::IDENTITY,
                            alpha: 1.0,
                            blend_mode: BlendMode::Normal,
                            content: NodeContent::Shape(renderer::Shape {
                                geometry: path,
                                fill: None,
                                stroke: Some(Stroke {
                                    paint: Paint::Solid(color),
                                    width,
                                    opacity,
                                    cap: match s.lc {
                                        1 => LineCap::Butt,
                                        3 => LineCap::Square,
                                        _ => LineCap::Round,
                                    },
                                    join: match s.lj {
                                        1 => LineJoin::Miter,
                                        3 => LineJoin::Bevel,
                                        _ => LineJoin::Round,
                                    },
                                    miter_limit: s.ml,
                                    dash: dash.clone(),
                                }),
                                trim: trim.clone(),
                            }),
                            masks: vec![], styles: vec![],
                            matte: None,
                            effects: vec![],
                            is_adjustment_layer: false,
                        });
                    }
                }
                data::Shape::Group(g) => {
                    let group_nodes = self.process_shapes(&g.it, frame);
                    processed_nodes.push(RenderNode {
                        transform: Mat3::IDENTITY,
                        alpha: 1.0,
                        blend_mode: BlendMode::Normal,
                        content: NodeContent::Group(group_nodes),
                        masks: vec![], styles: vec![],
                        matte: None,
                        effects: vec![],
                        is_adjustment_layer: false,
                    });
                }
                _ => {}
            }
        }
        processed_nodes
    }

    fn apply_repeater(
        &self,
        copies: f32,
        _offset: f32,
        anchor: Vec2,
        pos: Vec2,
        scale: Vec2,
        rot: f32,
        start_op: f32,
        end_op: f32,
        geoms: &mut Vec<PendingGeometry>,
        nodes: &mut Vec<RenderNode>,
    ) {
        let num_copies = copies.round() as usize;
        if num_copies <= 1 {
            return;
        }

        let original_geoms = geoms.clone();
        let original_nodes = nodes.clone();

        let mat_t = Mat3::from_translation(pos);
        let mat_r = Mat3::from_rotation_z(rot.to_radians());
        let mat_s = Mat3::from_scale(scale);
        let mat_a = Mat3::from_translation(-anchor);
        let mat_pre_a = Mat3::from_translation(anchor);

        let pivot_transform = mat_pre_a * mat_r * mat_s * mat_a;
        let step_transform = mat_t * pivot_transform;

        geoms.clear();
        nodes.clear();

        for i in 0..num_copies {
            let t = if num_copies > 1 {
                i as f32 / (num_copies as f32 - 1.0)
            } else {
                0.0
            };
            let op = start_op + (end_op - start_op) * t;

            let mut copy_transform = Mat3::IDENTITY;
            for _ in 0..i {
                copy_transform = copy_transform * step_transform;
            }

            for geom in &original_geoms {
                let mut g = geom.clone();
                g.transform = copy_transform * g.transform;
                geoms.push(g);
            }

            for node in &original_nodes {
                let mut n = node.clone();
                n.transform = copy_transform * n.transform;
                n.alpha *= op;
                nodes.push(n);
            }
        }
    }

    fn apply_modifier_to_active(
        &self,
        active: &mut Vec<PendingGeometry>,
        modifier: &impl GeometryModifier,
    ) {
        for geom in active.iter_mut() {
            // Bake to path if not already
            let mut path = geom.to_path(self);
            modifier.modify(&mut path);

            // Reset transform to identity because it's baked into path
            geom.transform = Mat3::IDENTITY;
            geom.kind = GeometryKind::Path(path);
        }
    }

    fn convert_geometry(&self, geom: &PendingGeometry) -> ShapeGeometry {
        geom.to_shape_geometry(self)
    }

    fn generate_polystar_path(&self, params: &PolystarParams) -> BezPath {
        let mut path = BezPath::new();
        let num_points = params.points.round();
        if num_points < 3.0 {
            return path;
        }

        let is_star = params.kind == 1;

        // Check for roundness (Flower shape)
        // If either outer or inner roundness is significant, we generate smooth cubic beziers.
        // We ignore `corner_radius` in this mode as the shape is intrinsically smooth.
        let has_roundness =
            params.outer_roundness.abs() > 0.01 || (is_star && params.inner_roundness.abs() > 0.01);

        let total_points = if is_star {
            num_points * 2.0
        } else {
            num_points
        } as usize;

        let current_angle = (params.rotation - 90.0).to_radians();
        let angle_step = 2.0 * PI / total_points as f64;

        if has_roundness {
            // "Flower" Generation (Cubic Bezier)
            let mut elements = Vec::with_capacity(total_points);

            for i in 0..total_points {
                let (r, roundness) = if is_star {
                    if i % 2 == 0 {
                        (params.outer_radius, params.outer_roundness)
                    } else {
                        (params.inner_radius, params.inner_roundness)
                    }
                } else {
                    (params.outer_radius, params.outer_roundness)
                };

                let angle = current_angle as f64 + angle_step * i as f64;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let x = params.pos.x as f64 + r as f64 * cos_a;
                let y = params.pos.y as f64 + r as f64 * sin_a;
                let vertex = Point::new(x, y);

                // Tangent Vector T = (-sin, cos) (Perpendicular to Radius (cos, sin))
                let tx = -sin_a;
                let ty = cos_a;
                let tangent = kurbo::Vec2::new(tx, ty);

                // cp_dist = radius * theta * roundness_pct * 0.01
                let cp_d = r as f64 * angle_step * roundness as f64 * 0.01;

                // Incoming CP (from previous segment): Vertex - T * d
                let in_cp = vertex - tangent * cp_d;
                // Outgoing CP (to next segment): Vertex + T * d
                let out_cp = vertex + tangent * cp_d;

                elements.push((vertex, in_cp, out_cp));
            }

            // Build Path
            // Move to first vertex
            if elements.is_empty() {
                return path;
            }
            path.move_to(elements[0].0);

            let len = elements.len();
            for i in 0..len {
                let curr_idx = i;
                let next_idx = (i + 1) % len;

                let curr_out_cp = elements[curr_idx].2;
                let next_in_cp = elements[next_idx].1;
                let next_vertex = elements[next_idx].0;

                path.curve_to(curr_out_cp, next_in_cp, next_vertex);
            }
            path.close_path();
            return path;
        }

        // Legacy (Straight) Generation with Round Corners
        let mut vertices = Vec::with_capacity(total_points);
        for i in 0..total_points {
            let r = if is_star {
                if i % 2 == 0 {
                    params.outer_radius
                } else {
                    params.inner_radius
                }
            } else {
                params.outer_radius
            };
            let angle = current_angle as f64 + angle_step * i as f64;
            let x = params.pos.x as f64 + r as f64 * angle.cos();
            let y = params.pos.y as f64 + r as f64 * angle.sin();
            vertices.push(Point::new(x, y));
        }

        let radius = params.corner_radius as f64;
        if radius <= 0.1 {
            if !vertices.is_empty() {
                path.move_to(vertices[0]);
                for v in vertices.iter().skip(1) {
                    path.line_to(*v);
                }
                path.close_path();
            }
            return path;
        }

        let len = vertices.len();
        for i in 0..len {
            let prev = vertices[(i + len - 1) % len];
            let curr = vertices[i];
            let next = vertices[(i + 1) % len];

            let v1 = prev - curr;
            let v2 = next - curr;
            let len1 = v1.hypot();
            let len2 = v2.hypot();

            if len1 < 0.001 || len2 < 0.001 {
                if i == 0 {
                    path.move_to(curr);
                } else {
                    path.line_to(curr);
                }
                continue;
            }

            let u1 = v1 * (1.0 / len1);
            let u2 = v2 * (1.0 / len2);
            let dot = (u1.x * u2.x + u1.y * u2.y).clamp(-1.0, 1.0);
            let angle = dot.acos();

            let dist = if angle.abs() < 0.001 {
                0.0
            } else {
                radius / (angle / 2.0).tan()
            };

            let max_d = (len1.min(len2)) * 0.5;
            let d = dist.min(max_d);

            let p_start = curr + u1 * d;
            let p_end = curr + u2 * d;

            if i == 0 {
                path.move_to(p_start);
            } else {
                path.line_to(p_start);
            }
            path.quad_to(curr, p_end);
        }
        path.close_path();
        path
    }

    fn convert_path(&self, p: &data::PathShape, frame: f32) -> BezPath {
        let path_data = Animator::resolve(&p.ks, frame, |v| v.clone(), data::BezierPath::default());
        self.convert_bezier_path(&path_data)
    }

    fn convert_bezier_path(&self, path_data: &data::BezierPath) -> BezPath {
        let mut bp = BezPath::new();
        if path_data.v.is_empty() {
            return bp;
        }

        let start = path_data.v[0];
        bp.move_to(Point::new(start[0] as f64, start[1] as f64));

        for i in 0..path_data.v.len() {
            let next_idx = (i + 1) % path_data.v.len();
            if next_idx == 0 && !path_data.c {
                break;
            }

            let p0 = path_data.v[i];
            let p1 = path_data.v[next_idx];

            let o = if i < path_data.o.len() {
                path_data.o[i]
            } else {
                [0.0, 0.0]
            };
            let in_ = if next_idx < path_data.i.len() {
                path_data.i[next_idx]
            } else {
                [0.0, 0.0]
            };

            let cp1 = [p0[0] + o[0], p0[1] + o[1]];
            let cp2 = [p1[0] + in_[0], p1[1] + in_[1]];

            bp.curve_to(
                Point::new(cp1[0] as f64, cp1[1] as f64),
                Point::new(cp2[0] as f64, cp2[1] as f64),
                Point::new(p1[0] as f64, p1[1] as f64),
            );
        }

        if path_data.c {
            bp.close_path();
        }
        bp
    }

    fn process_masks(&self, masks_props: &[data::MaskProperties], frame: f32) -> Vec<Mask> {
        let mut masks = Vec::new();
        for m in masks_props {
            // Check mode
            let mode = match m.mode.as_deref() {
                Some("n") => MaskMode::None,
                Some("a") => MaskMode::Add,
                Some("s") => MaskMode::Subtract,
                Some("i") => MaskMode::Intersect,
                Some("l") => MaskMode::Lighten,
                Some("d") => MaskMode::Darken,
                Some("f") => MaskMode::Difference,
                _ => continue, // Skip unknown
            };

            let path_data =
                Animator::resolve(&m.pt, frame, |v| v.clone(), data::BezierPath::default());
            let geometry = self.convert_bezier_path(&path_data);
            let opacity = Animator::resolve(&m.o, frame, |v| *v / 100.0, 1.0);
            let expansion = Animator::resolve(&m.x, frame, |v| *v, 0.0);
            let inverted = m.inv;

            masks.push(Mask {
                mode,
                geometry,
                opacity,
                expansion,
                inverted,
            });
        }
        masks
    }
}

// Helpers

struct ColorStop {
    t: f32,
    r: f32,
    g: f32,
    b: f32,
}

struct AlphaStop {
    t: f32,
    a: f32,
}

/// Parses gradient stops from the raw array.
///
/// The raw array contains flattened color stops followed by alpha stops.
/// - Color stops: [offset, r, g, b] (4 floats)
/// - Alpha stops: [offset, alpha] (2 floats)
///
/// This function handles cases where color and alpha stops are misaligned by generating
/// a "superset" of stops. It collects all unique offsets from both color and alpha stops,
/// and at each unique offset, it interpolates both the color and alpha values.
/// This ensures correct rendering even if "Smoothness" (non-linear interpolation)
/// introduces extra stops in one channel but not the other.
fn parse_gradient_stops(raw: &[f32], color_count: usize) -> Vec<GradientStop> {
    let mut stops = Vec::new();
    if raw.is_empty() {
        return stops;
    }

    let mut color_stops = Vec::new();
    let mut alpha_stops = Vec::new();

    let color_data_len = color_count * 4;
    // Parse Colors
    for chunk in raw
        .iter()
        .take(color_data_len)
        .collect::<Vec<_>>()
        .chunks(4)
    {
        if chunk.len() == 4 {
            color_stops.push(ColorStop {
                t: *chunk[0],
                r: *chunk[1],
                g: *chunk[2],
                b: *chunk[3],
            });
        }
    }

    // Parse Alphas
    if raw.len() > color_data_len {
        for chunk in raw[color_data_len..].chunks(2) {
            if chunk.len() == 2 {
                alpha_stops.push(AlphaStop {
                    t: chunk[0],
                    a: chunk[1],
                });
            }
        }
    }

    // If no alpha stops, just return colors (with alpha 1.0)
    if alpha_stops.is_empty() {
        for c in color_stops {
            stops.push(GradientStop {
                offset: c.t,
                color: Vec4::new(c.r, c.g, c.b, 1.0),
            });
        }
        return stops;
    }

    // If we have alphas, we need to merge.
    let mut unique_t: Vec<f32> = Vec::new();
    for c in &color_stops {
        unique_t.push(c.t);
    }
    for a in &alpha_stops {
        unique_t.push(a.t);
    }
    unique_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_t.dedup();

    for t in unique_t {
        let (r, g, b) = interpolate_color(&color_stops, t);
        let a = interpolate_alpha(&alpha_stops, t);

        stops.push(GradientStop {
            offset: t,
            color: Vec4::new(r, g, b, a),
        });
    }

    stops
}

fn interpolate_color(stops: &[ColorStop], t: f32) -> (f32, f32, f32) {
    if stops.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    if t <= stops[0].t {
        return (stops[0].r, stops[0].g, stops[0].b);
    }
    if t >= stops.last().unwrap().t {
        let last = stops.last().unwrap();
        return (last.r, last.g, last.b);
    }

    for i in 0..stops.len() - 1 {
        let s1 = &stops[i];
        let s2 = &stops[i + 1];
        if t >= s1.t && t <= s2.t {
            let range = s2.t - s1.t;
            let ratio = if range == 0.0 {
                0.0
            } else {
                (t - s1.t) / range
            };
            return (
                s1.r + (s2.r - s1.r) * ratio,
                s1.g + (s2.g - s1.g) * ratio,
                s1.b + (s2.b - s1.b) * ratio,
            );
        }
    }
    (1.0, 1.0, 1.0)
}

fn interpolate_alpha(stops: &[AlphaStop], t: f32) -> f32 {
    if stops.is_empty() {
        return 1.0;
    }
    if t <= stops[0].t {
        return stops[0].a;
    }
    if t >= stops.last().unwrap().t {
        return stops.last().unwrap().a;
    }

    for i in 0..stops.len() - 1 {
        let s1 = &stops[i];
        let s2 = &stops[i + 1];
        if t >= s1.t && t <= s2.t {
            let range = s2.t - s1.t;
            let ratio = if range == 0.0 {
                0.0
            } else {
                (t - s1.t) / range
            };
            return s1.a + (s2.a - s1.a) * ratio;
        }
    }
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::PathEl;
    use lottie_data::model as data;

    #[test]
    fn test_polystar_generation() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        let star = data::Shape::Polystar(data::PolystarShape {
            nm: None,
            p: data::PositionProperty::Unified(data::Property {
                k: data::Value::Static([50.0, 50.0]),
                ..Default::default()
            }),
            or: data::Property {
                k: data::Value::Static(20.0),
                ..Default::default()
            },
            os: data::Property::default(),
            r: data::Property::default(),
            pt: data::Property {
                k: data::Value::Static(5.0),
                ..Default::default()
            },
            sy: 1, // Star
            ir: Some(data::Property {
                k: data::Value::Static(10.0),
                ..Default::default()
            }),
            is: None,
        });

        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        let shapes = vec![star, fill];
        let nodes = builder.process_shapes(&shapes, 0.0);

        assert_eq!(nodes.len(), 1);
        match &nodes[0].content {
            NodeContent::Shape(s) => {
                // 5 points star = 10 vertices.
                // Elements: MoveTo(1) + LineTo(9) + Close(1) = 11 elements.
                // Wait, loop 0..10.
                // i=0: MoveTo.
                // i=1..9: LineTo (9 times).
                // Total 10 points. 1 Move, 9 Lines.
                // ClosePath adds Close.
                // Total elements: 11.
                if let renderer::ShapeGeometry::Path(path) = &s.geometry {
                    let count = path.elements().len();
                    assert_eq!(count, 11);
                } else {
                    panic!("Expected Path Geometry");
                }
            }
            _ => panic!("Expected Shape"),
        }
    }

    #[test]
    fn test_poly_flower() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        let poly = data::Shape::Polystar(data::PolystarShape {
            nm: None,
            p: data::PositionProperty::Unified(data::Property {
                k: data::Value::Static([50.0, 50.0]),
                ..Default::default()
            }),
            or: data::Property {
                k: data::Value::Static(20.0),
                ..Default::default()
            },
            os: data::Property {
                k: data::Value::Static(50.0), // 50% Roundness
                ..Default::default()
            },
            r: data::Property::default(),
            pt: data::Property {
                k: data::Value::Static(5.0),
                ..Default::default()
            },
            sy: 2, // Polygon
            ir: None,
            is: None,
        });

        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        let shapes = vec![poly, fill];
        let nodes = builder.process_shapes(&shapes, 0.0);

        assert_eq!(nodes.len(), 1);
        match &nodes[0].content {
            NodeContent::Shape(s) => {
                if let renderer::ShapeGeometry::Path(path) = &s.geometry {
                    // Check for curves
                    let mut curve_count = 0;
                    for el in path.elements() {
                        if let PathEl::CurveTo(_, _, _) = el {
                            curve_count += 1;
                        }
                    }
                    // 5 points polygon = 5 vertices.
                    // Should have 5 cubic curves.
                    assert_eq!(curve_count, 5, "Expected 5 cubic curves for 5-point poly flower");
                } else {
                    panic!("Expected Path Geometry");
                }
            }
            _ => panic!("Expected Shape"),
        }
    }

    #[test]
    fn test_polystar_flower() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        let star = data::Shape::Polystar(data::PolystarShape {
            nm: None,
            p: data::PositionProperty::Unified(data::Property {
                k: data::Value::Static([50.0, 50.0]),
                ..Default::default()
            }),
            or: data::Property {
                k: data::Value::Static(20.0),
                ..Default::default()
            },
            os: data::Property {
                k: data::Value::Static(50.0), // 50% Roundness (Flower)
                ..Default::default()
            },
            r: data::Property::default(),
            pt: data::Property {
                k: data::Value::Static(5.0),
                ..Default::default()
            },
            sy: 1, // Star
            ir: Some(data::Property {
                k: data::Value::Static(10.0),
                ..Default::default()
            }),
            is: None,
        });

        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        let shapes = vec![star, fill];
        let nodes = builder.process_shapes(&shapes, 0.0);

        assert_eq!(nodes.len(), 1);
        match &nodes[0].content {
            NodeContent::Shape(s) => {
                if let renderer::ShapeGeometry::Path(path) = &s.geometry {
                    // Check for curves
                    let mut curve_count = 0;
                    for el in path.elements() {
                        if let PathEl::CurveTo(_, _, _) = el {
                            curve_count += 1;
                        }
                    }
                    // 5 points star has 10 vertices.
                    // New logic iterates 10 times, each adds a CurveTo.
                    assert_eq!(
                        curve_count, 10,
                        "Expected 10 cubic curves for 5-point star flower"
                    );
                } else {
                    panic!("Expected Path Geometry");
                }
            }
            _ => panic!("Expected Shape"),
        }
    }

    #[test]
    fn test_repeater_geometry() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property {
                k: data::Value::Static([10.0, 10.0]),
                ..Default::default()
            },
            p: data::Property::default(),
            r: data::Property::default(),
        });

        let repeater = data::Shape::Repeater(data::RepeaterShape {
            nm: None,
            c: data::Property {
                k: data::Value::Static(3.0),
                ..Default::default()
            },
            o: data::Property::default(),
            m: 0,
            tr: data::RepeaterTransform {
                t: data::Transform::default(),
                so: data::Property::default(),
                eo: data::Property::default(),
            },
        });

        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        let shapes = vec![rect, repeater, fill];
        let nodes = builder.process_shapes(&shapes, 0.0);

        // Expect 3 nodes (3 filled rects)
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn test_external_image_asset() {
        let asset = data::Asset {
            id: "image_0".to_string(),
            w: Some(100),
            h: Some(100),
            nm: None,
            layers: None,
            u: None,
            p: None,
            e: None,
        };

        let layer = data::Layer {
            ty: 2,
            ind: Some(1),
            parent: None,
            nm: Some("Image Layer".to_string()),
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: None,
            masks_properties: None,
            tt: None,
            ef: None, sy: None,
            ref_id: Some("image_0".to_string()),
            w: None,
            h: None,
            color: None,
            sw: None,
            sh: None,
            shapes: None,
            t: None,
        };

        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 500,
            h: 500,
            layers: vec![layer],
            assets: vec![asset],
        };

        let mut player = LottiePlayer::new();
        player.load(model);

        // Set external asset
        let dummy_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        player.set_asset("image_0".to_string(), dummy_data.clone());

        // Render
        let tree = player.render_tree();
        let root = tree.root;

        // Verify
        if let NodeContent::Group(children) = root.content {
            assert_eq!(children.len(), 1);
            let child = &children[0];
            if let NodeContent::Image(img) = &child.content {
                assert_eq!(img.width, 100);
                assert_eq!(img.height, 100);
                assert_eq!(img.data, Some(dummy_data));
            } else {
                panic!("Expected Image content, got {:?}", child.content);
            }
        } else {
            panic!("Expected Group content for root, got {:?}", root.content);
        }
    }

    #[test]
    fn test_zigzag_modifier() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        // Rect
        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property {
                k: data::Value::Static([100.0, 100.0]),
                ..Default::default()
            },
            p: data::Property {
                k: data::Value::Static([50.0, 50.0]),
                ..Default::default()
            },
            r: data::Property::default(),
        });

        // ZigZag
        let zigzag = data::Shape::ZigZag(data::ZigZagShape {
            nm: None,
            r: data::Property {
                k: data::Value::Static(10.0),
                ..Default::default()
            }, // 10 Ridges
            s: data::Property {
                k: data::Value::Static(10.0),
                ..Default::default()
            }, // Size 10
            pt: data::Property {
                k: data::Value::Static(1.0),
                ..Default::default()
            }, // Corner
        });

        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        let shapes = vec![rect, zigzag, fill];
        let nodes = builder.process_shapes(&shapes, 0.0);

        assert_eq!(nodes.len(), 1);
        match &nodes[0].content {
            NodeContent::Shape(s) => {
                // Rect usually has 4 sides. 10 Ridges total? Or per segment?
                // My ZigZag implementation treats the whole path.
                // It walks the path.
                // If 10 ridges, we expect ~10 points + start/end.
                // Rect path elements count: Move, Line, Line, Line, Close.
                // ZigZag path elements count: Move, Line... (many).
                if let renderer::ShapeGeometry::Path(path) = &s.geometry {
                    let count = path.elements().len();
                    println!("ZigZag element count: {}", count);
                    assert!(count > 5, "ZigZag should add points. Got {}", count);
                } else {
                    panic!("Expected Path Geometry");
                }
            }
            _ => panic!("Expected Shape"),
        }
    }

    #[test]
    fn test_wiggle_modifier() {
        // Wiggle modifier changes geometry based on time.
        // We test that geometry is different at t=0 vs t=1.
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder1 = SceneGraphBuilder::new(&model, 0.0, &assets, None);
        let builder2 = SceneGraphBuilder::new(&model, 10.0, &assets, None); // different time

        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property {
                k: data::Value::Static([100.0, 100.0]),
                ..Default::default()
            },
            p: data::Property {
                k: data::Value::Static([50.0, 50.0]),
                ..Default::default()
            },
            r: data::Property::default(),
        });

        let wiggle = data::Shape::WigglePath(data::WigglePathShape {
            nm: None,
            s: data::Property {
                k: data::Value::Static(10.0),
                ..Default::default()
            },
            w: data::Property {
                k: data::Value::Static(10.0),
                ..Default::default()
            }, // Size > 0
            r: data::Property::default(),
            sh: data::Property::default(),
        });

        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        let shapes = vec![rect, wiggle, fill];

        let nodes1 = builder1.process_shapes(&shapes, 0.0);
        let nodes2 = builder2.process_shapes(&shapes, 10.0);

        let path1 = match &nodes1[0].content {
            NodeContent::Shape(s) => &s.geometry,
            _ => panic!(),
        };
        let path2 = match &nodes2[0].content {
            NodeContent::Shape(s) => &s.geometry,
            _ => panic!(),
        };

        // Check if point positions differ
        // We can inspect elements
        // This relies on `kurbo` `BezPath` equality which is not derived?
        // Or inspect points.

        let b1 = if let renderer::ShapeGeometry::Path(p) = path1 { p } else { panic!() };
        let b2 = if let renderer::ShapeGeometry::Path(p) = path2 { p } else { panic!() };

        let p1 = match b1.elements()[0] {
            PathEl::MoveTo(p) => p,
            _ => Point::ZERO,
        };
        let p2 = match b2.elements()[0] {
            PathEl::MoveTo(p) => p,
            _ => Point::ZERO,
        };

        // Wiggle might affect MoveTo point if it wiggles vertices.
        // Note: Rect starts at corner.
        assert_ne!(
            p1, p2,
            "Wiggle should displace points differently at different times"
        );
    }

    #[test]
    fn test_time_remap() {
        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property {
                k: data::Value::Static([10.0, 10.0]),
                ..Default::default()
            },
            p: data::Property {
                k: data::Value::Animated(vec![
                    data::Keyframe {
                        t: 0.0,
                        s: Some([0.0, 0.0]),
                        e: Some([100.0, 100.0]),
                        i: None,
                        o: None,
                        to: None,
                        ti: None,
                        h: None,
                    },
                    data::Keyframe {
                        t: 60.0,
                        s: Some([100.0, 100.0]),
                        e: None,
                        i: None,
                        o: None,
                        to: None,
                        ti: None,
                        h: None,
                    },
                ]),
                ..Default::default()
            },
            r: data::Property::default(),
        });

        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        let inner_layer = data::Layer {
            ty: 4,
            ind: Some(1),
            parent: None,
            nm: Some("Inner".to_string()),
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: None,
            masks_properties: None,
            tt: None,
            ef: None, sy: None,
            ref_id: None,
            w: None,
            h: None,
            color: None,
            sw: None,
            sh: None,
            shapes: Some(vec![rect, fill]),
            t: None,
        };

        let asset = data::Asset {
            id: "precomp1".to_string(),
            w: Some(100),
            h: Some(100),
            nm: None,
            layers: Some(vec![inner_layer]),
            u: None,
            p: None,
            e: None,
        };

        let precomp_layer = data::Layer {
            ty: 0,
            ind: Some(1),
            parent: None,
            nm: Some("PreComp Instance".to_string()),
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: Some(data::Property {
                k: data::Value::Static(0.5), // 0.5s = 30 frames
                ..Default::default()
            }),
            masks_properties: None,
            tt: None,
            ef: None, sy: None,
            ref_id: Some("precomp1".to_string()),
            w: Some(100),
            h: Some(100),
            color: None,
            sw: None,
            sh: None,
            shapes: None,
            t: None,
        };

        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![precomp_layer],
            assets: vec![asset],
        };

        let mut player = LottiePlayer::new();
        player.load(model);

        let tree = player.render_tree();
        let root = tree.root;

        if let NodeContent::Group(l1) = root.content {
            assert_eq!(l1.len(), 1);
            let precomp_node = &l1[0];
            if let NodeContent::Group(l2) = &precomp_node.content {
                assert_eq!(l2.len(), 1);
                let inner_node = &l2[0];
                if let NodeContent::Group(l3) = &inner_node.content {
                    assert_eq!(l3.len(), 1);
                    let shape_node = &l3[0];
                    if let NodeContent::Shape(s) = &shape_node.content {
                        if let renderer::ShapeGeometry::Path(path) = &s.geometry {
                            if let PathEl::MoveTo(p) = path.elements()[0] {
                                // Expect 45,45 (50 - 5)
                                assert!((p.x - 45.0).abs() < 0.1, "Expected x=45, got {}", p.x);
                                assert!((p.y - 45.0).abs() < 0.1, "Expected y=45, got {}", p.y);
                            } else {
                                panic!("Expected MoveTo");
                            }
                        } else {
                            panic!("Expected Path Geometry");
                        }
                    } else {
                        panic!("Expected Shape Content");
                    }
                } else {
                    panic!("Expected Group (Shape Layer)");
                }
            } else {
                panic!("Expected Group (PreComp Content)");
            }
        } else {
            panic!("Expected Group (Root)");
        }
    }

    #[test]
    fn test_effects_parsing() {
        use lottie_data::model::{Effect, EffectValue, Property, Value};
        use serde_json::json;
        use std::collections::HashMap;

        // Mock Effect Values
        let black = EffectValue {
            ty: Some(2),
            nm: Some("Black".to_string()),
            ix: None,
            v: Some(Property {
                k: Value::Static(json!([0.0, 0.0, 0.0, 1.0])),
                ..Default::default()
            }),
        };
        let white = EffectValue {
            ty: Some(2),
            nm: Some("White".to_string()),
            ix: None,
            v: Some(Property {
                k: Value::Static(json!([1.0, 1.0, 1.0, 1.0])),
                ..Default::default()
            }),
        };
        let amount = EffectValue {
            ty: Some(0),
            nm: Some("Intensity".to_string()),
            ix: None,
            v: Some(Property {
                k: Value::Static(json!(50.0)),
                ..Default::default()
            }),
        };

        let tint_ef = Effect {
            ty: Some(20),
            nm: Some("Tint".to_string()),
            ix: None,
            en: Some(1),
            ef: Some(vec![black, white, amount]),
        };

        let layer = data::Layer {
            ty: 4,
            ind: Some(1),
            parent: None,
            nm: Some("Layer".to_string()),
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: None,
            masks_properties: None,
            tt: None,
            ref_id: None,
            w: None,
            h: None,
            color: None,
            sw: None,
            sh: None,
            shapes: Some(vec![]),
            t: None,
            sy: None, ef: Some(vec![tint_ef]),
            ..data::Layer {
                ty: 4,
                ind: None,
                parent: None,
                nm: None,
                ip: 0.0,
                op: 0.0,
                st: 0.0,
                ks: data::Transform::default(),
                ao: None,
                tm: None,
                masks_properties: None,
                tt: None,
                ef: None, sy: None,
                ref_id: None,
                w: None,
                h: None,
                color: None,
                sw: None,
                sh: None,
                shapes: None,
                t: None,
            }
        };

        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![layer],
            assets: vec![],
        };

        let external = HashMap::new();
        let mut builder = SceneGraphBuilder::new(&model, 0.0, &external, None);
        let tree = builder.build();

        let root = tree.root;
        if let NodeContent::Group(children) = root.content {
            assert_eq!(children.len(), 1);
            let node = &children[0];
            assert_eq!(node.effects.len(), 1);
            match &node.effects[0] {
                renderer::Effect::Tint {
                    black,
                    white,
                    amount,
                } => {
                    assert_eq!(black.x, 0.0);
                    assert_eq!(white.x, 1.0);
                    assert_eq!(*amount, 0.5);
                }
                _ => panic!("Expected Tint effect"),
            }
        } else {
            panic!("Expected Group");
        }
    }
}

#[test]
fn test_text_animator() {
    let text_doc = data::TextDocument {
        t: "AB".to_string(),
        f: "Arial".to_string(),
        s: 50.0,
        ..Default::default()
    };

    // Animator: Scale 200% for first char (A)
    // Selector 0-50% covers the first of 2 chars (0 to 1 in indices)
    let animator = data::TextAnimatorData {
        nm: Some("Anim".to_string()),
        s: data::TextSelectorData {
            s: Some(data::Property {
                k: data::Value::Static(0.0),
                ..Default::default()
            }),
            e: Some(data::Property {
                k: data::Value::Static(50.0),
                ..Default::default()
            }),
            o: None,
        },
        a: data::TextStyleData {
            s: Some(data::Property {
                k: data::Value::Static([200.0, 200.0]),
                ..Default::default()
            }),
            ..Default::default()
        },
    };

    let text_data = data::TextData {
        d: data::Property {
            k: data::Value::Static(text_doc),
            ..Default::default()
        },
        a: Some(vec![animator]),
    };

    let layer = data::Layer {
        ty: 5,
        ind: Some(1),
        parent: None,
        nm: Some("Text".to_string()),
        ip: 0.0,
        op: 60.0,
        st: 0.0,
        ks: data::Transform::default(),
        ao: None,
        tm: None,
        masks_properties: None,
        tt: None,
        ef: None, sy: None,
        ref_id: None,
        w: None,
        h: None,
        color: None,
        sw: None,
        sh: None,
        shapes: None,
        t: Some(text_data),
        ..data::Layer {
            ty: 5,
            ind: None,
            parent: None,
            nm: None,
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: None,
            masks_properties: None,
            tt: None,
            ef: None, sy: None,
            ref_id: None,
            w: None,
            h: None,
            color: None,
            sw: None,
            sh: None,
            shapes: None,
            t: None,
        }
    };

    let model = LottieJson {
        v: None,
        ip: 0.0,
        op: 60.0,
        fr: 60.0,
        w: 100,
        h: 100,
        layers: vec![layer],
        assets: vec![],
    };

    let mut player = LottiePlayer::new();
    player.load(model);

    let tree = player.render_tree();
    let root = tree.root;

    if let NodeContent::Group(l1) = root.content {
        let text_node = &l1[0];
        if let NodeContent::Text(text) = &text_node.content {
            assert_eq!(text.glyphs.len(), 2);
            let ga = &text.glyphs[0];
            let gb = &text.glyphs[1];

            assert_eq!(ga.character, 'A');
            assert_eq!(gb.character, 'B');

            assert!(
                (ga.scale.x - 2.0).abs() < 0.01,
                "Expected A scale 2.0, got {:?}",
                ga.scale
            );
            assert!(
                (gb.scale.x - 1.0).abs() < 0.01,
                "Expected B scale 1.0, got {:?}",
                gb.scale
            );
        } else {
            panic!("Expected Text Content");
        }
    } else {
        panic!("Expected Group (Root)");
    }
}

    #[test]
    fn test_mask_expansion_and_inversion() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        let mask_prop = data::MaskProperties {
            inv: true,
            mode: Some("a".to_string()),
            pt: data::Property::default(), // Empty path
            o: data::Property::default(),  // Opacity 100%
            nm: None,
            x: data::Property {
                k: data::Value::Static(10.0),
                ..Default::default()
            },
        };

        let masks = builder.process_masks(&[mask_prop], 0.0);

        assert_eq!(masks.len(), 1);
        let mask = &masks[0];
        assert_eq!(mask.inverted, true);
        assert!((mask.expansion - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_embedded_base64_image() {
        // 1x1 Red PNG
        let base64_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        // Expected bytes (decoded)
        let expected_bytes = BASE64_STANDARD.decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==").unwrap();

        let asset = data::Asset {
            id: "img_b64".to_string(),
            w: Some(1),
            h: Some(1),
            nm: None,
            layers: None,
            u: None,
            p: Some(base64_png.to_string()),
            e: None,
        };

        let layer = data::Layer {
            ty: 2, // Image
            ind: Some(1),
            parent: None,
            nm: Some("Base64 Image".to_string()),
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: None,
            masks_properties: None,
            tt: None,
            ef: None, sy: None,
            ref_id: Some("img_b64".to_string()),
            w: None,
            h: None,
            color: None,
            sw: None,
            sh: None,
            shapes: None,
            t: None,
        };

        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![layer],
            assets: vec![asset],
        };

        let external = HashMap::new();
        let mut builder = SceneGraphBuilder::new(&model, 0.0, &external, None);
        let tree = builder.build();

        let root = tree.root;
        if let NodeContent::Group(children) = root.content {
            assert_eq!(children.len(), 1);
            let node = &children[0];
            if let NodeContent::Image(img) = &node.content {
                assert_eq!(img.width, 1);
                assert_eq!(img.height, 1);
                assert!(img.data.is_some());
                assert_eq!(img.data.as_ref().unwrap(), &expected_bytes);
            } else {
                panic!("Expected Image content");
            }
        } else {
            panic!("Expected Group content");
        }
    }

    #[test]
    fn test_merge_paths_logic() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        // Rect
        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property {
                k: data::Value::Static([100.0, 100.0]),
                ..Default::default()
            },
            p: data::Property::default(),
            r: data::Property::default(),
        });

        // Ellipse
        let ellipse = data::Shape::Ellipse(data::EllipseShape {
            nm: None,
            s: data::Property {
                k: data::Value::Static([50.0, 50.0]),
                ..Default::default()
            },
            p: data::Property::default(),
        });

        // Merge Paths (Add)
        let merge = data::Shape::MergePaths(data::MergePathsShape {
            nm: None,
            mm: 2, // Add
        });

        // Fill
        let fill = data::Shape::Fill(data::FillShape {
            nm: None,
            c: data::Property::default(),
            o: data::Property::default(),
            r: None,
        });

        // Order: Rect, Ellipse, Merge, Fill
        let shapes = vec![rect, ellipse, merge, fill];
        let nodes = builder.process_shapes(&shapes, 0.0);

        // Expect: 1 RenderNode (Fill applied to Merged Geometry)
        assert_eq!(nodes.len(), 1);

        match &nodes[0].content {
            NodeContent::Shape(s) => {
                match &s.geometry {
                    ShapeGeometry::Boolean { mode, shapes } => {
                        assert!(matches!(mode, MergeMode::Add));
                        assert_eq!(shapes.len(), 2);
                        // First shape should be Rect path (converted)
                        // Second shape should be Ellipse path (converted)
                        if let ShapeGeometry::Path(p1) = &shapes[0] {
                           // Verify rect elements count (Rect: Move, Line, Line, Line, Close)
                           assert!(p1.elements().len() >= 5);
                        } else {
                            panic!("Expected Path");
                        }
                    },
                    _ => panic!("Expected Boolean Geometry"),
                }
            },
            _ => panic!("Expected Shape Content"),
        }
    }

    #[test]
    fn test_stroke_effect_parsing_rfc009() {
        use lottie_data::model::{Effect, EffectValue, Property, Value};
        use serde_json::json;

        // Mock Effect Values
        let mut ef_vals = Vec::new();
        // Pad 0-2
        ef_vals.push(EffectValue { ty: None, nm: None, ix: None, v: None });
        ef_vals.push(EffectValue { ty: None, nm: None, ix: None, v: None });
        ef_vals.push(EffectValue { ty: None, nm: None, ix: None, v: None });

        // Color (Index 3)
        ef_vals.push(EffectValue {
            ty: Some(2),
            nm: Some("Color".to_string()),
            ix: Some(3),
            v: Some(Property {
                k: Value::Static(json!([1.0, 0.0, 0.0, 1.0])),
                ..Default::default()
            }),
        });

        // Brush Size (Index 4)
        ef_vals.push(EffectValue {
            ty: Some(0),
            nm: Some("Brush Size".to_string()),
            ix: Some(4),
            v: Some(Property {
                k: Value::Static(json!(15.0)),
                ..Default::default()
            }),
        });

        // Pad 5
        ef_vals.push(EffectValue { ty: None, nm: None, ix: None, v: None });
        // Opacity (Index 6)
        ef_vals.push(EffectValue {
            ty: Some(0),
            nm: Some("Opacity".to_string()),
            ix: Some(6),
            v: Some(Property {
                k: Value::Static(json!(50.0)), // 50%
                ..Default::default()
            }),
        });

        // All Masks (Checkbox)
        ef_vals.push(EffectValue {
            ty: Some(4),
            nm: Some("All Masks".to_string()),
            ix: Some(99),
            v: Some(Property {
                k: Value::Static(json!(1)), // 1 = True
                ..Default::default()
            }),
        });

        // Path (Drop Down) - Should be ignored if All Masks is True, but we parse it anyway
        ef_vals.push(EffectValue {
            ty: Some(7),
            nm: Some("Path".to_string()),
            ix: Some(99),
            v: Some(Property {
                k: Value::Static(json!(2)), // Mask Index 2
                ..Default::default()
            }),
        });

        let stroke_ef = Effect {
            ty: Some(22),
            nm: Some("Stroke".to_string()),
            ix: None,
            en: Some(1),
            ef: Some(ef_vals),
        };

        let layer = data::Layer {
            ty: 4,
            ind: Some(1),
            sy: None, ef: Some(vec![stroke_ef]),
            parent: None,
            nm: None,
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: None,
            masks_properties: None,
            tt: None,
            ref_id: None,
            w: None,
            h: None,
            color: None,
            sw: None,
            sh: None,
            shapes: None,
            t: None,
        };

        let model = LottieJson {
            layers: vec![layer],
            w: 100,
            h: 100,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            v: None,
            assets: vec![],
        };

        let external = HashMap::new();
        let mut builder = SceneGraphBuilder::new(&model, 0.0, &external, None);
        let tree = builder.build();

        let root = tree.root;
        let node = match root.content {
            NodeContent::Group(mut c) => c.remove(0),
            _ => panic!("Expected Group"),
        };

        assert_eq!(node.effects.len(), 1);
        if let renderer::Effect::Stroke {
            color,
            width,
            opacity,
            mask_index,
            all_masks,
        } = &node.effects[0]
        {
            assert_eq!(color.x, 1.0); // Red
            assert_eq!(*width, 15.0);
            assert_eq!(*opacity, 0.5);
            assert_eq!(*all_masks, true);
            assert_eq!(*mask_index, Some(2));
        } else {
            panic!("Expected Stroke effect");
        }
    }

    #[test]
    fn test_mask_mode_none() {
        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![],
            assets: vec![],
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets, None);

        let mask_prop = data::MaskProperties {
            inv: false,
            mode: Some("n".to_string()), // None
            pt: data::Property::default(),
            o: data::Property::default(),
            nm: None,
            x: data::Property::default(),
        };

        let masks = builder.process_masks(&[mask_prop], 0.0);

        assert_eq!(masks.len(), 1);
        match masks[0].mode {
            renderer::MaskMode::None => {}, // Pass
            _ => panic!("Expected MaskMode::None"),
        }
    }

#[cfg(test)]
mod text_tests {
    use super::*;
    use lottie_data::model as data;

    struct MockMeasurer;
    impl TextMeasurer for MockMeasurer {
        fn measure(&self, text: &str, _font: &str, _size: f32) -> f32 {
            // Assume 10px width per char
            text.len() as f32 * 10.0
        }
    }

    #[test]
    fn test_box_text_wrapping() {
        let text_doc = data::TextDocument {
            t: "Hello World".to_string(),
            f: "Arial".to_string(),
            s: 10.0,
            sz: Some([60.0, 100.0]), // Width 60. "Hello" (50) fits. " " (10). "World" (50).
                                     // "Hello " (60) fits exactly?
                                     // "Hello" = 50. " " = 10. "World" = 50.
                                     // "Hello " = 60. Fits.
                                     // "Hello World" = 110. > 60. Wrap.
                                     // Expect:
                                     // Line 1: "Hello " (or "Hello")
                                     // Line 2: "World"
            ps: Some([10.0, 10.0]),
            lh: 20.0,
            ..Default::default()
        };

        let text_data = data::TextData {
            d: data::Property {
                k: data::Value::Static(text_doc),
                ..Default::default()
            },
            a: None,
        };

        let layer = data::Layer {
            ty: 5,
            ind: Some(1),
            parent: None,
            nm: Some("Box Text".to_string()),
            ip: 0.0,
            op: 60.0,
            st: 0.0,
            ks: data::Transform::default(),
            ao: None,
            tm: None,
            masks_properties: None,
            tt: None,
            ef: None, sy: None,
            ref_id: None,
            w: None,
            h: None,
            color: None,
            sw: None,
            sh: None,
            shapes: None,
            t: Some(text_data),
        };

        let model = LottieJson {
            v: None,
            ip: 0.0,
            op: 60.0,
            fr: 60.0,
            w: 100,
            h: 100,
            layers: vec![layer],
            assets: vec![],
        };

        let mut player = LottiePlayer::new();
        player.load(model);
        player.set_text_measurer(Box::new(MockMeasurer));

        let tree = player.render_tree();

        if let NodeContent::Group(l1) = tree.root.content {
            let text_node = &l1[0];
            if let NodeContent::Text(text) = &text_node.content {
                // "Hello " is 6 chars. "World" is 5 chars. Total 11 glyphs.
                // Depending on whether we trim or keep spaces. My logic keeps them.
                assert_eq!(text.glyphs.len(), 11);

                // Check positions
                // Line 1: Hello_ (y=10.0)
                // Line 2: World (y=30.0) (10 + 20 lh)

                let g_h = &text.glyphs[0]; // 'H'
                let g_w = &text.glyphs[6]; // 'W' (index 6)

                assert!((g_h.pos.y - 10.0).abs() < 0.1, "H should be at y=10, got {}", g_h.pos.y);
                assert!((g_w.pos.y - 30.0).abs() < 0.1, "W should be at y=30, got {}", g_w.pos.y);

            } else {
                panic!("Expected Text");
            }
        }
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn test_gradient_stops_parsing() {
        // Case 1: Simple Linear Gradient (Red to Blue)
        // Raw: [0.0, 1.0, 0.0, 0.0,  1.0, 0.0, 0.0, 1.0] (2 colors)
        let raw1 = vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let stops1 = parse_gradient_stops(&raw1, 2);
        assert_eq!(stops1.len(), 2);
        assert_eq!(stops1[0].offset, 0.0);
        assert_eq!(stops1[0].color, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(stops1[1].offset, 1.0);
        assert_eq!(stops1[1].color, Vec4::new(0.0, 0.0, 1.0, 1.0));

        // Case 2: Misaligned Alpha Stops (Superset Generation)
        // Color: 0.0 Red, 1.0 Blue
        // Alpha: 0.5 Alpha 0.5 (Single point, should extend?)
        // Standard Lottie usually provides pairs. Let's assume a single pair [0.5, 0.5] acts as a point.
        let raw2 = vec![
            0.0, 1.0, 0.0, 0.0, // Red at 0
            1.0, 0.0, 0.0, 1.0, // Blue at 1
            0.5, 0.5            // Alpha 0.5 at 0.5
        ];
        let stops2 = parse_gradient_stops(&raw2, 2);
        // unique_t: 0.0, 0.5, 1.0.
        // t=0.0: Color=Red, Alpha=0.5 (extended)
        // t=0.5: Color=Mid(Purple), Alpha=0.5
        // t=1.0: Color=Blue, Alpha=0.5
        assert_eq!(stops2.len(), 3);
        assert_eq!(stops2[0].offset, 0.0);
        assert_eq!(stops2[1].offset, 0.5);
        assert_eq!(stops2[2].offset, 1.0);

        assert_eq!(stops2[0].color.w, 0.5);
        assert_eq!(stops2[1].color.w, 0.5);
        assert_eq!(stops2[2].color.w, 0.5);

        // Case 3: Complex Misalignment
        // Color: 0.0 Red, 1.0 Blue
        // Alpha: 0.2 Opaque (1.0), 0.8 Transparent (0.0)
        let raw3 = vec![
            0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0,
            0.2, 1.0,
            0.8, 0.0
        ];
        let stops3 = parse_gradient_stops(&raw3, 2);
        // unique_t: 0.0, 0.2, 0.8, 1.0
        assert_eq!(stops3.len(), 4);

        // Check t=0.2
        let s_02 = &stops3[1];
        assert_eq!(s_02.offset, 0.2);
        assert_eq!(s_02.color.w, 1.0); // Alpha should be 1.0
        // Color should be mix 20%
        assert!((s_02.color.x - 0.8).abs() < 0.01); // Red goes 1->0. At 0.2 it is 0.8.
        assert!((s_02.color.z - 0.2).abs() < 0.01); // Blue goes 0->1. At 0.2 it is 0.2.

         // Check t=0.8
        let s_08 = &stops3[2];
        assert_eq!(s_08.offset, 0.8);
        assert_eq!(s_08.color.w, 0.0); // Alpha should be 0.0
        // Color should be mix 80%
        assert!((s_08.color.x - 0.2).abs() < 0.01);
        assert!((s_08.color.z - 0.8).abs() < 0.01);
    }
}
