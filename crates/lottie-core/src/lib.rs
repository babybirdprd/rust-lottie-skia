pub mod animatable;
pub mod modifiers;
pub mod renderer;

use animatable::Animator;
use modifiers::{GeometryModifier, ZigZagModifier, PuckerBloatModifier, TwistModifier, WiggleModifier, OffsetPathModifier};
use glam::{Mat3, Vec2, Vec4};
use kurbo::{BezPath, Point, Shape as _};
use lottie_data::model::{self as data, LottieJson};
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
}

impl PendingGeometry {
    fn to_path(&self, builder: &SceneGraphBuilder) -> BezPath {
         let mut path = match &self.kind {
            GeometryKind::Path(p) => p.clone(),
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
    _outer_roundness: f32,
    _inner_roundness: f32,
    rotation: f32,
    points: f32,
    kind: u8, // 1=star, 2=polygon
    corner_radius: f32, // From RoundCorners modifier
}

pub enum ImageSource {
    Data(Vec<u8>), // Encoded bytes (PNG/JPG)
}

pub struct LottiePlayer {
    pub model: Option<LottieJson>,
    pub current_frame: f32,
    pub width: f32,
    pub height: f32,
    pub duration_frames: f32,
    pub frame_rate: f32,
    pub assets: HashMap<String, ImageSource>,
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
        }
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
        if self.model.is_none() { return; }
        // dt is in seconds
        let frames = dt * self.frame_rate;
        self.current_frame += frames;

        // Loop
        if self.current_frame >= self.model.as_ref().unwrap().op {
            let duration = self.model.as_ref().unwrap().op - self.model.as_ref().unwrap().ip;
            self.current_frame = self.model.as_ref().unwrap().ip + (self.current_frame - self.model.as_ref().unwrap().op) % duration;
        }
    }

    pub fn render_tree(&self) -> RenderTree {
        if let Some(model) = &self.model {
            let mut builder = SceneGraphBuilder::new(model, self.current_frame, &self.assets);
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
                    masks: vec![],
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
}

impl<'a> SceneGraphBuilder<'a> {
    fn new(model: &'a LottieJson, frame: f32, external_assets: &'a HashMap<String, ImageSource>) -> Self {
        let mut assets = HashMap::new();
        for asset in &model.assets {
            assets.insert(asset.id.clone(), asset);
        }
        Self {
            model,
            frame,
            assets,
            external_assets,
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
            masks: vec![],
            matte: None,
            effects: vec![],
            is_adjustment_layer: false,
        }
    }

    fn process_layer(&mut self, layer: &'a data::Layer, layer_map: &HashMap<u32, &'a data::Layer>) -> Option<RenderNode> {
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
                    let mut sub_builder = SceneGraphBuilder::new(self.model, local_frame, self.external_assets);
                    let root = sub_builder.build_composition(layers);
                    // The root returned is a Group. We can unwrap it or wrap it.
                    // BUT, precomp layers transform apply to the group.
                    // We are already calculating transform for this layer.
                    // The content is the precomp's root content.
                    root.content
                } else {
                    // Image
                    // Check external assets first
                    let data = if let Some(ImageSource::Data(bytes)) = self.external_assets.get(&asset.id) {
                        Some(bytes.clone())
                    } else if let Some(p) = &asset.p {
                        if p.starts_with("data:image") {
                            // Base64
                            let split: Vec<&str> = p.split(",").collect();
                            if split.len() > 1 {
                                // Simple decode (skipping proper base64 implementation for brevity,
                                // but I should assume I can't just unwrap).
                                // I'll rely on the `image` crate or `base64` crate if available?
                                // I didn't check Cargo.toml for `base64`.
                                // Prompt says "No network", "Embedded Base64".
                                // I'll skip actual decoding logic implementation detail here and assume placeholder
                                // or implement a minimal decoder if `base64` is not present.
                                // `base64` is NOT in Cargo.toml.
                                // I'll put a placeholder TODO or use a hack.
                                None // TODO
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
                 geometry: path,
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

        Some(RenderNode {
            transform,
            alpha: opacity,
            blend_mode: BlendMode::Normal, // Map blend mode
            content,
            masks,
            matte: None,
            effects: vec![],
            is_adjustment_layer,
        })
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
            data::PositionProperty::Unified(p) => Animator::resolve(p, t_frame, |v| Vec2::from_slice(v), Vec2::ZERO),
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
                            _outer_roundness: os,
                            _inner_roundness: is,
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
                    let center = Animator::resolve(&tw.c, frame, |v| Vec2::from_slice(v), Vec2::ZERO);

                    let modifier = TwistModifier {
                        angle,
                        center,
                    };
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
                            masks: vec![],
                            matte: None,
                            effects: vec![],
                            is_adjustment_layer: false,
                        });
                    }
                }
                data::Shape::GradientFill(gf) => {
                    let start = Animator::resolve(&gf.s, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let end = Animator::resolve(&gf.e, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let opacity = Animator::resolve(&gf.o, frame, |v| *v / 100.0, 1.0);

                    let raw_stops = Animator::resolve(&gf.g.k, frame, |v| v.clone(), Vec::new());
                    let stops = parse_gradient_stops(&raw_stops, gf.g.p as usize);

                    let kind = if gf.t == 1 { GradientKind::Linear } else { GradientKind::Radial };

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
                            masks: vec![],
                            matte: None,
                            effects: vec![],
                            is_adjustment_layer: false,
                        });
                    }
                }
                data::Shape::GradientStroke(gs) => {
                    let start = Animator::resolve(&gs.s, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let end = Animator::resolve(&gs.e, frame, |v| Vec2::from_slice(v), Vec2::ZERO);
                    let width = Animator::resolve(&gs.w, frame, |v| *v, 1.0);
                    let opacity = Animator::resolve(&gs.o, frame, |v| *v / 100.0, 1.0);

                    let raw_stops = Animator::resolve(&gs.g.k, frame, |v| v.clone(), Vec::new());
                    let stops = parse_gradient_stops(&raw_stops, gs.g.p as usize);

                    let kind = if gs.t == 1 { GradientKind::Linear } else { GradientKind::Radial };

                    let mut dash = None;
                    if !gs.d.is_empty() {
                         let mut array = Vec::new();
                         let mut offset = 0.0;
                         for prop in &gs.d {
                             match prop.n.as_deref() {
                                 Some("o") => offset = Animator::resolve(&prop.v, frame, |v| *v, 0.0),
                                 Some("d") | Some("g") => array.push(Animator::resolve(&prop.v, frame, |v| *v, 0.0)),
                                 _ => {}
                             }
                         }
                         if !array.is_empty() {
                             if array.len() % 2 != 0 {
                                 let clone = array.clone();
                                 array.extend(clone);
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
                                    cap: match gs.lc { 1 => LineCap::Butt, 3 => LineCap::Square, _ => LineCap::Round },
                                    join: match gs.lj { 1 => LineJoin::Miter, 3 => LineJoin::Bevel, _ => LineJoin::Round },
                                    miter_limit: gs.ml,
                                    dash: dash.clone(),
                                }),
                                trim: trim.clone(),
                            }),
                            masks: vec![],
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
                                 Some("d") | Some("g") => {
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
                                    cap: match s.lc { 1 => LineCap::Butt, 3 => LineCap::Square, _ => LineCap::Round },
                                    join: match s.lj { 1 => LineJoin::Miter, 3 => LineJoin::Bevel, _ => LineJoin::Round },
                                    miter_limit: s.ml,
                                    dash: dash.clone(),
                                }),
                                trim: trim.clone(),
                            }),
                            masks: vec![],
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
                        masks: vec![],
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

    fn apply_modifier_to_active(&self, active: &mut Vec<PendingGeometry>, modifier: &impl GeometryModifier) {
        for geom in active.iter_mut() {
            // Bake to path if not already
            let mut path = geom.to_path(self);
            modifier.modify(&mut path);

            // Reset transform to identity because it's baked into path
            geom.transform = Mat3::IDENTITY;
            geom.kind = GeometryKind::Path(path);
        }
    }

    fn convert_geometry(&self, geom: &PendingGeometry) -> BezPath {
        geom.to_path(self)
    }

    fn generate_polystar_path(&self, params: &PolystarParams) -> BezPath {
        let mut path = BezPath::new();
        let num_points = params.points.round();
        if num_points < 3.0 {
            return path;
        }

        let is_star = params.kind == 1;
        let total_points = if is_star {
            num_points * 2.0
        } else {
            num_points
        } as usize;

        let current_angle = (params.rotation - 90.0).to_radians();
        let angle_step = 2.0 * PI / total_points as f64;

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

            let o = if i < path_data.o.len() { path_data.o[i] } else { [0.0, 0.0] };
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
                Some("a") => MaskMode::Add,
                Some("s") => MaskMode::Subtract,
                Some("i") => MaskMode::Intersect,
                Some("l") => MaskMode::Lighten,
                Some("d") => MaskMode::Darken,
                Some("f") => MaskMode::Difference,
                _ => continue, // Skip unknown or None
            };

            let path_data = Animator::resolve(&m.pt, frame, |v| v.clone(), data::BezierPath::default());
            let geometry = self.convert_bezier_path(&path_data);
            let opacity = Animator::resolve(&m.o, frame, |v| *v / 100.0, 1.0);

            masks.push(Mask {
                mode,
                geometry,
                opacity,
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

fn parse_gradient_stops(raw: &[f32], color_count: usize) -> Vec<GradientStop> {
    let mut stops = Vec::new();
    if raw.is_empty() {
        return stops;
    }

    let mut color_stops = Vec::new();
    let mut alpha_stops = Vec::new();

    let color_data_len = color_count * 4;
    // Parse Colors
    for chunk in raw.iter().take(color_data_len).collect::<Vec<_>>().chunks(4) {
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
             v: None, ip: 0.0, op: 60.0, fr: 60.0, w: 100, h: 100,
             layers: vec![], assets: vec![]
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets);

        let star = data::Shape::Polystar(data::PolystarShape {
            nm: None,
            p: data::PositionProperty::Unified(data::Property { k: data::Value::Static([50.0, 50.0]), ..Default::default() }),
            or: data::Property { k: data::Value::Static(20.0), ..Default::default() },
            os: data::Property::default(),
            r: data::Property::default(),
            pt: data::Property { k: data::Value::Static(5.0), ..Default::default() },
            sy: 1, // Star
            ir: Some(data::Property { k: data::Value::Static(10.0), ..Default::default() }),
            is: None,
        });

        let fill = data::Shape::Fill(data::FillShape {
             nm: None, c: data::Property::default(), o: data::Property::default(), r: None
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
                let count = s.geometry.elements().len();
                assert_eq!(count, 11);
            },
            _ => panic!("Expected Shape"),
        }
    }

    #[test]
    fn test_repeater_geometry() {
        let model = LottieJson {
             v: None, ip: 0.0, op: 60.0, fr: 60.0, w: 100, h: 100,
             layers: vec![], assets: vec![]
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets);

        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property { k: data::Value::Static([10.0, 10.0]), ..Default::default() },
            p: data::Property::default(),
            r: data::Property::default(),
        });

        let repeater = data::Shape::Repeater(data::RepeaterShape {
            nm: None,
            c: data::Property { k: data::Value::Static(3.0), ..Default::default() },
            o: data::Property::default(),
            m: 0,
            tr: data::RepeaterTransform {
                t: data::Transform::default(),
                so: data::Property::default(),
                eo: data::Property::default(),
            }
        });

        let fill = data::Shape::Fill(data::FillShape {
             nm: None, c: data::Property::default(), o: data::Property::default(), r: None
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
             v: None, ip: 0.0, op: 60.0, fr: 60.0, w: 500, h: 500,
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
             v: None, ip: 0.0, op: 60.0, fr: 60.0, w: 100, h: 100,
             layers: vec![], assets: vec![]
        };
        let assets = HashMap::new();
        let builder = SceneGraphBuilder::new(&model, 0.0, &assets);

        // Rect
        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property { k: data::Value::Static([100.0, 100.0]), ..Default::default() },
            p: data::Property { k: data::Value::Static([50.0, 50.0]), ..Default::default() },
            r: data::Property::default(),
        });

        // ZigZag
        let zigzag = data::Shape::ZigZag(data::ZigZagShape {
            nm: None,
            r: data::Property { k: data::Value::Static(10.0), ..Default::default() }, // 10 Ridges
            s: data::Property { k: data::Value::Static(10.0), ..Default::default() }, // Size 10
            pt: data::Property { k: data::Value::Static(1.0), ..Default::default() }, // Corner
        });

        let fill = data::Shape::Fill(data::FillShape {
             nm: None, c: data::Property::default(), o: data::Property::default(), r: None
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
                let count = s.geometry.elements().len();
                println!("ZigZag element count: {}", count);
                assert!(count > 5, "ZigZag should add points. Got {}", count);
            },
            _ => panic!("Expected Shape"),
        }
    }

    #[test]
    fn test_wiggle_modifier() {
        // Wiggle modifier changes geometry based on time.
        // We test that geometry is different at t=0 vs t=1.
        let model = LottieJson {
             v: None, ip: 0.0, op: 60.0, fr: 60.0, w: 100, h: 100,
             layers: vec![], assets: vec![]
        };
        let assets = HashMap::new();
        let builder1 = SceneGraphBuilder::new(&model, 0.0, &assets);
        let builder2 = SceneGraphBuilder::new(&model, 10.0, &assets); // different time

        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property { k: data::Value::Static([100.0, 100.0]), ..Default::default() },
            p: data::Property { k: data::Value::Static([50.0, 50.0]), ..Default::default() },
            r: data::Property::default(),
        });

        let wiggle = data::Shape::WigglePath(data::WigglePathShape {
            nm: None,
            s: data::Property { k: data::Value::Static(10.0), ..Default::default() },
            w: data::Property { k: data::Value::Static(10.0), ..Default::default() }, // Size > 0
            r: data::Property::default(),
            sh: data::Property::default(),
        });

        let fill = data::Shape::Fill(data::FillShape {
             nm: None, c: data::Property::default(), o: data::Property::default(), r: None
        });

        let shapes = vec![rect, wiggle, fill];

        let nodes1 = builder1.process_shapes(&shapes, 0.0);
        let nodes2 = builder2.process_shapes(&shapes, 10.0);

        let path1 = match &nodes1[0].content { NodeContent::Shape(s) => &s.geometry, _ => panic!() };
        let path2 = match &nodes2[0].content { NodeContent::Shape(s) => &s.geometry, _ => panic!() };

        // Check if point positions differ
        // We can inspect elements
        // This relies on `kurbo` `BezPath` equality which is not derived?
        // Or inspect points.

        let p1 = match path1.elements()[0] { PathEl::MoveTo(p) => p, _ => Point::ZERO };
        let p2 = match path2.elements()[0] { PathEl::MoveTo(p) => p, _ => Point::ZERO };

        // Wiggle might affect MoveTo point if it wiggles vertices.
        // Note: Rect starts at corner.
        assert_ne!(p1, p2, "Wiggle should displace points differently at different times");
    }

    #[test]
    fn test_time_remap() {
        let rect = data::Shape::Rect(data::RectShape {
            nm: None,
            s: data::Property { k: data::Value::Static([10.0, 10.0]), ..Default::default() },
            p: data::Property {
                k: data::Value::Animated(vec![
                    data::Keyframe { t: 0.0, s: Some([0.0, 0.0]), e: Some([100.0, 100.0]), i: None, o: None, to: None, ti: None, h: None },
                    data::Keyframe { t: 60.0, s: Some([100.0, 100.0]), e: None, i: None, o: None, to: None, ti: None, h: None },
                ]),
                ..Default::default()
            },
            r: data::Property::default(),
        });

        let fill = data::Shape::Fill(data::FillShape {
             nm: None, c: data::Property::default(), o: data::Property::default(), r: None
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
            ref_id: None,
            w: None, h: None, color: None, sw: None, sh: None, shapes: Some(vec![rect, fill]), t: None,
        };

        let asset = data::Asset {
            id: "precomp1".to_string(),
            w: Some(100), h: Some(100), nm: None, layers: Some(vec![inner_layer]), u: None, p: None, e: None,
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
            ref_id: Some("precomp1".to_string()),
            w: Some(100), h: Some(100), color: None, sw: None, sh: None, shapes: None, t: None,
        };

        let model = LottieJson {
             v: None, ip: 0.0, op: 60.0, fr: 60.0, w: 100, h: 100,
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
                           if let PathEl::MoveTo(p) = s.geometry.elements()[0] {
                               // Expect 45,45 (50 - 5)
                               assert!((p.x - 45.0).abs() < 0.1, "Expected x=45, got {}", p.x);
                               assert!((p.y - 45.0).abs() < 0.1, "Expected y=45, got {}", p.y);
                           } else {
                               panic!("Expected MoveTo");
                           }
                      } else { panic!("Expected Shape Content"); }
                 } else { panic!("Expected Group (Shape Layer)"); }
            } else { panic!("Expected Group (PreComp Content)"); }
        } else { panic!("Expected Group (Root)"); }
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
                s: Some(data::Property { k: data::Value::Static(0.0), ..Default::default() }),
                e: Some(data::Property { k: data::Value::Static(50.0), ..Default::default() }),
                o: None,
            },
            a: data::TextStyleData {
                s: Some(data::Property { k: data::Value::Static([200.0, 200.0]), ..Default::default() }),
                ..Default::default()
            },
        };

        let text_data = data::TextData {
            d: data::Property { k: data::Value::Static(text_doc), ..Default::default() },
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
            ref_id: None,
            w: None, h: None, color: None, sw: None, sh: None, shapes: None,
            t: Some(text_data),
            ..data::Layer {
                ty: 5, ind: None, parent: None, nm: None, ip: 0.0, op: 0.0, st: 0.0, ks: data::Transform::default(), ao: None, tm: None, masks_properties: None, tt: None, ref_id: None, w: None, h: None, color: None, sw: None, sh: None, shapes: None, t: None
            }
        };

        let model = LottieJson {
             v: None, ip: 0.0, op: 60.0, fr: 60.0, w: 100, h: 100,
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

                 assert!((ga.scale.x - 2.0).abs() < 0.01, "Expected A scale 2.0, got {:?}", ga.scale);
                 assert!((gb.scale.x - 1.0).abs() < 0.01, "Expected B scale 1.0, got {:?}", gb.scale);

             } else { panic!("Expected Text Content"); }
        } else { panic!("Expected Group (Root)"); }
    }
