use glam::{Mat3, Vec4};
use kurbo::{BezPath, PathEl};
use lottie_core::{
    BlendMode as CoreBlendMode, ColorChannel as CoreColorChannel, Effect, FillRule as CoreFillRule,
    GradientKind, Justification, LineCap as CoreLineCap, LineJoin as CoreLineJoin, MatteMode,
    NodeContent, Paint as CorePaint, RenderNode, RenderTree,
};
use skia_safe::color_filters::Clamp;
use skia_safe::{
    canvas::SaveLayerRec, color_filters, gradient_shader, image_filters, BlendMode, Canvas, Color,
    Color4f, ColorChannel, Data, Font, FontMgr, FontStyle, Image as SkImage, Matrix, Paint,
    PaintStyle, Path, PathEffect, PathFillType, Point, Rect, TextBlob, TileMode,
};

pub struct SkiaRenderer;

impl SkiaRenderer {
    /// Draws the computed frame onto the provided canvas.
    ///
    /// # Arguments
    /// * `canvas` - The active Skia canvas.
    /// * `tree` - The computed state of the animation for the current frame.
    /// * `dest_rect` - The bounds to draw into (handles scaling).
    /// * `alpha` - Global opacity multiplier (0.0 to 1.0).
    ///
    /// Note: `canvas` is immutable reference because `skia-safe` uses interior mutability
    /// for its `Canvas` wrapper, but conceptually it is mutated.
    pub fn draw(canvas: &Canvas, tree: &RenderTree, dest_rect: Rect, alpha: f32) {
        canvas.save();

        // 4.1 Coordinate System & Transforms
        // Global Transform: Map the Lottie composition dimensions (w x h) to the target dest_rect.
        let scale_x = sanitize(dest_rect.width() / tree.width);
        let scale_y = sanitize(dest_rect.height() / tree.height);

        // Sanitize rect
        let left = sanitize(dest_rect.left);
        let top = sanitize(dest_rect.top);

        let mut global_matrix = Matrix::translate((left, top));
        global_matrix.pre_scale((scale_x, scale_y), None);

        canvas.concat(&global_matrix);

        // Draw Root Node
        draw_node(canvas, &tree.root, alpha);

        canvas.restore();
    }
}

fn draw_node(canvas: &Canvas, node: &RenderNode, parent_alpha: f32) {
    canvas.save();

    // Transform
    let matrix = glam_to_skia_matrix(node.transform);
    canvas.concat(&matrix);

    // Determine opacity
    let node_alpha = sanitize(node.alpha * parent_alpha);

    // Check if we need a layer
    let has_matte = node.matte.is_some();
    let has_effects = !node.effects.is_empty();
    let non_normal_blend = !matches!(node.blend_mode, CoreBlendMode::Normal);
    let is_group = matches!(node.content, NodeContent::Group(_));

    let atomic_opacity_needed = is_group && node_alpha < 1.0;

    let need_layer = has_matte || has_effects || non_normal_blend || atomic_opacity_needed;

    if need_layer {
        let mut paint = Paint::default();
        paint.set_alpha_f(node_alpha);
        paint.set_blend_mode(convert_blend_mode(node.blend_mode));

        // 4.5 Effects
        if has_effects {
            let mut filter: Option<skia_safe::ImageFilter> = None;
            for effect in &node.effects {
                let next_filter = match effect {
                    Effect::GaussianBlur { sigma } => {
                        let s = sanitize(*sigma);
                        image_filters::blur((s, s), TileMode::Decal, filter.clone(), None)
                    }
                    Effect::DropShadow {
                        color,
                        offset,
                        blur,
                    } => {
                        let c = glam_to_skia_color_legacy(*color);
                        let dx = sanitize(offset.x);
                        let dy = sanitize(offset.y);
                        let b = sanitize(*blur);
                        image_filters::drop_shadow((dx, dy), (b, b), c, None, filter, None)
                    }
                    Effect::ColorMatrix { matrix } => image_filters::color_filter(
                        color_filters::matrix_row_major(matrix, Clamp::Yes),
                        filter,
                        None,
                    ),
                    Effect::DisplacementMap {
                        scale,
                        x_channel,
                        y_channel,
                    } => image_filters::displacement_map(
                        (
                            convert_color_channel(*x_channel),
                            convert_color_channel(*y_channel),
                        ),
                        sanitize(*scale),
                        None,
                        filter,
                        None,
                    ),
                };
                filter = next_filter;
            }
            paint.set_image_filter(filter);
        }

        if has_matte {
            // 4.4.2 Track Mattes
            canvas.save_layer(&SaveLayerRec::default().paint(&paint));

            // Draw Content
            draw_content(canvas, &node.content, 1.0);

            // Matte Logic
            if let Some(matte) = &node.matte {
                let mut matte_paint = Paint::default();
                let blend = match matte.mode {
                    MatteMode::Alpha => BlendMode::DstIn,
                    MatteMode::AlphaInverted => BlendMode::DstOut,
                    MatteMode::Luma => {
                        // Luma to Alpha: A = 0.2126 R + 0.7152 G + 0.0722 B
                        #[rustfmt::skip]
                         let matrix = [
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.2126, 0.7152, 0.0722, 0.0, 0.0,
                         ];
                        matte_paint
                            .set_color_filter(color_filters::matrix_row_major(&matrix, Clamp::Yes));
                        BlendMode::DstIn
                    }
                    MatteMode::LumaInverted => {
                        // Luma Inverted to Alpha: A = 1 - (0.2126 R + ...)
                        // A = -0.2126 R - ... + 1.0
                        #[rustfmt::skip]
                         let matrix = [
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             -0.2126, -0.7152, -0.0722, 0.0, 1.0,
                         ];
                        matte_paint
                            .set_color_filter(color_filters::matrix_row_major(&matrix, Clamp::Yes));
                        BlendMode::DstIn
                    }
                };
                matte_paint.set_blend_mode(blend);

                canvas.save_layer(&SaveLayerRec::default().paint(&matte_paint));
                draw_node(canvas, &matte.node, 1.0);
                canvas.restore();
            }

            canvas.restore();
        } else {
            canvas.save_layer(&SaveLayerRec::default().paint(&paint));
            draw_content(canvas, &node.content, 1.0);
            canvas.restore();
        }
    } else {
        draw_content(canvas, &node.content, node_alpha);
    }

    canvas.restore();
}

fn draw_content(canvas: &Canvas, content: &NodeContent, alpha: f32) {
    match content {
        NodeContent::Group(children) => {
            for child in children {
                draw_node(canvas, child, alpha);
            }
        }
        NodeContent::Shape(shape) => {
            let mut path = kurbo_to_skia_path(&shape.geometry);

            // 4.2 Winding Rules
            if let Some(fill) = &shape.fill {
                path.set_fill_type(convert_fill_rule(fill.rule));

                let mut paint = Paint::default();
                paint.set_style(PaintStyle::Fill);
                paint.set_alpha_f(sanitize(fill.opacity * alpha));
                setup_paint_shader(&mut paint, &fill.paint);

                canvas.draw_path(&path, &paint);
            }

            if let Some(stroke) = &shape.stroke {
                let mut paint = Paint::default();
                paint.set_style(PaintStyle::Stroke);
                paint.set_alpha_f(sanitize(stroke.opacity * alpha));
                paint.set_stroke_width(sanitize(stroke.width));
                paint.set_stroke_cap(convert_cap(stroke.cap));
                paint.set_stroke_join(convert_join(stroke.join));

                if let Some(miter) = stroke.miter_limit {
                    paint.set_stroke_miter(sanitize(miter));
                }

                // 4.3.3 Dashed Lines
                if let Some(dash) = &stroke.dash {
                    let mut array: Vec<f32> = dash.array.iter().map(|&v| sanitize(v)).collect();
                    if array.len() % 2 != 0 {
                        let clone = array.clone();
                        array.extend(clone);
                    }
                    if let Some(effect) = PathEffect::dash(&array, sanitize(dash.offset)) {
                        paint.set_path_effect(effect);
                    }
                }

                setup_paint_shader(&mut paint, &stroke.paint);
                canvas.draw_path(&path, &paint);
            }
        }
        NodeContent::Text(text) => {
            let font_mgr = FontMgr::new();
            // Try to match family, fallback to empty string (default)
            let typeface = font_mgr
                .match_family_style(&text.font_family, FontStyle::normal())
                .or_else(|| font_mgr.match_family_style("", FontStyle::normal()));

            if let Some(typeface) = typeface {
                let font = Font::new(typeface, Some(text.size));

                // Handle multi-line text
                let lines = text.content.split('\n');
                let mut current_y = 0.0; // Start at baseline of first line? Or top? Lottie usually implies baseline.

                for line in lines {
                    if let Some(blob) = TextBlob::from_str(line, &font) {
                        let width = font.measure_str(line, None).0;
                        let x = match text.justify {
                            Justification::Left => 0.0,
                            Justification::Center => -width / 2.0,
                            Justification::Right => -width,
                        };

                        // Draw Fill
                        if let Some(fill) = &text.fill {
                            let mut paint = Paint::default();
                            paint.set_style(PaintStyle::Fill);
                            paint.set_alpha_f(sanitize(fill.opacity * alpha));
                            setup_paint_shader(&mut paint, &fill.paint);
                            canvas.draw_text_blob(&blob, (x, current_y), &paint);
                        }

                        // Draw Stroke
                        if let Some(stroke) = &text.stroke {
                            let mut paint = Paint::default();
                            paint.set_style(PaintStyle::Stroke);
                            paint.set_alpha_f(sanitize(stroke.opacity * alpha));
                            paint.set_stroke_width(sanitize(stroke.width));
                            setup_paint_shader(&mut paint, &stroke.paint);
                            canvas.draw_text_blob(&blob, (x, current_y), &paint);
                        }
                    }
                    current_y += text.line_height;
                }
            }
        }
        NodeContent::Image(image) => {
            let mut drawn = false;
            if let Some(data) = &image.data {
                // Attempt to decode image
                let sk_data = Data::new_copy(data);
                if let Some(img) = SkImage::from_encoded(sk_data) {
                    let mut paint = Paint::default();
                    paint.set_alpha_f(alpha);

                    let src = Rect::from_wh(img.width() as f32, img.height() as f32);
                    let dst = Rect::from_wh(image.width as f32, image.height as f32);
                    // Use Strict constraint
                    canvas.draw_image_rect(
                        img,
                        Some((&src, skia_safe::canvas::SrcRectConstraint::Strict)),
                        dst,
                        &paint,
                    );
                    drawn = true;
                }
            }

            if !drawn {
                // Placeholder: Magenta Rectangle
                let mut paint = Paint::default();
                paint.set_color(Color::MAGENTA);
                paint.set_style(PaintStyle::Fill);
                canvas.draw_rect(
                    Rect::from_wh(image.width as f32, image.height as f32),
                    &paint,
                );
            }
        }
    }
}

fn setup_paint_shader(paint: &mut Paint, core_paint: &CorePaint) {
    match core_paint {
        CorePaint::Solid(color) => {
            let c = glam_to_skia_color4f(*color);
            paint.set_color4f(c, None);
        }
        CorePaint::Gradient(grad) => {
            let colors: Vec<Color> = grad
                .stops
                .iter()
                .map(|s| glam_to_skia_color_legacy(s.color))
                .collect();
            let pos: Vec<f32> = grad.stops.iter().map(|s| sanitize(s.offset)).collect();
            let pt1 = Point::new(sanitize(grad.start.x), sanitize(grad.start.y));
            let pt2 = Point::new(sanitize(grad.end.x), sanitize(grad.end.y));

            let shader = match grad.kind {
                GradientKind::Linear => gradient_shader::linear(
                    (pt1, pt2),
                    colors.as_slice(),
                    Some(pos.as_slice()),
                    TileMode::Clamp,
                    None,
                    None,
                ),
                GradientKind::Radial => {
                    let radius = Point::distance(pt1, pt2);
                    gradient_shader::radial(
                        pt1,
                        radius,
                        colors.as_slice(),
                        Some(pos.as_slice()),
                        TileMode::Clamp,
                        None,
                        None,
                    )
                }
            };
            paint.set_shader(shader);
        }
    }
}

// Helpers

fn sanitize(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn glam_to_skia_matrix(m: Mat3) -> Matrix {
    if !m.is_finite() {
        return Matrix::translate((0.0, 0.0));
    }
    let c0 = m.col(0);
    let c1 = m.col(1);
    let c2 = m.col(2);

    Matrix::new_all(
        c0.x, c1.x, c2.x, // ScaleX, SkewX, TransX
        c0.y, c1.y, c2.y, // SkewY, ScaleY, TransY
        c0.z, c1.z, c2.z, // Persp0, Persp1, Persp2
    )
}

fn glam_to_skia_color4f(v: Vec4) -> Color4f {
    // Sanitize color components? Skia might handle them, but safety check:
    Color4f::new(sanitize(v.x), sanitize(v.y), sanitize(v.z), sanitize(v.w))
}

fn glam_to_skia_color_legacy(v: Vec4) -> Color {
    let c = glam_to_skia_color4f(v);
    c.to_color()
}

fn kurbo_to_skia_path(bez_path: &BezPath) -> Path {
    let mut path = Path::new();
    for el in bez_path.elements() {
        match el {
            PathEl::MoveTo(p) => {
                path.move_to((sanitize(p.x as f32), sanitize(p.y as f32)));
            }
            PathEl::LineTo(p) => {
                path.line_to((sanitize(p.x as f32), sanitize(p.y as f32)));
            }
            PathEl::QuadTo(p1, p2) => {
                path.quad_to(
                    (sanitize(p1.x as f32), sanitize(p1.y as f32)),
                    (sanitize(p2.x as f32), sanitize(p2.y as f32)),
                );
            }
            PathEl::CurveTo(p1, p2, p3) => {
                path.cubic_to(
                    (sanitize(p1.x as f32), sanitize(p1.y as f32)),
                    (sanitize(p2.x as f32), sanitize(p2.y as f32)),
                    (sanitize(p3.x as f32), sanitize(p3.y as f32)),
                );
            }
            PathEl::ClosePath => {
                path.close();
            }
        }
    }
    path
}

fn convert_blend_mode(mode: CoreBlendMode) -> BlendMode {
    match mode {
        CoreBlendMode::Normal => BlendMode::SrcOver,
        CoreBlendMode::Multiply => BlendMode::Multiply,
        CoreBlendMode::Screen => BlendMode::Screen,
        CoreBlendMode::Overlay => BlendMode::Overlay,
        CoreBlendMode::Darken => BlendMode::Darken,
        CoreBlendMode::Lighten => BlendMode::Lighten,
        CoreBlendMode::ColorDodge => BlendMode::ColorDodge,
        CoreBlendMode::ColorBurn => BlendMode::ColorBurn,
        CoreBlendMode::HardLight => BlendMode::HardLight,
        CoreBlendMode::SoftLight => BlendMode::SoftLight,
        CoreBlendMode::Difference => BlendMode::Difference,
        CoreBlendMode::Exclusion => BlendMode::Exclusion,
        CoreBlendMode::Hue => BlendMode::Hue,
        CoreBlendMode::Saturation => BlendMode::Saturation,
        CoreBlendMode::Color => BlendMode::Color,
        CoreBlendMode::Luminosity => BlendMode::Luminosity,
    }
}

fn convert_fill_rule(rule: CoreFillRule) -> PathFillType {
    match rule {
        CoreFillRule::NonZero => PathFillType::Winding,
        CoreFillRule::EvenOdd => PathFillType::EvenOdd,
    }
}

fn convert_cap(cap: CoreLineCap) -> skia_safe::PaintCap {
    match cap {
        CoreLineCap::Butt => skia_safe::PaintCap::Butt,
        CoreLineCap::Round => skia_safe::PaintCap::Round,
        CoreLineCap::Square => skia_safe::PaintCap::Square,
    }
}

fn convert_join(join: CoreLineJoin) -> skia_safe::PaintJoin {
    match join {
        CoreLineJoin::Miter => skia_safe::PaintJoin::Miter,
        CoreLineJoin::Round => skia_safe::PaintJoin::Round,
        CoreLineJoin::Bevel => skia_safe::PaintJoin::Bevel,
    }
}

fn convert_color_channel(c: CoreColorChannel) -> ColorChannel {
    match c {
        CoreColorChannel::R => ColorChannel::R,
        CoreColorChannel::G => ColorChannel::G,
        CoreColorChannel::B => ColorChannel::B,
        CoreColorChannel::A => ColorChannel::A,
    }
}
