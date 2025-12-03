use glam::{Vec2, Vec4};
use lottie_data::model::{BezierPath, Property, TextDocument, Value};

pub trait Interpolatable: Sized + Clone {
    fn lerp(&self, other: &Self, t: f32) -> Self;

    fn lerp_spatial(
        &self,
        other: &Self,
        t: f32,
        _tan_in: Option<&Vec<f32>>,
        _tan_out: Option<&Vec<f32>>,
    ) -> Self {
        self.lerp(other, t)
    }
}

impl Interpolatable for TextDocument {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        if t < 1.0 {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl Interpolatable for BezierPath {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        if t < 1.0 {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl Interpolatable for f32 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Interpolatable for Vec2 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Vec2::lerp(*self, *other, t)
    }

    fn lerp_spatial(
        &self,
        other: &Self,
        t: f32,
        tan_in: Option<&Vec<f32>>,
        tan_out: Option<&Vec<f32>>,
    ) -> Self {
        let p0 = *self;
        let p3 = *other;

        let t_out = if let Some(to) = tan_out {
            if to.len() >= 2 {
                Vec2::new(to[0], to[1])
            } else {
                Vec2::ZERO
            }
        } else {
            Vec2::ZERO
        };

        let t_in = if let Some(ti) = tan_in {
            if ti.len() >= 2 {
                Vec2::new(ti[0], ti[1])
            } else {
                Vec2::ZERO
            }
        } else {
            Vec2::ZERO
        };

        let p1 = p0 + t_out;
        let p2 = p3 + t_in;

        let one_minus_t = 1.0 - t;
        let one_minus_t_sq = one_minus_t * one_minus_t;
        let one_minus_t_cub = one_minus_t_sq * one_minus_t;

        let t_sq = t * t;
        let t_cub = t_sq * t;

        p0 * one_minus_t_cub
            + p1 * 3.0 * one_minus_t_sq * t
            + p2 * 3.0 * one_minus_t * t_sq
            + p3 * t_cub
    }
}

impl Interpolatable for Vec4 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Vec4::lerp(*self, *other, t)
    }
}

// For gradient colors (Vec<f32>)
impl Interpolatable for Vec<f32> {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a + (b - a) * t)
            .collect()
    }
}

// Cubic Bezier Easing
pub fn solve_cubic_bezier(p1: Vec2, p2: Vec2, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Newton-Raphson
    let mut t = x;
    for _ in 0..8 {
        let one_minus_t = 1.0 - t;
        let x_est = 3.0 * one_minus_t * one_minus_t * t * p1.x
            + 3.0 * one_minus_t * t * t * p2.x
            + t * t * t;

        let err = x_est - x;
        if err.abs() < 1e-4 {
            break;
        }

        let dx_dt = 3.0 * one_minus_t * one_minus_t * p1.x
            + 6.0 * one_minus_t * t * (p2.x - p1.x)
            + 3.0 * t * t * (1.0 - p2.x);

        if dx_dt.abs() < 1e-6 {
            break;
        }
        t -= err / dx_dt;
    }

    let one_minus_t = 1.0 - t;
    3.0 * one_minus_t * one_minus_t * t * p1.y + 3.0 * one_minus_t * t * t * p2.y + t * t * t
}

pub struct Animator;

impl Animator {
    pub fn resolve<T, U>(
        prop: &Property<T>,
        frame: f32,
        converter: impl Fn(&T) -> U,
        default: U,
    ) -> U
    where
        U: Interpolatable,
    {
        match &prop.k {
            Value::Default => default,
            Value::Static(v) => converter(v),
            Value::Animated(keyframes) => {
                if keyframes.is_empty() {
                    return default;
                }

                // If before first keyframe
                if frame < keyframes[0].t {
                    if let Some(s) = &keyframes[0].s {
                        return converter(s);
                    }
                    return default;
                }

                let len = keyframes.len();
                // If after last keyframe
                if frame >= keyframes[len - 1].t {
                    let last = &keyframes[len - 1];
                    // Use end value if present, else start value
                    if let Some(e) = &last.e {
                        return converter(e);
                    }
                    if let Some(s) = &last.s {
                        return converter(s);
                    }
                    return default;
                }

                // Find segment
                for i in 0..len - 1 {
                    let kf_start = &keyframes[i];
                    let kf_end = &keyframes[i + 1];

                    if frame >= kf_start.t && frame < kf_end.t {
                        let start_val = kf_start
                            .s
                            .as_ref()
                            .map(|v| converter(v))
                            .unwrap_or(default.clone());
                        // End value: explicit 'e', or next keyframe's 's'?
                        // Lottie usually has 's' on start keyframe and 's' on next keyframe acting as end.
                        // Sometimes 'e' is on start keyframe.
                        let end_val = kf_start
                            .e
                            .as_ref()
                            .map(|v| converter(v))
                            .or_else(|| kf_end.s.as_ref().map(|v| converter(v)))
                            .unwrap_or(start_val.clone());

                        let duration = kf_end.t - kf_start.t;
                        if duration <= 0.0 {
                            return start_val;
                        }

                        let mut local_t = (frame - kf_start.t) / duration;

                        // Easing
                        // Out tangent of start, In tangent of end
                        let p1 = if let Some(o) = kf_start.o {
                            Vec2::new(o[0], o[1])
                        } else {
                            Vec2::new(0.0, 0.0)
                        };
                        let p2 = if let Some(i) = kf_end.i {
                            Vec2::new(i[0], i[1])
                        } else {
                            Vec2::new(1.0, 1.0)
                        };

                        // If Hold keyframe
                        if let Some(h) = kf_start.h {
                            if h == 1 {
                                return start_val;
                            }
                        }

                        local_t = solve_cubic_bezier(p1, p2, local_t);

                        return start_val.lerp_spatial(
                            &end_val,
                            local_t,
                            kf_end.ti.as_ref(),
                            kf_start.to.as_ref(),
                        );
                    }
                }

                default
            }
        }
    }
}
