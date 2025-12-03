# Lottie Parity Checklist

This document tracks the feature parity of `rust-lottie-skia` against the standard Lottie specification.

**Legend:**
- `[x]` Implemented in `rust-lottie-skia`
- `[ ]` Not implemented
- `-> engine`: Feature should be handled by `director-engine`.

---

## 1. Shapes & Geometry (95%)
Basic vector shapes and path construction.

- [x] **Rectangle**
- [x] **Ellipse**
- [x] **Polystar** (Star & Polygon)
- [x] **Path** (Bezier)
- [x] **Groups** (Nested shapes)
- [x] **Transform** (Anchor, Position, Scale, Rotation, Opacity, Skew)

## 2. Fills & Strokes (90%)
Styling of vector shapes.

- [x] **Solid Fill**
- [x] **Gradient Fill** (Linear & Radial)
- [x] **Solid Stroke**
- [x] **Gradient Stroke** (Linear & Radial)
- [x] **Stroke Width**
- [x] **Line Cap** (Butt, Round, Square)
- [x] **Line Join** (Miter, Round, Bevel)
- [x] **Miter Limit**
- [x] **Dashed Lines**
- [ ] **Gradient Interpolation** (Smoothness control)

## 3. Shape Modifiers (90%)
Procedural modifications to geometry.

- [x] **Trim Paths** (Start, End, Offset)
- [x] **Repeater** (Copies, Offset, Transform, Opacity)
- [x] **Round Corners**
- [x] **Pucker & Bloat**
- [x] **Twist**
- [x] **Zig Zag**
- [x] **Wiggle Paths** (Deterministic noise based)
- [x] **Offset Paths** (Data model supported, logic is pass-through MVP)
- [ ] **Merge Paths** (Boolean operations: Union, Intersect, etc. - **Critical for icons**)

## 4. Layers & Composition (90%)
Layer types and composition structure.

- [x] **Shape Layer**
- [x] **Solid Layer**
- [x] **Null Layer**
- [x] **Parenting** (Transform hierarchy)
- [x] **Pre-Composition** (Nested compositions)
- [x] **Image Layer** (External Files & Embedded Base64)
- [x] **Time Remapping** (Non-linear playback)
- [x] **Adjustment Layer** (Applies effects to background)
- [ ] **3D Layer** (Camera, Z-axis - `lottie-core` is strictly 2D)

## 5. Masks & Mattes (100%)
Visibility and compositing.

- [x] **Mask Mode: Add**
- [x] **Mask Mode: Subtract**
- [x] **Mask Mode: Intersect**
- [x] **Mask Mode: Lighten/Darken/Difference** (Handled via Skia BlendModes)
- [x] **Mask Opacity**
- [x] **Mask Expansion** (Expansion supported, Contraction ignored)
- [x] **Mask Inversion**
- [x] **Alpha Matte**
- [x] **Alpha Inverted Matte**
- [x] **Luma Matte**
- [x] **Luma Inverted Matte**

## 6. Text (80%)
Text rendering and typography.

- [x] **Basic Text Rendering**
- [x] **Font Support** (`-> engine` for system fonts)
- [x] **Fill & Stroke**
- [x] **Justification**
- [x] **Line Height**
- [x] **Tracking**
- [x] **Text Animators** (Range Selectors: Position, Scale, Rotation, Opacity, Tracking, Color)
- [ ] **Text on Path**
- [ ] **Paragraph Text** (Text wrapping inside a bounding box)

## 7. Effects (40%)
Post-processing defined *inside* the Lottie JSON.

- [x] **Drop Shadow**
- [x] **Gaussian Blur**
- [x] **Color Matrix**
- [x] **Displacement Map**
- [x] **Tint**
- [x] **Tritone**
- [x] **Fill (Effect)**
- [x] **Stroke (Effect)** (Data model only)
- [x] **Levels** (Data model only)
- [ ] **Turbulent Displace**

## 8. Animation & Interpolation (90%)
Keyframes and timing.

- [x] **Linear Interpolation**
- [x] **Bezier Interpolation** (Temporal Ease In/Out)
- [x] **Hold Interpolation**
- [x] **Spatial Bezier** (Curved motion paths)
- [ ] **Expressions** (`-> engine` via Rhai, though syntax differs)

## 9. Audio (0%)
Sound playback defined in Lottie.

- [ ] **Audio Layers** (Data parsing) `-> engine`

## 10. Optimization & Misc
- [ ] **Render Caching** `-> engine`
- [ ] **Marker Support**