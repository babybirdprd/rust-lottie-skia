# Lottie Parity Checklist

This document tracks the feature parity of `rust-lottie-skia` against the standard Lottie specification (supported by Skia/Skottie).

**Overall Completion: ~45%**

## 1. Shapes & Geometry (85%)
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
- [ ] **Gradient Interpolation** (Smoothness control / Highlight Angle)

## 3. Shape Modifiers (40%)
Procedural modifications to shapes.

- [x] **Trim Paths** (Start, End, Offset)
- [x] **Repeater** (Copies, Offset, Transform, Opacity)
- [x] **Round Corners**
- [ ] **Pucker & Bloat**
- [ ] **Twist**
- [ ] **Zig Zag**
- [ ] **Offset Paths**
- [ ] **Wiggle Paths**
- [ ] **Merge Paths**

## 4. Layers & Composition (60%)
Layer types and composition structure.

- [x] **Shape Layer**
- [x] **Solid Layer**
- [x] **Image Layer (External Files)**
- [ ] **Image Layer (Embedded Base64)** (Stubbed in code, implementation missing)
- [x] **Null Layer**
- [x] **Pre-Composition** (Nested compositions)
- [x] **Parenting** (Transform hierarchy)
- [ ] **Time Remapping** (Property `tm` is currently ignored)
- [ ] **3D Layer** (Camera, Z-axis)
- [ ] **Adjustment Layer**

## 5. Masks & Mattes (80%)
Visibility and compositing.

- [x] **Mask Mode: Add**
- [x] **Mask Mode: Subtract**
- [x] **Mask Mode: Intersect**
- [ ] **Mask Mode: Lighten/Darken/Difference**
- [x] **Mask Opacity**
- [ ] **Mask Expansion** (Property `x` missing from data model)
- [x] **Alpha Matte**
- [x] **Alpha Inverted Matte**
- [x] **Luma Matte**
- [x] **Luma Inverted Matte**

## 6. Text (50%)
Text rendering and typography.

- [x] **Basic Text Rendering**
- [x] **Font Support** (System fonts + Fallback)
- [x] **Fill & Stroke**
- [x] **Justification** (Left, Center, Right)
- [x] **Line Height**
- [x] **Tracking**
- [ ] **Text on Path**
- [ ] **Text Animators** (Range Selectors, Wiggly, etc.)
- [ ] **Paragraph Text** (Box text)

## 7. Effects (10%)
Post-processing effects.

- [x] **Drop Shadow**
- [x] **Gaussian Blur**
- [x] **Color Matrix** (Custom)
- [x] **Displacement Map**
- [ ] **Tint**
- [ ] **Tritone**
- [ ] **Fill (Effect)**
- [ ] **Stroke (Effect)**
- [ ] **Levels** (Individual Controls)
- [ ] **Curves**
- [ ] **Hue/Saturation**
- [ ] **Invert**
- [ ] **Brightness & Contrast**
- [ ] **Mosaic**
- [ ] **Threshold**
- [ ] **Turbulent Displace**
- [ ] **Wave Warp**
- [ ] **Mesh Warp**
- [ ] **Twirl**
- [ ] **Spherize**
- [ ] **Bulge**
- [ ] **Corner Pin**
- [ ] **CC Radial Fast Blur**
- [ ] **CC Vector Blur**
- [ ] **Matte Choker**
- [ ] **Simple Choker**
- [ ] **Roughen Edges**
- [ ] **Venetian Blinds**

## 8. Animation & Interpolation (50%)
Keyframes and timing.

- [x] **Linear Interpolation**
- [x] **Bezier Interpolation** (Temporal Ease In/Out)
- [x] **Hold Interpolation**
- [ ] **Spatial Bezier** (Curved motion paths supported by data model, but renderer uses Linear Lerp)
- [ ] **Expressions** (JavaScript/Rhai scripting)

## 9. Audio (0%)
Sound playback.

- [ ] **Audio Layers**
- [ ] **Audio Waveforms**

## 10. Optimization & Misc
- [x] **Layer Compositing** (Skia `saveLayer` for Mattes/Effects)
- [ ] **Render Caching** (Bitmap caching of static sub-trees)
- [ ] **Marker Support**
- [ ] **Metadata**