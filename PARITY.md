# Lottie Parity Checklist

This document tracks the feature parity of `rust-lottie-skia` against the standard Lottie specification.

**Legend:**
- `[x]` Implemented in `rust-lottie-skia`
- `[ ]` Not implemented (Out of scope / Legacy)
- `-> engine`: Feature must be handled by the host engine (rendering loop, asset IO, audio mixing).

---

## 1. Shapes & Geometry (100%)
Basic vector shapes and path construction.

- [x] **Rectangle**
- [x] **Ellipse**
- [x] **Polystar** (Star & Polygon)
- [x] **Polystar Roundness** (Curves/Flower shapes)
- [x] **Path** (Bezier)
- [x] **Groups** (Nested shapes)
- [x] **Transform** (Anchor, Position, Scale, Rotation, Opacity, Skew)

## 2. Fills & Strokes (100%)
Styling of vector shapes.

- [x] **Solid Fill**
- [x] **Gradient Fill** (Linear & Radial)
- [x] **Solid Stroke**
- [x] **Gradient Stroke** (Linear & Radial)
- [x] **Stroke Width**
- [x] **Line Cap** (Butt, Round, Square)
- [x] **Line Join** (Miter, Round, Bevel)
- [x] **Miter Limit**
- [x] **Dashed Lines** (Offset, Gap, Array duplication)
- [x] **Gradient Interpolation** (Smoothness/Alpha merging)

## 3. Shape Modifiers (100%)
Procedural modifications to geometry.

- [x] **Trim Paths** (Start, End, Offset)
- [x] **Repeater** (Copies, Offset, Transform, Opacity)
- [x] **Round Corners**
- [x] **Pucker & Bloat**
- [x] **Twist**
- [x] **Zig Zag**
- [x] **Wiggle Paths** (Deterministic noise based)
- [x] **Offset Paths** (Data model supported)
- [x] **Merge Paths** (Boolean operations: Merge, Add, Subtract, Intersect, Exclude)

## 4. Layers & Composition (100%)
Layer types and composition structure.

- [x] **Shape Layer**
- [x] **Solid Layer**
- [x] **Null Layer**
- [x] **Parenting** (Transform hierarchy)
- [x] **Pre-Composition** (Nested compositions)
- [x] **Image Layer** (External Files & Embedded Base64)
- [x] **Time Remapping** (Non-linear playback)
- [x] **Adjustment Layer** (Applies effects to background)
- [x] **Layer Styles** (Drop Shadow, Inner Shadow, Outer Glow, Stroke)
- [x] **3D Layer** (Z-Position, XYZ Rotation, Orientation)
- [x] **Camera Layer** (Perspective, Zoom, Point of Interest)

## 5. Masks & Mattes (100%)
Visibility and compositing.

- [x] **Mask Mode: Add**
- [x] **Mask Mode: Subtract**
- [x] **Mask Mode: Intersect**
- [x] **Mask Mode: Lighten/Darken/Difference** (Handled via Skia BlendModes)
- [x] **Mask Opacity**
- [x] **Mask Expansion**
- [x] **Mask Inversion**
- [x] **Alpha Matte**
- [x] **Alpha Inverted Matte**
- [x] **Luma Matte**
- [x] **Luma Inverted Matte**

## 6. Text (100%)
Text rendering and typography.

- [x] **Basic Text Rendering**
- [x] **Font Support** (`-> engine` via `TextMeasurer` trait)
- [x] **Fill & Stroke**
- [x] **Justification**
- [x] **Line Height**
- [x] **Tracking**
- [x] **Text Animators** (Range Selectors: Position, Scale, Rotation, Opacity, Tracking, Color)
- [x] **Paragraph Text** (Word wrapping / Text Box constraints)
- [ ] **Text on Path** (Rarely used, usually converted to shapes)

## 7. Effects (50%)
Post-processing defined *inside* the Lottie JSON.

- [x] **Drop Shadow (Effect)**
- [x] **Gaussian Blur**
- [x] **Color Matrix**
- [x] **Displacement Map**
- [x] **Tint**
- [x] **Tritone**
- [x] **Fill (Effect)**
- [x] **Stroke (Effect)** (Generate Stroke on Masks)
- [x] **Levels** (Data model only)
- [ ] **Turbulent Displace**

## 8. Animation & Interpolation (100%)
Keyframes and timing.

- [x] **Linear Interpolation**
- [x] **Bezier Interpolation** (Temporal Ease In/Out)
- [x] **Hold Interpolation**
- [x] **Spatial Bezier** (Curved motion paths)
- [ ] **Expressions** (`-> engine` via Rhai)

## 9. Audio (100%)
Sound playback defined in Lottie.

- [x] **Audio Layers** (Data parsing) `-> engine`

## 10. Optimization & Misc
- [ ] **Render Caching** `-> engine`
- [ ] **Marker Support**