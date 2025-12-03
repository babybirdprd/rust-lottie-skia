# Lottie Parity Checklist

This document tracks the feature parity of `rust-lottie-skia` against the standard Lottie specification.

**Legend:**
- `[x]` Implemented in `rust-lottie-skia`
- `[ ]` Not implemented
- `-> engine`: Feature should be handled by `director-engine` (rendering loop, asset management, or audio mixing), or the library should just expose data for the engine to use.

---

## 1. Shapes & Geometry (85%)
Basic vector shapes and path construction. **(Library Responsibility)**

- [x] **Rectangle**
- [x] **Ellipse**
- [x] **Polystar** (Star & Polygon)
- [x] **Path** (Bezier)
- [x] **Groups** (Nested shapes)
- [x] **Transform** (Anchor, Position, Scale, Rotation, Opacity, Skew)

## 2. Fills & Strokes (90%)
Styling of vector shapes. **(Library Responsibility)**

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

## 3. Shape Modifiers (40%)
Procedural modifications to geometry. **(Library Responsibility)**

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
- [x] **Null Layer**
- [x] **Parenting** (Transform hierarchy)
- [x] **Pre-Composition** (Nested compositions)
- [x] **Image Layer** (Placement & Transform handled by Lib)
  - `-> engine`: **Asset Loading**. The Engine's `AssetLoader` should provide the bytes/pixels for external or embedded images.
- [ ] **Time Remapping** (Property `tm` exists but logic is missing in Lib)
- [ ] **3D Layer** (Camera, Z-axis)
- [ ] **Adjustment Layer** (Applies effects to layers below)

## 5. Masks & Mattes (80%)
Visibility and compositing. **(Library Responsibility)**

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
- [x] **Font Support**
  - `-> engine`: **Font System**. The Engine holds the `cosmic-text` / `FontSystem`. The Library should request font glyphs/paths from the Engine.
- [x] **Fill & Stroke**
- [x] **Justification**
- [x] **Line Height**
- [x] **Tracking**
- [ ] **Text on Path**
- [ ] **Text Animators** (Range Selectors, Wiggly, etc. - Critical for Lottie text)
- [ ] **Paragraph Text** (Box text wrapping)

## 7. Effects (10%)
Post-processing defined *inside* the Lottie JSON.

- [x] **Drop Shadow**
- [x] **Gaussian Blur**
- [x] **Color Matrix**
- [x] **Displacement Map**
- [ ] **Tint / Tritone**
- [ ] **Fill / Stroke (Effect)**
- [ ] **Curves / Levels**
- [ ] **Turbulent Displace**
- **Note:** If the user wants to apply effects *programmatically* (via Rhai), that is handled by the **Engine**. The Library only needs to handle effects baked into the JSON file.

## 8. Animation & Interpolation (50%)
Keyframes and timing.

- [x] **Linear Interpolation**
- [x] **Bezier Interpolation** (Temporal Ease In/Out)
- [x] **Hold Interpolation**
- [ ] **Spatial Bezier** (Curved motion paths - Data model exists, but renderer uses Linear Lerp)
- [ ] **Expressions** (JavaScript/Rhai)
  - `-> engine`: **Scripting**. The Engine already has `Rhai`. Ideally, Lottie expressions could be bridged to the Engine's scripting context, though Lottie uses JS syntax.

## 9. Audio (0%)
Sound playback defined in Lottie.

- [ ] **Audio Layers**
  - `-> engine`: **Mixing**. The Library should simply extract the audio reference and timing (start/end). The Engine's `AudioMixer` should handle the actual mixing and playback.
- [ ] **Audio Waveforms** (Visualizing audio data)

## 10. Optimization & Misc
- [ ] **Render Caching**
  - `-> engine`: **Bitmap Caching**. The Engine controls the surface. It should decide if a Lottie node is static and cache it to an Image/Surface to avoid re-running the Lottie renderer every frame.
- [ ] **Marker Support** (Expose markers for Engine events)