# rust-lottie-skia

**A pure-Rust Lottie animation player and Skia painter.**

> ⚠️ **Architectural Note:** This crate is a **Painter**, not a Game Engine.

It is designed to be embedded into larger rendering engines (like `director-engine`). It does **not** manage windows, event loops, audio devices, or GPU contexts. It simply takes a Lottie file and a Time value, and issues draw commands to a provided `skia_safe::Canvas`.

## 📦 Modules

*   **`lottie-data`**: Serde data models for the Lottie JSON format.
*   **`lottie-core`**: The "ViewModel". Handles the scene graph, interpolation, bezier math, and state management.
*   **`lottie-skia`**: The "View". Translates the computed `lottie-core` state into `skia-safe` draw calls.

## 🚀 Usage

### 1. Load Data
```rust
use lottie_data::model::LottieJson;
use lottie_core::LottiePlayer;

let json_data = std::fs::read_to_string("animation.json")?;
let model: LottieJson = serde_json::from_str(&json_data)?;

let mut player = LottiePlayer::new();
player.load(model);
```

### 2. Update (The "Tick")
You must tell the player what time it is. The player handles the math to interpolate properties.

```rust
// Advance time (e.g., inside your engine's update loop)
// You can set exact frames or advance by seconds.
player.current_frame = target_frame; 
```

### 3. Render (The "Paint")
Pass your Engine's Skia Canvas to the renderer.

```rust
use lottie_skia::SkiaRenderer;

// Your engine provides the canvas and the layout rect
let canvas: &mut skia_safe::Canvas = ...; 
let rect = skia_safe::Rect::from_wh(500.0, 500.0);

// Get the computed state tree
let tree = player.render_tree();

// Draw it
SkiaRenderer::draw(canvas, &tree, rect, 1.0 /* opacity */);
```

## 🛠 Integration with Engines

If you are integrating this into an engine (e.g., `director-engine`), you should wrap `LottiePlayer` in a Node/Entity.

*   **Asset Loading**: This crate does not load images from disk/network. You must provide image data to `player.set_asset(id, bytes)`.
*   **Font System**: This crate renders text paths, but relies on the host to ensure fonts are available if using system fonts.
*   **Audio**: This crate does not play audio. It parses audio layer data which the host engine can extract to schedule playback.

## 🚧 Status

See `PARITY.md` for the current feature support status.