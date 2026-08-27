mod analysis;
mod app;
mod assets;
mod blind;
mod chain;
mod theme;

use app::Muspector;
use assets::Assets;
use gpui::{AppContext, Application, Bounds, WindowBounds, WindowOptions, px, size};
use std::rc::Rc;

fn main() {
    Application::with_platform(Rc::new(gpui_macos::MacPlatform::new(false)))
        .with_assets(Assets)
        .run(|cx| {
            let bounds = Bounds::centered(None, size(px(1440.0), px(820.0)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(1050.0), px(680.0))),
                    ..Default::default()
                },
                |_window, cx| cx.new(Muspector::new),
            )
            .expect("failed to open Muspector window");
            cx.activate(true);
        });
}
