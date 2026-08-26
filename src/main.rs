mod analysis;
mod app;
mod assets;
mod theme;

use app::Muspector;
use assets::Assets;
use gpui::{AppContext, Application, Bounds, WindowBounds, WindowOptions, px, size};

fn main() {
    Application::new().with_assets(Assets).run(|cx| {
        let bounds = Bounds::centered(None, size(px(520.0), px(680.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_window, cx| cx.new(Muspector::new),
        )
        .expect("failed to open Muspector window");
        cx.activate(true);
    });
}
