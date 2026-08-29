#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

mod analysis;
mod app;
mod assets;
mod audio;
mod blind;
mod chain;
mod clip;
mod icon;
mod identity;
mod project;
mod theme;

use app::Muspector;
use assets::Assets;
use gpui::{AppContext, Bounds, WindowBounds, WindowOptions, px, size};

fn main() {
    gpui_platform::application().with_assets(Assets).run(|cx| {
        let bounds = Bounds::centered(None, size(px(1440.0), px(820.0)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(size(px(1050.0), px(680.0))),
                app_id: Some("dev.shiftz.muspector".to_owned()),
                ..Default::default()
            },
            |window, cx| {
                let view = cx.new(Muspector::new);
                let weak = view.downgrade();
                window.on_window_should_close(cx, move |window, cx| {
                    weak.update(cx, |this, cx| this.window_should_close(window, cx))
                        .unwrap_or(true)
                });
                view
            },
        )
        .expect("failed to open Muspector window");
        cx.activate(true);
    });
}
