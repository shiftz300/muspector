use crate::{
    analysis::{self, Report},
    theme,
};
use gpui::{
    Animation, AnimationExt, AnyElement, AppContext, ClickEvent, Context, Div, ExternalPaths,
    IntoElement, PathBuilder, PathPromptOptions, Render, Styled, Timer, Window, canvas, div,
    ease_in_out, point, prelude::*, px, svg,
};
use std::path::{Path, PathBuf};
use std::time::Duration;

const FADE_OUT: Duration = Duration::from_millis(90);
const FADE_IN: Duration = Duration::from_millis(130);
const TINT: Duration = Duration::from_millis(140);
const HOLD: Duration = Duration::from_millis(2800);
const OPEN: usize = 0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Inspect,
    Remix,
}

impl Tab {
    fn slot(self) -> usize {
        match self {
            Self::Inspect => 1,
            Self::Remix => 2,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fade {
    Idle,
    Out,
    In,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hover {
    Idle,
    In,
    Over,
    Out,
}

enum State {
    Empty,
    Loading(PathBuf),
    Ready(Report),
}

struct Alert {
    text: String,
    fade: Fade,
}

pub struct Muspector {
    tab: Tab,
    state: State,
    job: u64,
    fade: Fade,
    cycle: usize,
    hovers: [Hover; 3],
    glows: [usize; 3],
    alert: Option<Alert>,
    notice: usize,
}

impl Muspector {
    pub fn new(_cx: &mut Context<Self>) -> Self {
        Self {
            tab: Tab::Inspect,
            state: State::Empty,
            job: 0,
            fade: Fade::Idle,
            cycle: 0,
            hovers: [Hover::Idle; 3],
            glows: [0; 3],
            alert: None,
            notice: 0,
        }
    }

    fn warn(&mut self, text: String, cx: &mut Context<Self>) {
        self.notice = self.notice.wrapping_add(1);
        let notice = self.notice;
        self.alert = Some(Alert {
            text,
            fade: Fade::In,
        });
        cx.notify();

        cx.spawn(async move |view, cx| {
            Timer::after(TINT).await;
            let active = view
                .update(cx, |this, cx| {
                    if this.notice != notice {
                        return false;
                    }
                    if let Some(alert) = &mut this.alert {
                        alert.fade = Fade::Idle;
                    }
                    cx.notify();
                    true
                })
                .unwrap_or(false);
            if !active {
                return;
            }

            Timer::after(HOLD).await;
            let active = view
                .update(cx, |this, cx| {
                    if this.notice != notice {
                        return false;
                    }
                    if let Some(alert) = &mut this.alert {
                        alert.fade = Fade::Out;
                    }
                    cx.notify();
                    true
                })
                .unwrap_or(false);
            if !active {
                return;
            }

            Timer::after(TINT).await;
            let _ = view.update(cx, |this, cx| {
                if this.notice == notice {
                    this.alert = None;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn hover(&mut self, slot: usize, hovered: bool, cx: &mut Context<Self>) {
        let settled = matches!(
            (hovered, self.hovers[slot]),
            (true, Hover::In | Hover::Over) | (false, Hover::Idle | Hover::Out)
        );
        if settled {
            return;
        }

        self.glows[slot] = self.glows[slot].wrapping_add(1);
        let glow = self.glows[slot];
        self.hovers[slot] = if hovered { Hover::In } else { Hover::Out };
        cx.notify();

        cx.spawn(async move |view, cx| {
            Timer::after(TINT).await;
            let _ = view.update(cx, |this, cx| {
                if this.glows[slot] == glow {
                    this.hovers[slot] = if hovered { Hover::Over } else { Hover::Idle };
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn pick(&mut self, cx: &mut Context<Self>) {
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open".into()),
        });
        cx.spawn(async move |view, cx| {
            if let Ok(Ok(Some(paths))) = receiver.await
                && let Some(path) = paths.into_iter().next()
            {
                let _ = view.update(cx, |this, cx| this.start(path, cx));
            }
        })
        .detach();
    }

    fn start(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if !supported(&path) {
            self.warn("Unsupported audio format".to_owned(), cx);
            return;
        }

        let previous = match &self.state {
            State::Ready(report) => Some(report.clone()),
            State::Empty | State::Loading(_) => None,
        };
        if self.tab != Tab::Inspect || self.fade != Fade::Idle {
            self.cycle = self.cycle.wrapping_add(1);
            self.tab = Tab::Inspect;
            self.fade = Fade::Idle;
        }
        if self.alert.take().is_some() {
            self.notice = self.notice.wrapping_add(1);
        }
        self.job = self.job.wrapping_add(1);
        let job = self.job;
        self.state = State::Loading(path.clone());
        cx.notify();

        let task = cx.background_spawn(async move { analysis::inspect(&path) });
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| {
                if this.job != job {
                    return;
                }
                match result {
                    Ok(report) => {
                        this.state = State::Ready(report);
                        cx.notify();
                    }
                    Err(error) => {
                        this.state = previous.map(State::Ready).unwrap_or(State::Empty);
                        this.warn(format!("{error:#}"), cx);
                    }
                }
            });
        })
        .detach();
    }

    fn switch(&mut self, tab: Tab, cx: &mut Context<Self>) {
        if tab == self.tab || self.fade != Fade::Idle {
            return;
        }

        self.cycle = self.cycle.wrapping_add(1);
        let cycle = self.cycle;
        self.fade = Fade::Out;
        cx.notify();

        cx.spawn(async move |view, cx| {
            Timer::after(FADE_OUT).await;
            let changed = view
                .update(cx, |this, cx| {
                    if this.cycle != cycle {
                        return false;
                    }
                    this.tab = tab;
                    this.fade = Fade::In;
                    cx.notify();
                    true
                })
                .unwrap_or(false);
            if !changed {
                return;
            }

            Timer::after(FADE_IN).await;
            let _ = view.update(cx, |this, cx| {
                if this.cycle == cycle {
                    this.fade = Fade::Idle;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn nav(&self, tab: Tab, label: &'static str, cx: &mut Context<Self>) -> AnyElement {
        let selected = self.tab == tab;
        let slot = tab.slot();
        let (enter, leave) = match tab {
            Tab::Inspect => ("inspect-hover-in", "inspect-hover-out"),
            Tab::Remix => ("remix-hover-in", "remix-hover-out"),
        };
        let node = div()
            .id(label)
            .w(px(68.0))
            .h(px(32.0))
            .flex()
            .items_center()
            .justify_center()
            .rounded(theme::RADIUS)
            .text_sm()
            .line_height(px(32.0))
            .text_center()
            .whitespace_nowrap()
            .font_weight(gpui::FontWeight::SEMIBOLD)
            .text_color(if selected { theme::INK } else { theme::MUTED })
            .when(selected, |node| node.bg(theme::SURFACE))
            .cursor_pointer()
            .on_hover(cx.listener(move |this, hovered: &bool, _window, cx| {
                this.hover(slot, *hovered, cx);
            }))
            .on_click(cx.listener(move |this, _: &ClickEvent, _window, cx| {
                this.switch(tab, cx);
            }))
            .child(label);

        if !selected {
            return match self.hovers[slot] {
                Hover::Idle => node.into_any_element(),
                Hover::Over => node.bg(theme::HOVER).into_any_element(),
                Hover::In => node
                    .with_animation(
                        (enter, self.glows[slot]),
                        Animation::new(TINT).with_easing(ease_in_out),
                        |node, delta| node.bg(theme::mix(theme::TRACK, theme::HOVER, delta)),
                    )
                    .into_any_element(),
                Hover::Out => node
                    .with_animation(
                        (leave, self.glows[slot]),
                        Animation::new(TINT).with_easing(ease_in_out),
                        |node, delta| node.bg(theme::mix(theme::HOVER, theme::TRACK, delta)),
                    )
                    .into_any_element(),
            };
        }

        match self.fade {
            Fade::Out => node
                .with_animation(
                    ("tab-out", self.cycle),
                    Animation::new(FADE_OUT).with_easing(ease_in_out),
                    |node, delta| node.opacity(1.0 - delta * 0.45),
                )
                .into_any_element(),
            Fade::In => node
                .with_animation(
                    ("tab-in", self.cycle),
                    Animation::new(FADE_IN).with_easing(ease_in_out),
                    |node, delta| node.opacity(0.55 + delta * 0.45),
                )
                .into_any_element(),
            _ => node.into_any_element(),
        }
    }

    fn fade(&self, content: AnyElement) -> AnyElement {
        let node = div().size_full().flex().flex_col().child(content);
        match self.fade {
            Fade::Out => node
                .with_animation(
                    ("content-out", self.cycle),
                    Animation::new(FADE_OUT).with_easing(ease_in_out),
                    |node, delta| node.opacity(1.0 - delta),
                )
                .into_any_element(),
            Fade::In => node
                .with_animation(
                    ("content-in", self.cycle),
                    Animation::new(FADE_IN).with_easing(ease_in_out),
                    |node, delta| node.opacity(delta),
                )
                .into_any_element(),
            Fade::Idle => node.into_any_element(),
        }
    }

    fn inspect(&self, cx: &mut Context<Self>) -> AnyElement {
        match &self.state {
            State::Empty => self.dropzone(cx),
            State::Loading(path) => self.loading(path),
            State::Ready(report) => self.report(report, cx),
        }
    }

    fn toast(&self) -> Option<AnyElement> {
        let alert = self.alert.as_ref()?;
        let node = div()
            .absolute()
            .top(px(76.0))
            .left_5()
            .right_5()
            .flex()
            .justify_center()
            .child(
                div()
                    .max_w(px(400.0))
                    .px_4()
                    .py_3()
                    .rounded(theme::RADIUS)
                    .border_1()
                    .border_color(theme::ERROR)
                    .bg(theme::PANEL)
                    .text_sm()
                    .text_color(theme::INK)
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .text_color(theme::ERROR)
                            .font_weight(gpui::FontWeight::BOLD)
                            .child("!"),
                    )
                    .child(alert.text.clone()),
            );

        Some(match alert.fade {
            Fade::In => node
                .with_animation(
                    ("toast-in", self.notice),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |node, delta| node.opacity(delta),
                )
                .into_any_element(),
            Fade::Out => node
                .with_animation(
                    ("toast-out", self.notice),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |node, delta| node.opacity(1.0 - delta),
                )
                .into_any_element(),
            Fade::Idle => node.into_any_element(),
        })
    }

    fn dropzone(&self, cx: &mut Context<Self>) -> AnyElement {
        let button = div()
            .id("open")
            .mt_3()
            .h(px(34.0))
            .px_4()
            .rounded(theme::RADIUS)
            .text_color(theme::ON_ACCENT)
            .text_sm()
            .font_weight(gpui::FontWeight::SEMIBOLD)
            .flex()
            .items_center()
            .justify_center()
            .cursor_pointer()
            .on_hover(cx.listener(|this, hovered: &bool, _window, cx| {
                this.hover(OPEN, *hovered, cx);
            }))
            .on_click(cx.listener(|this, _event, _window, cx| this.pick(cx)))
            .child("Open");

        let button = match self.hovers[OPEN] {
            Hover::Idle => button.bg(theme::ACCENT).into_any_element(),
            Hover::Over => button.bg(theme::ACCENT_HOVER).into_any_element(),
            Hover::In => button
                .with_animation(
                    ("open-in", self.glows[OPEN]),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |button, delta| {
                        button.bg(theme::mix(theme::ACCENT, theme::ACCENT_HOVER, delta))
                    },
                )
                .into_any_element(),
            Hover::Out => button
                .with_animation(
                    ("open-out", self.glows[OPEN]),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |button, delta| {
                        button.bg(theme::mix(theme::ACCENT_HOVER, theme::ACCENT, delta))
                    },
                )
                .into_any_element(),
        };

        div()
            .id("dropzone")
            .flex_1()
            .w_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_3()
            .rounded(theme::RADIUS)
            .border_1()
            .border_dashed()
            .border_color(theme::LINE)
            .bg(theme::SURFACE)
            .child(
                div()
                    .text_base()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::INK)
                    .child("Drop audio here"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme::MUTED)
                    .child("WAV · FLAC · MP3 · AAC · ALAC · OGG"),
            )
            .child(button)
            .into_any_element()
    }

    fn loading(&self, path: &Path) -> AnyElement {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("Audio")
            .to_owned();
        div()
            .flex_1()
            .w_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_3()
            .rounded(theme::RADIUS)
            .border_1()
            .border_color(theme::LINE)
            .bg(theme::SURFACE)
            .child(
                div()
                    .size(px(48.0))
                    .rounded_full()
                    .border_2()
                    .border_color(theme::ACCENT)
                    .flex()
                    .items_center()
                    .justify_center()
                    .font_weight(gpui::FontWeight::BOLD)
                    .text_color(theme::ACCENT)
                    .child("···"),
            )
            .child(
                div()
                    .mt_2()
                    .text_base()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::INK)
                    .child("Inspecting"),
            )
            .child(div().text_sm().text_color(theme::MUTED).child(name))
            .into_any_element()
    }

    fn report(&self, report: &Report, cx: &mut Context<Self>) -> AnyElement {
        div()
            .flex_1()
            .w_full()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .w_full()
                    .p_4()
                    .rounded(theme::RADIUS)
                    .border_1()
                    .border_color(theme::LINE)
                    .bg(theme::SURFACE)
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .size(px(40.0))
                            .rounded(theme::RADIUS)
                            .bg(theme::ACCENT_SOFT)
                            .text_color(theme::ACCENT)
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(
                                svg()
                                    .path("icons/audio.svg")
                                    .size(px(18.0))
                                    .text_color(theme::ACCENT),
                            ),
                    )
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::INK)
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .child(report.name()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(theme::MUTED)
                                    .child(report.format_text()),
                            ),
                    )
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::INK)
                            .child(report.duration_text()),
                    ),
            )
            .child(
                div()
                    .w_full()
                    .flex()
                    .gap_3()
                    .child(metric("Peak", format!("{:.1} dB", report.peak)))
                    .child(metric("RMS", format!("{:.1} dB", report.rms)))
                    .child(metric("Crest", format!("{:.1} dB", report.crest))),
            )
            .child(spectrum(report))
            .child(
                div()
                    .w_full()
                    .p_4()
                    .rounded(theme::RADIUS)
                    .border_1()
                    .border_color(theme::LINE)
                    .bg(theme::SURFACE)
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(section("Headroom", "Near-full-scale samples"))
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(if report.clips == 0 {
                                theme::INK
                            } else {
                                theme::ERROR
                            })
                            .child(report.clips.to_string()),
                    ),
            )
            .child(
                div()
                    .id("another")
                    .h(px(34.0))
                    .w_full()
                    .rounded(theme::RADIUS)
                    .border_1()
                    .border_color(theme::LINE)
                    .bg(theme::PANEL)
                    .text_sm()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::INK)
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .hover(|node| node.bg(theme::HOVER))
                    .on_click(cx.listener(|this, _event, _window, cx| this.pick(cx)))
                    .child("Another"),
            )
            .into_any_element()
    }

    fn remix(&self) -> AnyElement {
        div()
            .flex_1()
            .w_full()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_3()
            .rounded(theme::RADIUS)
            .border_1()
            .border_color(theme::LINE)
            .bg(theme::SURFACE)
            .child(
                div()
                    .text_base()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::INK)
                    .child("Remix"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme::MUTED)
                    .child("Playback arrives after the inspector."),
            )
            .child(
                div()
                    .mt_2()
                    .px_3()
                    .py_1()
                    .rounded_full()
                    .bg(theme::CANVAS)
                    .text_xs()
                    .text_color(theme::FAINT)
                    .child("Later"),
            )
            .into_any_element()
    }
}

impl Render for Muspector {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let content = match self.tab {
            Tab::Inspect => self.inspect(cx),
            Tab::Remix => self.remix(),
        };

        div()
            .id("root")
            .size_full()
            .relative()
            .bg(theme::CANVAS)
            .text_color(theme::INK)
            .flex()
            .flex_col()
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _window, cx| {
                if let Some(path) = paths.paths().first() {
                    this.start(path.clone(), cx);
                }
            }))
            .child(
                div()
                    .h(px(70.0))
                    .px_5()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .size(px(26.0))
                                    .rounded(theme::RADIUS)
                                    .bg(theme::ACCENT)
                                    .text_color(theme::SURFACE)
                                    .font_weight(gpui::FontWeight::BOLD)
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .child("M"),
                            )
                            .child(
                                div()
                                    .text_sm()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Muspector"),
                            ),
                    )
                    .child(
                        div()
                            .p_1()
                            .rounded(theme::RADIUS)
                            .bg(theme::TRACK)
                            .flex()
                            .gap_1()
                            .child(self.nav(Tab::Inspect, "Inspect", cx))
                            .child(self.nav(Tab::Remix, "Remix", cx)),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .px_5()
                    .pb_5()
                    .flex()
                    .flex_col()
                    .child(self.fade(content)),
            )
            .children(self.toast())
    }
}

fn metric(label: &'static str, value: String) -> Div {
    div()
        .flex_1()
        .min_w_0()
        .p_3()
        .rounded(theme::RADIUS)
        .border_1()
        .border_color(theme::LINE)
        .bg(theme::SURFACE)
        .flex()
        .flex_col()
        .gap_1()
        .child(div().text_xs().text_color(theme::MUTED).child(label))
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(theme::INK)
                .child(value),
        )
}

fn section(title: &'static str, note: &'static str) -> Div {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_sm()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(theme::INK)
                .child(title),
        )
        .child(div().text_xs().text_color(theme::MUTED).child(note))
}

fn spectrum(report: &Report) -> Div {
    let curve = report.spectrum.clone();
    let high = (report.rate as f64 / 2.0).clamp(40.0, 20_000.0);
    let centroid = position(report.centroid, high);
    let rolloff = position(report.rolloff, high);
    let chart = div()
        .h(px(118.0))
        .w_full()
        .relative()
        .rounded(theme::RADIUS)
        .overflow_hidden()
        .bg(theme::TRACK)
        .child(
            canvas(
                move |_, _, _| {},
                move |bounds, _, window, _| {
                    let left = bounds.origin.x + px(10.0);
                    let right = bounds.origin.x + bounds.size.width - px(10.0);
                    let top = bounds.origin.y + px(10.0);
                    let bottom = bounds.origin.y + bounds.size.height - px(20.0);
                    let width = right - left;
                    let height = bottom - top;

                    let mut grid = theme::LINE;
                    grid.a = 0.55;
                    for step in 1..4 {
                        let y = top + height * (step as f32 / 4.0);
                        let mut path = PathBuilder::stroke(px(1.0));
                        path.move_to(point(left, y));
                        path.line_to(point(right, y));
                        if let Ok(path) = path.build() {
                            window.paint_path(path, grid);
                        }
                    }

                    if curve.len() > 1 {
                        let points: Vec<_> = curve
                            .iter()
                            .enumerate()
                            .map(|(index, value)| {
                                let x = left + width * (index as f32 / (curve.len() - 1) as f32);
                                let level = ((*value + 72.0) / 72.0).clamp(0.0, 1.0) as f32;
                                point(x, bottom - height * level)
                            })
                            .collect();

                        let mut fill = PathBuilder::fill();
                        fill.move_to(point(left, bottom));
                        for point in &points {
                            fill.line_to(*point);
                        }
                        fill.line_to(point(right, bottom));
                        fill.close();
                        if let Ok(path) = fill.build() {
                            window.paint_path(path, theme::ACCENT_SOFT);
                        }

                        let mut line = PathBuilder::stroke(px(1.5));
                        for (index, point) in points.into_iter().enumerate() {
                            if index == 0 {
                                line.move_to(point);
                            } else {
                                line.line_to(point);
                            }
                        }
                        if let Ok(path) = line.build() {
                            window.paint_path(path, theme::ACCENT);
                        }
                    }

                    for (position, color) in [(centroid, theme::ACCENT), (rolloff, theme::MUTED)] {
                        let x = left + width * position;
                        let mut marker = PathBuilder::stroke(px(1.0));
                        marker = marker.dash_array(&[px(3.0), px(3.0)]);
                        marker.move_to(point(x, top));
                        marker.line_to(point(x, bottom));
                        if let Ok(path) = marker.build() {
                            window.paint_path(path, color);
                        }
                    }
                },
            )
            .size_full(),
        )
        .child(
            div()
                .absolute()
                .left_3()
                .right_3()
                .bottom_1()
                .flex()
                .justify_between()
                .text_xs()
                .text_color(theme::FAINT)
                .child("20 Hz")
                .child(format!("{:.0} kHz", high / 1_000.0)),
        );

    div()
        .w_full()
        .p_3()
        .rounded(theme::RADIUS)
        .border_1()
        .border_color(theme::LINE)
        .bg(theme::SURFACE)
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .flex()
                .items_start()
                .justify_between()
                .child(section("Spectrum", "Log frequency · 72 dB range"))
                .child(
                    div()
                        .flex()
                        .gap_3()
                        .text_xs()
                        .text_color(theme::MUTED)
                        .child(format!("C  {:.0} Hz", report.centroid))
                        .child(format!("R  {:.0} Hz", report.rolloff)),
                ),
        )
        .child(chart)
        .child(
            div()
                .flex()
                .gap_2()
                .child(energy("Low", report.low))
                .child(energy("Mid", report.mid))
                .child(energy("High", report.high)),
        )
}

fn energy(label: &'static str, value: f64) -> Div {
    div()
        .flex_1()
        .px_3()
        .py_2()
        .rounded(theme::RADIUS)
        .bg(theme::TRACK)
        .flex()
        .items_center()
        .justify_between()
        .child(div().text_xs().text_color(theme::MUTED).child(label))
        .child(
            div()
                .text_xs()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(theme::INK)
                .child(format!("{value:.1}")),
        )
}

fn position(frequency: f64, high: f64) -> f32 {
    if frequency <= 20.0 {
        return 0.0;
    }
    ((frequency / 20.0).ln() / (high / 20.0).ln()).clamp(0.0, 1.0) as f32
}

fn supported(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "wav"
                    | "wave"
                    | "flac"
                    | "mp3"
                    | "m4a"
                    | "mp4"
                    | "aac"
                    | "ogg"
                    | "oga"
                    | "aif"
                    | "aiff"
            )
        })
        .unwrap_or(false)
}
