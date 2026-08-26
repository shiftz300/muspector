use crate::{
    analysis::{self, Report},
    chain::{Chain, Effect, Param},
    theme,
};
use gpui::{
    Animation, AnimationExt, AnyElement, AppContext, Bounds, ClickEvent, Context, Div,
    ExternalPaths, FocusHandle, IntoElement, KeyDownEvent, MouseMoveEvent, PathBuilder,
    PathPromptOptions, Pixels, Render, ScrollHandle, ScrollWheelEvent, Styled, Timer, Window,
    canvas, div, ease_in_out, point, prelude::*, px, size, svg,
};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
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

struct Edit {
    effect: usize,
    param: usize,
    text: String,
    fresh: bool,
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
    cursor: Option<f32>,
    fit: bool,
    baseline: Option<Chain>,
    focus: FocusHandle,
    edit: Option<Edit>,
    cards: [usize; 6],
    tracks: [ScrollHandle; 2],
}

impl Muspector {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let mut this = Self {
            tab: Tab::Inspect,
            state: State::Empty,
            job: 0,
            fade: Fade::Idle,
            cycle: 0,
            hovers: [Hover::Idle; 3],
            glows: [0; 3],
            alert: None,
            notice: 0,
            cursor: None,
            fit: false,
            baseline: None,
            focus: cx.focus_handle(),
            edit: None,
            cards: [0; 6],
            tracks: [ScrollHandle::new(), ScrollHandle::new()],
        };
        if let Some(path) = std::env::args_os().nth(1) {
            this.start(PathBuf::from(path), cx);
        }
        this
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
        self.cursor = None;
        self.fit = false;
        self.edit = None;
        self.cards = [0; 6];
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
                        this.baseline = Some(report.chain.clone());
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
        self.cursor = None;
        self.edit = None;
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

    fn toggle(&mut self, effect: usize, cx: &mut Context<Self>) {
        if let State::Ready(report) = &mut self.state
            && let Some(item) = report.chain.effects.get_mut(effect)
        {
            item.active = !item.active;
            self.cards[effect] = self.cards[effect].wrapping_add(1);
            cx.notify();
        }
    }

    fn adjust(&mut self, effect: usize, param: usize, direction: f64, cx: &mut Context<Self>) {
        self.edit = None;
        if let State::Ready(report) = &mut self.state
            && let Some(param) = report
                .chain
                .effects
                .get_mut(effect)
                .and_then(|effect| effect.params.get_mut(param))
        {
            param.shift(direction);
            cx.notify();
        }
    }

    fn reset(&mut self, cx: &mut Context<Self>) {
        if let (State::Ready(report), Some(baseline)) = (&mut self.state, &self.baseline) {
            report.chain = baseline.clone();
            self.edit = None;
            for card in &mut self.cards {
                *card = card.wrapping_add(1);
            }
            cx.notify();
        }
    }

    fn begin(&mut self, effect: usize, param: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(value) = (match &self.state {
            State::Ready(report) => report
                .chain
                .effects
                .get(effect)
                .and_then(|effect| effect.params.get(param))
                .map(|param| param.value),
            State::Empty | State::Loading(_) => None,
        }) else {
            return;
        };
        self.edit = Some(Edit {
            effect,
            param,
            text: value.to_string(),
            fresh: true,
        });
        window.focus(&self.focus);
        cx.notify();
    }

    fn key(&mut self, event: &KeyDownEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if self.edit.is_none() {
            cx.propagate();
            return;
        }
        cx.stop_propagation();
        match event.keystroke.key.as_str() {
            "enter" | "return" => {
                let edit = self.edit.take().expect("edit checked above");
                if let Ok(value) = edit.text.parse::<f64>() {
                    if let State::Ready(report) = &mut self.state
                        && let Some(param) = report
                            .chain
                            .effects
                            .get_mut(edit.effect)
                            .and_then(|effect| effect.params.get_mut(edit.param))
                    {
                        param.set(value);
                    }
                    cx.notify();
                } else {
                    self.warn("Enter a valid number".to_owned(), cx);
                }
            }
            "escape" => {
                self.edit = None;
                cx.notify();
            }
            "backspace" => {
                if let Some(edit) = &mut self.edit {
                    if edit.fresh {
                        edit.text.clear();
                        edit.fresh = false;
                    } else {
                        edit.text.pop();
                    }
                    cx.notify();
                }
            }
            _ => {
                if let Some(text) = &event.keystroke.key_char
                    && text
                        .chars()
                        .all(|value| value.is_ascii_digit() || matches!(value, '.' | '-'))
                    && let Some(edit) = &mut self.edit
                {
                    if edit.fresh {
                        edit.text.clear();
                        edit.fresh = false;
                    }
                    edit.text.push_str(text);
                    cx.notify();
                }
            }
        }
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
            .id("report")
            .flex_1()
            .min_h_0()
            .w_full()
            .flex()
            .flex_col()
            .overflow_y_scroll()
            .scrollbar_width(px(8.0))
            .track_scroll(&self.tracks[0])
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
            .child(self.summary(&report.chain, cx))
            .child(self.profile(report, cx))
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

    fn summary(&self, chain: &Chain, cx: &mut Context<Self>) -> AnyElement {
        let names: Vec<_> = chain.active().map(|effect| effect.kind.name()).collect();
        let card = div()
            .id("chain")
            .w_full()
            .p_3()
            .rounded(theme::RADIUS)
            .border_1()
            .border_color(theme::LINE)
            .bg(theme::SURFACE)
            .flex()
            .items_center()
            .justify_between()
            .gap_3()
            .cursor_pointer()
            .hover(|node| node.bg(theme::HOVER))
            .on_click(cx.listener(|this, _event, _window, cx| this.switch(Tab::Remix, cx)))
            .child(section("Chain", "Heuristic candidate · open to tune"))
            .child(
                div()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme::MUTED)
                            .child(format!("{:.0}%", chain.score * 100.0)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::ACCENT)
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .child(names.join("  →  ")),
                    ),
            );
        card.with_animation(
            ("chain-in", self.job),
            Animation::new(Duration::from_millis(180)).with_easing(ease_in_out),
            |card, delta| card.opacity(delta),
        )
        .into_any_element()
    }

    fn profile(&self, report: &Report, cx: &mut Context<Self>) -> AnyElement {
        let points = report.profile.points.clone();
        let duration = report.duration;
        let cursor = self.cursor;
        let bounds: Rc<RefCell<Option<Bounds<Pixels>>>> = Rc::new(RefCell::new(None));
        let capture = bounds.clone();
        let locate = bounds.clone();

        let detail = cursor
            .and_then(|position| {
                let index = (position * points.len().saturating_sub(1) as f32).round() as usize;
                points.get(index).map(|point| {
                    let peak = f64::from(point.min.abs().max(point.max.abs()));
                    format!(
                        "{}  ·  Peak {:.1} dB  ·  RMS {:.1} dB",
                        stamp(duration * f64::from(position)),
                        level(peak),
                        point.level
                    )
                })
            })
            .unwrap_or_else(|| "Hover to inspect".to_owned());

        let chart = div()
            .id("profile")
            .h(px(136.0))
            .w_full()
            .relative()
            .rounded(theme::RADIUS)
            .overflow_hidden()
            .bg(theme::TRACK)
            .cursor_crosshair()
            .child(
                canvas(
                    move |area, _, _| {
                        *capture.borrow_mut() = Some(area);
                    },
                    move |area, _, window, _| {
                        let left = area.origin.x + px(10.0);
                        let right = area.origin.x + area.size.width - px(10.0);
                        let top = area.origin.y + px(10.0);
                        let bottom = area.origin.y + area.size.height - px(20.0);
                        let width = right - left;
                        let wave_bottom = top + (bottom - top) * 0.62;
                        let middle = top + (wave_bottom - top) * 0.5;
                        let amplitude = (wave_bottom - top) * 0.45;
                        let level_top = wave_bottom + px(12.0);

                        let mut grid = theme::LINE;
                        grid.a = 0.55;
                        for y in [middle, wave_bottom, level_top] {
                            let mut path = PathBuilder::stroke(px(1.0));
                            path.move_to(point(left, y));
                            path.line_to(point(right, y));
                            if let Ok(path) = path.build() {
                                window.paint_path(path, grid);
                            }
                        }

                        if points.len() > 1 {
                            let x = |index: usize| {
                                left + width * (index as f32 / (points.len() - 1) as f32)
                            };
                            let mut wave = PathBuilder::fill();
                            for (index, item) in points.iter().enumerate() {
                                let point = point(x(index), middle - amplitude * item.max);
                                if index == 0 {
                                    wave.move_to(point);
                                } else {
                                    wave.line_to(point);
                                }
                            }
                            for (index, item) in points.iter().enumerate().rev() {
                                wave.line_to(point(x(index), middle - amplitude * item.min));
                            }
                            wave.close();
                            if let Ok(path) = wave.build() {
                                window.paint_path(path, theme::ACCENT_SOFT);
                            }

                            let mut peak = PathBuilder::stroke(px(1.0));
                            for (index, item) in points.iter().enumerate() {
                                let sample = item.min.abs().max(item.max.abs());
                                let point = point(x(index), middle - amplitude * sample);
                                if index == 0 {
                                    peak.move_to(point);
                                } else {
                                    peak.line_to(point);
                                }
                            }
                            if let Ok(path) = peak.build() {
                                window.paint_path(path, theme::ACCENT);
                            }

                            let mut rms = PathBuilder::stroke(px(1.5));
                            for (index, item) in points.iter().enumerate() {
                                let value = ((item.level + 72.0) / 72.0).clamp(0.0, 1.0) as f32;
                                let point = point(x(index), bottom - (bottom - level_top) * value);
                                if index == 0 {
                                    rms.move_to(point);
                                } else {
                                    rms.line_to(point);
                                }
                            }
                            if let Ok(path) = rms.build() {
                                window.paint_path(path, theme::ACCENT_HOVER);
                            }
                        }

                        if let Some(position) = cursor {
                            let x = left + width * position;
                            let mut marker = PathBuilder::stroke(px(1.0));
                            marker.move_to(point(x, top));
                            marker.line_to(point(x, bottom));
                            if let Ok(path) = marker.build() {
                                window.paint_path(path, theme::INK);
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
                    .top_1()
                    .text_xs()
                    .text_color(theme::FAINT)
                    .child("Wave"),
            )
            .child(
                div()
                    .absolute()
                    .left_3()
                    .top(px(78.0))
                    .text_xs()
                    .text_color(theme::FAINT)
                    .child("RMS"),
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
                    .child("0:00")
                    .child(stamp(duration)),
            )
            .on_mouse_move(cx.listener(move |this, event: &MouseMoveEvent, _, cx| {
                let Some(area) = *locate.borrow() else {
                    return;
                };
                let width = f32::from(area.size.width);
                if width <= 0.0 {
                    return;
                }
                let position =
                    (f32::from(event.position.x - area.origin.x) / width).clamp(0.0, 1.0);
                if this
                    .cursor
                    .is_none_or(|current| (current - position).abs() > 0.002)
                {
                    this.cursor = Some(position);
                    cx.notify();
                }
            }))
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !hovered && this.cursor.take().is_some() {
                    cx.notify();
                }
            }));

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
                    .child(section("Profile", "Waveform · short-term RMS"))
                    .child(div().text_xs().text_color(theme::MUTED).child(detail)),
            )
            .child(chart)
            .into_any_element()
    }

    fn remix(&self, cx: &mut Context<Self>) -> AnyElement {
        let State::Ready(report) = &self.state else {
            return div()
                .flex_1()
                .w_full()
                .flex()
                .flex_col()
                .items_center()
                .justify_center()
                .gap_2()
                .rounded(theme::RADIUS)
                .border_1()
                .border_color(theme::LINE)
                .bg(theme::SURFACE)
                .child(
                    div()
                        .text_base()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(theme::INK)
                        .child("No chain"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme::MUTED)
                        .child("Inspect audio to build a candidate."),
                )
                .into_any_element();
        };

        div()
            .id("remix")
            .flex_1()
            .min_h_0()
            .w_full()
            .flex()
            .flex_col()
            .overflow_y_scroll()
            .scrollbar_width(px(8.0))
            .track_scroll(&self.tracks[1])
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
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_base()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Candidate"),
                            )
                            .child(div().text_xs().text_color(theme::MUTED).child(format!(
                                "{} · {:.0}% heuristic confidence",
                                report.name(),
                                report.chain.score * 100.0
                            ))),
                    )
                    .child(
                        div()
                            .id("reset")
                            .px_3()
                            .py_1()
                            .rounded(theme::RADIUS)
                            .border_1()
                            .border_color(theme::LINE)
                            .text_xs()
                            .text_color(theme::INK)
                            .cursor_pointer()
                            .hover(|node| node.bg(theme::HOVER))
                            .on_click(cx.listener(|this, _event, _window, cx| this.reset(cx)))
                            .child("Reset"),
                    ),
            )
            .children(
                report
                    .chain
                    .effects
                    .iter()
                    .enumerate()
                    .map(|(index, effect)| self.effect(index, effect, cx)),
            )
            .into_any_element()
    }

    fn effect(&self, index: usize, effect: &Effect, cx: &mut Context<Self>) -> AnyElement {
        let active = effect.active;
        let card = div()
            .w_full()
            .min_w(px(500.0))
            .p_3()
            .rounded(theme::RADIUS)
            .border_1()
            .border_color(if active {
                theme::ACCENT_SOFT
            } else {
                theme::LINE
            })
            .bg(theme::SURFACE)
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .items_start()
                    .justify_between()
                    .gap_3()
                    .child(
                        div()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .text_color(theme::INK)
                                            .child(effect.kind.name()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(theme::MUTED)
                                            .child(format!("{:.0}%", effect.score * 100.0)),
                                    ),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(theme::MUTED)
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .child(effect.evidence.clone()),
                            ),
                    )
                    .child(
                        div()
                            .id(("toggle", index))
                            .w(px(42.0))
                            .py_1()
                            .rounded(theme::RADIUS)
                            .bg(if active {
                                theme::ACCENT_SOFT
                            } else {
                                theme::TRACK
                            })
                            .text_xs()
                            .text_color(if active { theme::ACCENT } else { theme::FAINT })
                            .text_center()
                            .cursor_pointer()
                            .on_click(cx.listener(move |this, _event, _window, cx| {
                                this.toggle(index, cx);
                            }))
                            .child(if active { "On" } else { "Off" }),
                    ),
            )
            .child(div().w_full().min_w(px(436.0)).flex().gap_2().children(
                effect.params.iter().enumerate().map(|(param, value)| {
                    let edit = self
                        .edit
                        .as_ref()
                        .filter(|edit| edit.effect == index && edit.param == param)
                        .map(|edit| edit.text.clone());
                    knob(index, param, value, edit, active, cx)
                }),
            ));
        let end = if active { 1.0 } else { 0.58 };
        let start = if self.cards[index] == 0 {
            0.0
        } else if active {
            0.58
        } else {
            1.0
        };
        let token = self.job.wrapping_mul(10_000) + index as u64 * 100 + self.cards[index] as u64;
        card.with_animation(
            ("effect", token),
            Animation::new(Duration::from_millis(180)).with_easing(ease_in_out),
            move |card, delta| card.opacity(start + (end - start) * delta),
        )
        .into_any_element()
    }
}

impl Render for Muspector {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if matches!(self.state, State::Ready(_)) && !self.fit {
            self.fit = true;
            let current = window.viewport_size();
            let available = window
                .display(cx)
                .map(|display| display.bounds().size)
                .unwrap_or(size(px(620.0), px(1_000.0)));
            let width = current
                .width
                .max(px(620.0))
                .min((available.width - px(40.0)).max(px(480.0)));
            let height = current
                .height
                .max(px(1_000.0))
                .min((available.height - px(80.0)).max(px(640.0)));
            let fitted = size(width, height);
            if fitted != current {
                window.resize(fitted);
            }
        }

        let content = match self.tab {
            Tab::Inspect => self.inspect(cx),
            Tab::Remix => self.remix(cx),
        };

        div()
            .id("root")
            .size_full()
            .min_w(px(580.0))
            .relative()
            .track_focus(&self.focus)
            .on_key_down(cx.listener(|this, event, window, cx| this.key(event, window, cx)))
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
                    .relative()
                    .px_5()
                    .pb_5()
                    .flex()
                    .flex_col()
                    .child(self.fade(content))
                    .child(bar(&self.tracks[match self.tab {
                        Tab::Inspect => 0,
                        Tab::Remix => 1,
                    }])),
            )
            .children(self.toast())
    }
}

fn bar(handle: &ScrollHandle) -> AnyElement {
    let handle = handle.clone();
    div()
        .absolute()
        .top_0()
        .right(px(4.0))
        .bottom_5()
        .w(px(4.0))
        .child(
            canvas(
                move |_, _, _| {},
                move |bounds, _, window, _| {
                    let maximum = f32::from(handle.max_offset().height).max(0.0);
                    if maximum <= 0.5 {
                        return;
                    }
                    let height = f32::from(bounds.size.height);
                    let viewport = f32::from(handle.bounds().size.height).max(1.0);
                    let thumb = (height * viewport / (viewport + maximum)).clamp(28.0, height);
                    let progress = (-f32::from(handle.offset().y) / maximum).clamp(0.0, 1.0);
                    let top = f32::from(bounds.origin.y) + (height - thumb) * progress;
                    let left = bounds.origin.x;
                    let right = bounds.origin.x + bounds.size.width;
                    let top = px(top);
                    let bottom = top + px(thumb);
                    let mut path = PathBuilder::fill();
                    path.add_polygon(
                        &[
                            point(left, top),
                            point(right, top),
                            point(right, bottom),
                            point(left, bottom),
                        ],
                        true,
                    );
                    if let Ok(path) = path.build() {
                        let mut color = theme::FAINT;
                        color.a = 0.72;
                        window.paint_path(path, color);
                    }
                },
            )
            .size_full(),
        )
        .into_any_element()
}

fn knob(
    effect: usize,
    index: usize,
    param: &Param,
    edit: Option<String>,
    active: bool,
    cx: &mut Context<Muspector>,
) -> AnyElement {
    let normal = param.normal();
    let label = param.name;
    let editing = edit.is_some();
    let value = edit
        .map(|text| format!("{text}| "))
        .unwrap_or_else(|| param.text());
    div()
        .id(("knob", effect * 100 + index))
        .flex_none()
        .w(px(140.0))
        .p_2()
        .rounded(theme::RADIUS)
        .bg(theme::TRACK)
        .flex()
        .flex_col()
        .items_center()
        .gap_1()
        .cursor_ns_resize()
        .on_scroll_wheel(
            cx.listener(move |this, event: &ScrollWheelEvent, _window, cx| {
                let movement = f32::from(event.delta.pixel_delta(px(16.0)).y);
                if movement.abs() > 0.01 {
                    this.adjust(effect, index, -f64::from(movement.signum()), cx);
                    cx.stop_propagation();
                }
            }),
        )
        .child(
            div().size(px(46.0)).child(
                canvas(
                    move |_, _, _| {},
                    move |bounds, _, window, _| {
                        let center = bounds.center();
                        let radius = px(18.0);
                        let mut ring = PathBuilder::stroke(px(2.0));
                        ring.move_to(point(center.x + radius, center.y));
                        ring.arc_to(
                            point(radius, radius),
                            px(0.0),
                            false,
                            false,
                            point(center.x - radius, center.y),
                        );
                        ring.arc_to(
                            point(radius, radius),
                            px(0.0),
                            false,
                            false,
                            point(center.x + radius, center.y),
                        );
                        if let Ok(path) = ring.build() {
                            window
                                .paint_path(path, if active { theme::LINE } else { theme::FAINT });
                        }

                        let angle = (-135.0_f32 + normal * 270.0).to_radians();
                        let end = point(
                            center.x + px(angle.sin() * 13.0),
                            center.y - px(angle.cos() * 13.0),
                        );
                        let mut hand = PathBuilder::stroke(px(2.0));
                        hand.move_to(center);
                        hand.line_to(end);
                        if let Ok(path) = hand.build() {
                            window.paint_path(
                                path,
                                if active { theme::ACCENT } else { theme::MUTED },
                            );
                        }
                    },
                )
                .size_full(),
            ),
        )
        .child(div().text_xs().text_color(theme::MUTED).child(label))
        .child(
            div()
                .w_full()
                .flex()
                .items_center()
                .justify_between()
                .gap_1()
                .child(step(effect, index, -1.0, "−", cx))
                .child(
                    div()
                        .id(("value", effect * 100 + index))
                        .min_w_0()
                        .px_1()
                        .rounded(theme::RADIUS)
                        .when(editing, |node| node.bg(theme::ACCENT_SOFT))
                        .text_xs()
                        .text_color(if editing { theme::ACCENT } else { theme::INK })
                        .cursor_text()
                        .on_click(cx.listener(move |this, _event, window, cx| {
                            this.begin(effect, index, window, cx);
                        }))
                        .child(value),
                )
                .child(step(effect, index, 1.0, "+", cx)),
        )
        .into_any_element()
}

fn step(
    effect: usize,
    param: usize,
    direction: f64,
    label: &'static str,
    cx: &mut Context<Muspector>,
) -> AnyElement {
    div()
        .id((
            "step",
            effect * 100 + param * 2 + usize::from(direction > 0.0),
        ))
        .size(px(20.0))
        .rounded(theme::RADIUS)
        .bg(theme::SURFACE)
        .text_xs()
        .text_color(theme::MUTED)
        .flex()
        .items_center()
        .justify_center()
        .cursor_pointer()
        .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
        .on_click(cx.listener(move |this, _event, _window, cx| {
            this.adjust(effect, param, direction, cx);
        }))
        .child(label)
        .into_any_element()
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

fn level(value: f64) -> f64 {
    20.0 * value.max(1.0e-12).log10()
}

fn stamp(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let hours = total / 3_600;
    let minutes = (total % 3_600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
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
