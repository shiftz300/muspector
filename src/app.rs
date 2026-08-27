use crate::{
    analysis::{self, Report},
    chain::{Chain, Effect, Param},
    theme,
};
use gpui::{
    Animation, AnimationExt, AnyElement, App, AppContext, BorderStyle, Bounds, ClickEvent, Context,
    Corners, DispatchPhase, Div, DragMoveEvent, Edges, Element, ElementId, ExternalPaths,
    FocusHandle, GlobalElementId, Hitbox, HitboxBehavior, InspectorElementId, IntoElement,
    KeyDownEvent, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathBuilder,
    PathPromptOptions, PinchEvent, Pixels, Point, Position, PromptLevel, Render, ScrollHandle,
    ScrollWheelEvent, Style, Styled, Transformation, Window, canvas, div, ease_in_out, point,
    prelude::*, px, quad, radians, relative, size, svg, transparent_black,
};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;
use sysinfo::System;

const TINT: Duration = Duration::from_millis(140);
const CARD: Duration = Duration::from_millis(180);
const HOLD: Duration = Duration::from_millis(2800);
const REST: Duration = Duration::from_millis(760);
const OPEN: usize = 0;
const DEVICE_CLOSED: f32 = 50.0;
const DEVICE_OPENED: f32 = 336.0;
const DEVICE_RAIL: f32 = 40.0;

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum OverviewDrag {
    Pan,
    Left,
    Right,
}

enum State {
    Empty,
    Loading(PathBuf),
    Ready(Box<Report>),
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

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Pressure {
    cpu: usize,
    ram: usize,
}

struct InspectorScrollbar {
    handle: ScrollHandle,
    drag: Rc<RefCell<Option<f32>>>,
}

struct ScrollPaint {
    track: Bounds<Pixels>,
    thumb: Bounds<Pixels>,
    hitbox: Hitbox,
}

impl IntoElement for InspectorScrollbar {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for InspectorScrollbar {
    type RequestLayoutState = ();
    type PrepaintState = Option<ScrollPaint>;

    fn id(&self) -> Option<ElementId> {
        Some("inspector-scrollbar".into())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let style = Style {
            position: Position::Absolute,
            inset: Edges::default(),
            size: size(relative(1.0), relative(1.0)).map(Into::into),
            ..Default::default()
        };
        (window.request_layout(style, [], cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        _cx: &mut App,
    ) -> Self::PrepaintState {
        let viewport = self.handle.bounds();
        let height = (f32::from(viewport.size.height) - 8.0).max(0.0);
        let (maximum, thumb) = scroll_size(&self.handle, height)?;
        let progress = (-f32::from(self.handle.offset().y) / maximum).clamp(0.0, 1.0);
        let track = Bounds::new(
            point(viewport.right() - px(10.0), viewport.top() + px(4.0)),
            size(px(10.0), px(height)),
        );
        let thumb = Bounds::new(
            point(
                track.origin.x + px(3.0),
                track.origin.y + px((height - thumb) * progress),
            ),
            size(px(5.0), px(thumb)),
        );
        Some(ScrollPaint {
            track,
            thumb,
            hitbox: window.insert_hitbox(track, HitboxBehavior::BlockMouseExceptScroll),
        })
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        _cx: &mut App,
    ) {
        let Some(ScrollPaint {
            track,
            thumb,
            hitbox,
        }) = prepaint.take()
        else {
            return;
        };
        let active = self.drag.borrow().is_some();
        let mut color = theme::INK;
        color.a = if active { 0.72 } else { 0.46 };
        window.paint_quad(quad(
            thumb,
            Corners::all(px(3.0)),
            color,
            Edges::default(),
            transparent_black(),
            BorderStyle::default(),
        ));

        window.on_mouse_event({
            let handle = self.handle.clone();
            let drag = self.drag.clone();
            let hitbox = hitbox.clone();
            move |event: &MouseDownEvent, phase, window, cx| {
                if phase != DispatchPhase::Bubble
                    || event.button != MouseButton::Left
                    || !hitbox.is_hovered(window)
                {
                    return;
                }
                let local = f32::from(event.position.y - track.origin.y);
                let current = f32::from(thumb.origin.y - track.origin.y);
                let thumb_height = f32::from(thumb.size.height);
                let anchor = if thumb.contains(&event.position) {
                    local - current
                } else {
                    thumb_height * 0.5
                };
                *drag.borrow_mut() = Some(anchor);
                if !thumb.contains(&event.position) {
                    set_scroll(&handle, f32::from(track.size.height), local, anchor);
                }
                window.refresh();
                cx.stop_propagation();
            }
        });

        window.on_mouse_event({
            let handle = self.handle.clone();
            let drag = self.drag.clone();
            move |event: &MouseMoveEvent, phase, window, cx| {
                if phase != DispatchPhase::Capture || !event.dragging() {
                    return;
                }
                let Some(anchor) = *drag.borrow() else {
                    return;
                };
                let local = f32::from(event.position.y - track.origin.y);
                set_scroll(&handle, f32::from(track.size.height), local, anchor);
                window.refresh();
                cx.stop_propagation();
            }
        });

        window.on_mouse_event({
            let drag = self.drag.clone();
            move |event: &MouseUpEvent, phase, window, cx| {
                if phase != DispatchPhase::Capture
                    || event.button != MouseButton::Left
                    || drag.borrow().is_none()
                {
                    return;
                }
                *drag.borrow_mut() = None;
                window.refresh();
                cx.stop_propagation();
            }
        });
    }
}

#[derive(Clone, PartialEq)]
struct Snapshot {
    chain: Chain,
    baseline: Chain,
    dirty: bool,
    expanded: [bool; 6],
}

#[derive(Clone)]
struct Step {
    label: String,
    snapshot: Snapshot,
}

#[derive(Clone, Default)]
struct History {
    entries: Vec<Step>,
    cursor: usize,
    merge: Option<String>,
}

impl History {
    fn detected(chain: Chain, baseline: Chain, expanded: [bool; 6]) -> Self {
        Self {
            entries: vec![Step {
                label: "Detected".to_owned(),
                snapshot: Snapshot {
                    chain,
                    baseline,
                    dirty: false,
                    expanded,
                },
            }],
            cursor: 0,
            merge: None,
        }
    }

    fn current(&self) -> Option<&Snapshot> {
        self.entries.get(self.cursor).map(|step| &step.snapshot)
    }

    fn record(&mut self, label: String, snapshot: Snapshot, merge: bool) {
        if self.current() == Some(&snapshot) {
            return;
        }
        self.entries.truncate(self.cursor.saturating_add(1));
        let replace = merge
            && self.merge.as_deref() == Some(label.as_str())
            && self.cursor > 0
            && self.cursor + 1 == self.entries.len();
        if replace {
            self.entries[self.cursor] = Step {
                label: label.clone(),
                snapshot,
            };
        } else {
            self.entries.push(Step {
                label: label.clone(),
                snapshot,
            });
            self.cursor = self.entries.len() - 1;
        }
        self.merge = merge.then_some(label);
    }
}

struct Control {
    default: Option<f32>,
    edit: Option<String>,
    active: bool,
    drag: EffectDrag,
}

#[derive(Clone)]
struct EffectDrag {
    index: usize,
    name: String,
    named: bool,
    kind: &'static str,
    score: f64,
    evidence: String,
    params: Vec<Param>,
    active: bool,
    expanded: bool,
    position: Point<Pixels>,
}

#[derive(Clone)]
struct Tab {
    path: PathBuf,
    report: Box<Report>,
    baseline: Chain,
    dirty: bool,
    history: History,
}

#[derive(Clone)]
struct TabDrag {
    path: PathBuf,
    name: String,
    position: Point<Pixels>,
}

impl TabDrag {
    fn position(mut self, position: Point<Pixels>) -> Self {
        self.position = position;
        self
    }
}

impl Render for TabDrag {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .pl(self.position.x - px(70.0))
            .pt(self.position.y - px(18.0))
            .child(
                div()
                    .w(px(140.0))
                    .h(px(36.0))
                    .px_3()
                    .rounded(theme::RADIUS)
                    .border_1()
                    .border_color(theme::ACCENT_SOFT)
                    .bg(theme::SURFACE)
                    .shadow_md()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_sm()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .flex()
                    .items_center()
                    .child(self.name.clone()),
            )
    }
}

impl EffectDrag {
    fn position(mut self, position: Point<Pixels>) -> Self {
        self.position = position;
        self
    }
}

impl Render for EffectDrag {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let width = if self.expanded {
            DEVICE_OPENED
        } else {
            DEVICE_CLOSED
        };
        div()
            .pl(self.position.x - px(width / 2.0))
            .pt(self.position.y - px(84.0))
            .child(
                div()
                    .w(px(width))
                    .h(px(180.0))
                    .p_1()
                    .rounded(theme::RADIUS)
                    .border_1()
                    .border_color(if self.active {
                        theme::ACCENT
                    } else {
                        theme::LINE
                    })
                    .bg(theme::SURFACE)
                    .shadow_md()
                    .flex()
                    .child(
                        div()
                            .w(px(DEVICE_RAIL))
                            .min_w(px(DEVICE_RAIL))
                            .flex()
                            .flex_col()
                            .gap(px(3.0))
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .w_full()
                                    .justify_center()
                                    .gap_1()
                                    .child(div().size(px(16.0)).rounded_full().bg(if self.active {
                                        theme::ACCENT
                                    } else {
                                        theme::FAINT
                                    }))
                                    .child(
                                        div()
                                            .size(px(16.0))
                                            .rounded_full()
                                            .bg(theme::TRACK)
                                            .text_base()
                                            .text_color(theme::MUTED)
                                            .flex()
                                            .items_center()
                                            .justify_center()
                                            .child(if self.expanded { "‹" } else { "›" }),
                                    ),
                            )
                            .child(
                                div()
                                    .text_size(px(10.0))
                                    .text_color(theme::MUTED)
                                    .child(format!("{:02}", self.index + 1)),
                            )
                            .child(
                                div()
                                    .w_full()
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .text_size(px(10.0))
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .text_color(theme::INK)
                                    .child(self.kind),
                            )
                            .child(
                                div()
                                    .w_full()
                                    .text_size(px(10.0))
                                    .text_color(theme::FAINT)
                                    .child(format!("{:.0}%", self.score * 100.0)),
                            )
                            .child(
                                div().relative().w_full().flex_1().min_h_0().children(
                                    self.named.then(|| {
                                        vertical(&self.name, theme::INK, -20.0, 70.0, true)
                                    }),
                                ),
                            ),
                    )
                    .when(self.expanded, |node| {
                        node.child(
                            div()
                                .w(px(288.0))
                                .min_w(px(288.0))
                                .h_full()
                                .px_1()
                                .flex()
                                .flex_col()
                                .gap_1()
                                .child(
                                    div()
                                        .h(px(32.0))
                                        .overflow_hidden()
                                        .text_size(px(10.0))
                                        .line_height(px(12.0))
                                        .text_color(theme::MUTED)
                                        .child(self.evidence.clone()),
                                )
                                .child(
                                    div().flex_1().flex().gap_1().children(
                                        self.params
                                            .iter()
                                            .enumerate()
                                            .map(|(index, param)| drag_knob(index, param)),
                                    ),
                                ),
                        )
                    }),
            )
    }
}

pub struct Muspector {
    state: State,
    tabs: Vec<Tab>,
    active: Option<usize>,
    dirty: bool,
    tab_dragging: Option<usize>,
    job: u64,
    hovers: [Hover; 1],
    glows: [usize; 1],
    alert: Option<Alert>,
    notice: usize,
    cursor: Option<f32>,
    selection: Option<(f32, f32)>,
    drag: Option<f32>,
    pan: Option<(f32, f32)>,
    zoom: f32,
    view: f32,
    scale: f32,
    overview: Option<OverviewDrag>,
    overview_anchor: f32,
    fit: bool,
    baseline: Option<Chain>,
    focus: FocusHandle,
    edit: Option<Edit>,
    cards: [usize; 6],
    folds: [usize; 6],
    expanded: [bool; 6],
    moves: [usize; 6],
    shifts: [f32; 6],
    motion: usize,
    dragging: Option<usize>,
    history: History,
    history_open: bool,
    history_motion: usize,
    inspector_drag: Rc<RefCell<Option<f32>>>,
    rail: Hover,
    rail_motion: usize,
    rail_notice: usize,
    pressure: Pressure,
    history_track: ScrollHandle,
    tracks: [ScrollHandle; 3],
}

impl Muspector {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let mut this = Self {
            state: State::Empty,
            tabs: Vec::new(),
            active: None,
            dirty: false,
            tab_dragging: None,
            job: 0,
            hovers: [Hover::Idle; 1],
            glows: [0; 1],
            alert: None,
            notice: 0,
            cursor: None,
            selection: None,
            drag: None,
            pan: None,
            zoom: 1.0,
            view: 0.0,
            scale: 1.0,
            overview: None,
            overview_anchor: 0.0,
            fit: false,
            baseline: None,
            focus: cx.focus_handle(),
            edit: None,
            cards: [0; 6],
            folds: [0; 6],
            expanded: [false; 6],
            moves: [0; 6],
            shifts: [0.0; 6],
            motion: 0,
            dragging: None,
            history: History::default(),
            history_open: false,
            history_motion: 0,
            inspector_drag: Rc::new(RefCell::new(None)),
            rail: Hover::Idle,
            rail_motion: 0,
            rail_notice: 0,
            pressure: Pressure::default(),
            history_track: ScrollHandle::new(),
            tracks: [
                ScrollHandle::new(),
                ScrollHandle::new(),
                ScrollHandle::new(),
            ],
        };
        if let Some(path) = std::env::args_os().nth(1) {
            this.start(PathBuf::from(path), cx);
        }
        Self::monitor(cx);
        this
    }

    fn monitor(cx: &mut Context<Self>) {
        cx.spawn(async move |view, cx| {
            let task = cx.background_spawn(async move {
                let mut system = System::new();
                system.refresh_cpu_usage();
                system.refresh_memory();
                system
            });
            let mut system = task.await;

            loop {
                cx.background_executor().timer(Duration::from_secs(1)).await;
                let task = cx.background_spawn(async move {
                    system.refresh_cpu_usage();
                    system.refresh_memory();
                    let total = system.total_memory();
                    let pressure = Pressure {
                        cpu: pressure_level(system.global_cpu_usage()),
                        ram: if total == 0 {
                            0
                        } else {
                            pressure_level(system.used_memory() as f32 / total as f32 * 100.0)
                        },
                    };
                    (system, pressure)
                });
                let result = task.await;
                system = result.0;
                if view
                    .update(cx, |this, cx| {
                        if this.pressure != result.1 {
                            this.pressure = result.1;
                            cx.notify();
                        }
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
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
            cx.background_executor().timer(TINT).await;
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

            cx.background_executor().timer(HOLD).await;
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

            cx.background_executor().timer(TINT).await;
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
            cx.background_executor().timer(TINT).await;
            let _ = view.update(cx, |this, cx| {
                if this.glows[slot] == glow {
                    this.hovers[slot] = if hovered { Hover::Over } else { Hover::Idle };
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn reveal_rail(&mut self, cx: &mut Context<Self>) {
        self.rail_notice = self.rail_notice.wrapping_add(1);
        let notice = self.rail_notice;

        if matches!(self.rail, Hover::Idle | Hover::Out) {
            self.rail = Hover::In;
            self.rail_motion = self.rail_motion.wrapping_add(1).max(1);
            let motion = self.rail_motion;
            cx.notify();

            cx.spawn(async move |view, cx| {
                cx.background_executor().timer(TINT).await;
                let _ = view.update(cx, move |this, cx| {
                    if this.rail_motion == motion && this.rail == Hover::In {
                        this.rail = Hover::Over;
                        cx.notify();
                    }
                });
            })
            .detach();
        }

        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(REST).await;
            let keep = view
                .update(cx, move |this, cx| {
                    if this.rail_notice != notice {
                        return None;
                    }
                    if this.inspector_drag.borrow().is_some() {
                        this.reveal_rail(cx);
                        return None;
                    }
                    this.rail = Hover::Out;
                    this.rail_motion = this.rail_motion.wrapping_add(1).max(1);
                    cx.notify();
                    Some(this.rail_motion)
                })
                .ok()
                .flatten();
            let Some(motion) = keep else {
                return;
            };

            cx.background_executor().timer(TINT).await;
            let _ = view.update(cx, move |this, cx| {
                if this.rail_motion == motion && this.rail == Hover::Out {
                    this.rail = Hover::Idle;
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

    fn reset_editor(&mut self) {
        self.cursor = None;
        self.selection = None;
        self.drag = None;
        self.pan = None;
        self.zoom = 1.0;
        self.view = 0.0;
        self.scale = 1.0;
        self.overview = None;
        self.overview_anchor = 0.0;
        self.edit = None;
        self.cards = [0; 6];
        self.folds = [0; 6];
        self.expanded = [false; 6];
        self.moves = [0; 6];
        self.shifts = [0.0; 6];
        self.dragging = None;
        *self.inspector_drag.borrow_mut() = None;
        self.rail = Hover::Idle;
        self.rail_motion = 0;
        self.rail_notice = self.rail_notice.wrapping_add(1);
        self.history_track = ScrollHandle::new();
        self.tracks = [
            ScrollHandle::new(),
            ScrollHandle::new(),
            ScrollHandle::new(),
        ];
    }

    fn sync_active(&mut self) {
        let Some(index) = self.active else {
            return;
        };
        let State::Ready(report) = &self.state else {
            return;
        };
        let Some(tab) = self.tabs.get_mut(index) else {
            return;
        };
        tab.report = report.clone();
        if let Some(baseline) = &self.baseline {
            tab.baseline = baseline.clone();
        }
        tab.dirty = self.dirty;
        tab.history = self.history.clone();
    }

    fn activate(&mut self, index: usize, cx: &mut Context<Self>) {
        if index >= self.tabs.len()
            || (self.active == Some(index) && matches!(self.state, State::Ready(_)))
        {
            return;
        }
        self.sync_active();
        let tab = self.tabs[index].clone();
        self.active = Some(index);
        self.state = State::Ready(tab.report);
        self.baseline = Some(tab.baseline);
        self.dirty = tab.dirty;
        self.history = tab.history;
        self.reset_editor();
        if let Some(snapshot) = self.history.current() {
            self.expanded = snapshot.expanded;
        } else if let State::Ready(report) = &self.state
            && let Some(index) = report.chain.effects.iter().position(|effect| effect.active)
            && index < self.expanded.len()
        {
            self.expanded[index] = true;
        }
        cx.notify();
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
        if let Some(index) = self.active
            && let Some(tab) = self.tabs.get_mut(index)
        {
            tab.dirty = true;
        }
    }

    fn snapshot(&self) -> Option<Snapshot> {
        let State::Ready(report) = &self.state else {
            return None;
        };
        Some(Snapshot {
            chain: report.chain.clone(),
            baseline: self.baseline.clone()?,
            dirty: self.dirty,
            expanded: self.expanded,
        })
    }

    fn record(&mut self, label: impl Into<String>, merge: bool) {
        let Some(snapshot) = self.snapshot() else {
            return;
        };
        self.history.record(label.into(), snapshot, merge);
        if let Some(index) = self.active
            && let Some(tab) = self.tabs.get_mut(index)
        {
            tab.history = self.history.clone();
        }
    }

    fn restore(&mut self, cursor: usize, cx: &mut Context<Self>) {
        let Some(step) = self.history.entries.get(cursor).cloned() else {
            return;
        };
        self.history.cursor = cursor;
        self.history.merge = None;
        if let State::Ready(report) = &mut self.state {
            report.chain = step.snapshot.chain.clone();
        } else {
            return;
        }
        self.baseline = Some(step.snapshot.baseline.clone());
        self.dirty = step.snapshot.dirty;
        self.expanded = step.snapshot.expanded;
        self.edit = None;
        self.cards = [0; 6];
        self.folds = [0; 6];
        self.moves = [0; 6];
        self.shifts = [0.0; 6];
        self.dragging = None;
        if let Some(index) = self.active
            && let Some(tab) = self.tabs.get_mut(index)
        {
            tab.report.chain = step.snapshot.chain;
            tab.baseline = step.snapshot.baseline;
            tab.dirty = step.snapshot.dirty;
            tab.history = self.history.clone();
        }
        cx.notify();
    }

    fn undo(&mut self, cx: &mut Context<Self>) {
        if self.history.cursor > 0 {
            self.restore(self.history.cursor - 1, cx);
        }
    }

    fn redo(&mut self, cx: &mut Context<Self>) {
        if self.history.cursor + 1 < self.history.entries.len() {
            self.restore(self.history.cursor + 1, cx);
        }
    }

    fn jump(&mut self, cursor: usize, cx: &mut Context<Self>) {
        self.restore(cursor, cx);
    }

    fn close_now(&mut self, index: usize, cx: &mut Context<Self>) {
        if index >= self.tabs.len() {
            return;
        }
        let was_active = self.active == Some(index);
        self.tabs.remove(index);
        self.tab_dragging = None;
        if self.tabs.is_empty() {
            self.active = None;
            self.state = State::Empty;
            self.baseline = None;
            self.dirty = false;
            self.history = History::default();
            self.reset_editor();
        } else if was_active {
            self.active = None;
            self.activate(index.min(self.tabs.len() - 1), cx);
            return;
        } else if let Some(active) = self.active
            && index < active
        {
            self.active = Some(active - 1);
        }
        cx.notify();
    }

    fn close(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        if index >= self.tabs.len() {
            return;
        }
        self.sync_active();
        if !self.tabs[index].dirty {
            self.close_now(index, cx);
            return;
        }
        let path = self.tabs[index].path.clone();
        let name = self.tabs[index].report.name();
        let answer = window.prompt(
            PromptLevel::Warning,
            &format!("Discard changes to {name}?"),
            Some("The adjusted effect chain has not been saved."),
            &["Discard", "Cancel"],
            cx,
        );
        cx.spawn(async move |view, cx| {
            if answer.await.unwrap_or(1) != 0 {
                return;
            }
            let _ = view.update(cx, |this, cx| {
                if let Some(index) = this.tabs.iter().position(|tab| tab.path == path) {
                    this.close_now(index, cx);
                }
            });
        })
        .detach();
    }

    fn reorder_tab(&mut self, from: usize, to: usize, cx: &mut Context<Self>) {
        if from == to || from >= self.tabs.len() || to >= self.tabs.len() {
            return;
        }
        let tab = self.tabs.remove(from);
        self.tabs.insert(to, tab);
        if let Some(active) = self.active {
            self.active = Some(if active == from {
                to
            } else if from < active && active <= to {
                active - 1
            } else if to <= active && active < from {
                active + 1
            } else {
                active
            });
        }
        self.tab_dragging = Some(to);
        cx.notify();
    }

    fn start(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        if !supported(&path) {
            self.warn("Unsupported audio format".to_owned(), cx);
            return;
        }

        if let Some(index) = self.tabs.iter().position(|tab| tab.path == path) {
            self.activate(index, cx);
            self.warn("This file is already open".to_owned(), cx);
            return;
        }
        if matches!(&self.state, State::Loading(loading) if loading == &path) {
            self.warn("This file is already being inspected".to_owned(), cx);
            return;
        }

        self.sync_active();
        let previous = match &self.state {
            State::Ready(report) => Some(report.clone()),
            State::Empty | State::Loading(_) => self
                .active
                .and_then(|index| self.tabs.get(index))
                .map(|tab| tab.report.clone()),
        };
        if self.alert.take().is_some() {
            self.notice = self.notice.wrapping_add(1);
        }
        self.job = self.job.wrapping_add(1);
        self.reset_editor();
        self.fit = false;
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
                        let expanded = report
                            .chain
                            .effects
                            .iter()
                            .position(|effect| effect.active)
                            .unwrap_or(0);
                        let report = Box::new(report);
                        let baseline = report.chain.clone();
                        let mut expanded_state = [false; 6];
                        if expanded < expanded_state.len() {
                            expanded_state[expanded] = true;
                        }
                        let history = History::detected(
                            report.chain.clone(),
                            baseline.clone(),
                            expanded_state,
                        );
                        this.tabs.push(Tab {
                            path: report.path.clone(),
                            report: report.clone(),
                            baseline: baseline.clone(),
                            dirty: false,
                            history: history.clone(),
                        });
                        this.active = Some(this.tabs.len() - 1);
                        this.dirty = false;
                        this.baseline = Some(baseline);
                        this.history = history;
                        this.state = State::Ready(report);
                        this.expanded = expanded_state;
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

    fn toggle(&mut self, effect: usize, cx: &mut Context<Self>) {
        let action = {
            let State::Ready(report) = &mut self.state else {
                return;
            };
            let Some(item) = report.chain.effects.get_mut(effect) else {
                return;
            };
            item.active = !item.active;
            format!(
                "{} {}",
                if item.active { "Enable" } else { "Bypass" },
                item.name()
            )
        };
        self.mark_dirty();
        self.record(action, false);

        let slot: usize = effect;
        self.cards[slot] = self.cards[slot].wrapping_add(1);
        let card: usize = self.cards[slot];
        cx.notify();

        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(CARD).await;
            let _ = view.update(cx, move |this, cx| {
                if this.cards[slot] == card {
                    this.cards[slot] = 0;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn adjust(&mut self, effect: usize, param: usize, direction: f64, cx: &mut Context<Self>) {
        self.edit = None;
        let action = if let State::Ready(report) = &mut self.state
            && let Some(effect) = report.chain.effects.get_mut(effect)
        {
            let effect_name = effect.name().to_owned();
            effect.params.get_mut(param).map(|param| {
                param.shift(direction);
                format!("Adjust {effect_name} {}", param.name)
            })
        } else {
            None
        };
        if let Some(action) = action {
            self.mark_dirty();
            self.record(action, true);
            cx.notify();
        }
    }

    fn reset_param(&mut self, effect: usize, param: usize, cx: &mut Context<Self>) {
        let Some((kind, model, name)) = (match &self.state {
            State::Ready(report) => report.chain.effects.get(effect).and_then(|effect| {
                effect
                    .params
                    .get(param)
                    .map(|param| (effect.kind, effect.model.clone(), param.name))
            }),
            State::Empty | State::Loading(_) => None,
        }) else {
            return;
        };
        let Some(value) = self.baseline.as_ref().and_then(|chain| {
            chain
                .effects
                .iter()
                .find(|effect| effect.kind == kind && effect.model == model)
                .and_then(|effect| effect.params.iter().find(|param| param.name == name))
                .map(|param| param.value)
        }) else {
            return;
        };
        if let State::Ready(report) = &mut self.state
            && let Some(param) = report
                .chain
                .effects
                .get_mut(effect)
                .and_then(|effect| effect.params.get_mut(param))
        {
            param.set(value);
            self.edit = None;
            self.mark_dirty();
            let effect_name = model.as_deref().unwrap_or_else(|| kind.name());
            self.record(format!("Reset {effect_name} {name}"), false);
            cx.notify();
        }
    }

    fn reset(&mut self, cx: &mut Context<Self>) {
        if let (State::Ready(report), Some(baseline)) = (&mut self.state, &self.baseline) {
            report.chain = baseline.clone();
            self.edit = None;
            self.cards = [0; 6];
            self.folds = [0; 6];
            self.expanded = [false; 6];
            self.moves = [0; 6];
            self.shifts = [0.0; 6];
            self.dragging = None;
            self.dirty = false;
            if let Some(index) = self.active
                && let Some(tab) = self.tabs.get_mut(index)
            {
                tab.dirty = false;
            }
            if let Some(index) = report.chain.effects.iter().position(|effect| effect.active)
                && index < self.expanded.len()
            {
                self.expanded[index] = true;
            }
            self.record("Reset chain", false);
            cx.notify();
        }
    }

    fn ask_reset(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(path) = self
            .active
            .and_then(|index| self.tabs.get(index))
            .map(|tab| tab.path.clone())
        else {
            return;
        };
        let answer = window.prompt(
            PromptLevel::Warning,
            "Reset the signal chain?",
            Some("All parameter, bypass, and order changes will return to the detected values."),
            &["Reset", "Cancel"],
            cx,
        );
        cx.spawn(async move |view, cx| {
            if answer.await.unwrap_or(1) != 0 {
                return;
            }
            let _ = view.update(cx, |this, cx| {
                let current = this
                    .active
                    .and_then(|index| this.tabs.get(index))
                    .map(|tab| &tab.path);
                if current == Some(&path) {
                    this.reset(cx);
                }
            });
        })
        .detach();
    }

    fn ask_rescan(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(index) = self.active else {
            return;
        };
        let Some(tab) = self.tabs.get(index) else {
            return;
        };
        let Some((start, end)) = self
            .selection
            .map(|(start, end)| ordered(start, end))
            .filter(|(start, end)| {
                let bins = tab.report.profile.points.len().saturating_sub(1).max(1);
                end - start > 0.5 / bins as f32
            })
        else {
            self.warn("Select a range to rescan".to_owned(), cx);
            return;
        };
        let path = tab.path.clone();
        let from = tab.report.duration * f64::from(start);
        let to = tab.report.duration * f64::from(end);
        let detail = format!(
            "Analyze {} – {} and replace the current detected signal chain?",
            stamp(from),
            stamp(to)
        );
        let answer = window.prompt(
            PromptLevel::Warning,
            "Rescan the selected range?",
            Some(&detail),
            &["Rescan", "Cancel"],
            cx,
        );
        cx.spawn(async move |view, cx| {
            if answer.await.unwrap_or(1) != 0 {
                return;
            }
            let _ = view.update(cx, |this, cx| this.rescan(path, (start, end), cx));
        })
        .detach();
    }

    fn rescan(&mut self, path: PathBuf, selection: (f32, f32), cx: &mut Context<Self>) {
        let Some(index) = self.tabs.iter().position(|tab| tab.path == path) else {
            return;
        };
        self.sync_active();
        let previous = self.tabs[index].clone();
        let from = previous.report.duration * f64::from(selection.0);
        let to = previous.report.duration * f64::from(selection.1);
        self.job = self.job.wrapping_add(1);
        let job = self.job;

        let task = cx.background_spawn(async move { analysis::inspect_range(&path, from, to) });
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| {
                if this.job != job {
                    return;
                }
                let Some(index) = this.tabs.iter().position(|tab| tab.path == previous.path) else {
                    return;
                };
                match result {
                    Ok(scanned) => {
                        let chain = scanned.chain;
                        let baseline = chain.clone();
                        let active = this.active == Some(index);
                        let mut expanded = [false; 6];
                        if let Some(effect) = chain.effects.iter().position(|effect| effect.active)
                            && effect < expanded.len()
                        {
                            expanded[effect] = true;
                        }

                        this.tabs[index].report.chain = chain.clone();
                        this.tabs[index].baseline = baseline.clone();
                        this.tabs[index].dirty = false;

                        if active {
                            if let State::Ready(report) = &mut this.state {
                                report.chain = chain.clone();
                            }
                            this.baseline = Some(baseline.clone());
                            this.dirty = false;
                            this.edit = None;
                            this.cards = [0; 6];
                            this.folds = [0; 6];
                            this.expanded = expanded;
                            this.moves = [0; 6];
                            this.shifts = [0.0; 6];
                            this.dragging = None;
                            this.selection = Some(selection);
                            this.record("Rescan selection", false);
                        } else {
                            let snapshot = Snapshot {
                                chain,
                                baseline,
                                dirty: false,
                                expanded,
                            };
                            this.tabs[index].history.record(
                                "Rescan selection".to_owned(),
                                snapshot,
                                false,
                            );
                        }
                        cx.notify();
                    }
                    Err(error) => {
                        this.warn(format!("{error:#}"), cx);
                    }
                }
            });
        })
        .detach();
    }

    fn expand(&mut self, effect: usize, cx: &mut Context<Self>) {
        if effect >= self.expanded.len() {
            return;
        }
        self.expanded[effect] = !self.expanded[effect];
        self.folds[effect] = self.folds[effect].wrapping_add(1).max(1);
        let fold = self.folds[effect];
        cx.notify();

        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(CARD).await;
            let _ = view.update(cx, move |this, cx| {
                if this.folds[effect] == fold {
                    this.folds[effect] = 0;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn reorder(&mut self, from: usize, to: usize, cx: &mut Context<Self>) {
        let State::Ready(report) = &mut self.state else {
            return;
        };
        if from == to || from >= report.chain.effects.len() || to >= report.chain.effects.len() {
            return;
        }
        let distance = if self.expanded[from] {
            DEVICE_OPENED + 8.0
        } else {
            DEVICE_CLOSED + 8.0
        };
        let effect = report.chain.effects.remove(from);
        report.chain.effects.insert(to, effect);
        self.mark_dirty();
        self.moves = [0; 6];
        self.shifts = [0.0; 6];
        self.motion = self.motion.wrapping_add(1).max(1);
        let motion = self.motion;
        if from < to {
            self.expanded[from..=to].rotate_left(1);
            self.cards[from..=to].rotate_left(1);
            self.folds[from..=to].rotate_left(1);
            for slot in from..to {
                self.moves[slot] = motion;
                self.shifts[slot] = distance;
            }
        } else {
            self.expanded[to..=from].rotate_right(1);
            self.cards[to..=from].rotate_right(1);
            self.folds[to..=from].rotate_right(1);
            for slot in (to + 1)..=from {
                self.moves[slot] = motion;
                self.shifts[slot] = -distance;
            }
        }
        self.dragging = Some(to);
        self.edit = None;
        self.record("Reorder effects", true);
        cx.notify();

        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(CARD).await;
            let _ = view.update(cx, move |this, cx| {
                if this.motion == motion {
                    this.moves = [0; 6];
                    this.shifts = [0.0; 6];
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn finish_drag(&mut self, cx: &mut Context<Self>) {
        if self.dragging.take().is_some() {
            self.history.merge = None;
            cx.notify();
        }
    }

    fn scroll_chain(&mut self, event: &ScrollWheelEvent, cx: &mut Context<Self>) {
        let delta = event.delta.pixel_delta(px(16.0));
        let movement = if delta.x.abs() > delta.y.abs() {
            delta.x
        } else {
            delta.y
        };
        let maximum = f32::from(self.tracks[1].max_offset().x).max(0.0);
        if maximum <= 0.5 || movement.abs() <= px(0.01) {
            return;
        }
        let current = self.tracks[1].offset();
        let x = (f32::from(current.x - movement)).clamp(-maximum, 0.0);
        self.tracks[1].set_offset(point(px(x), current.y));
        cx.stop_propagation();
        cx.notify();
    }

    fn scale(&mut self, factor: f32, anchor: f32, cx: &mut Context<Self>) {
        let old_span = 1.0 / self.zoom;
        let position = self.view + anchor.clamp(0.0, 1.0) * old_span;
        self.zoom = (self.zoom * factor).clamp(1.0, 32.0);
        let span = 1.0 / self.zoom;
        self.view = (position - anchor.clamp(0.0, 1.0) * span).clamp(0.0, 1.0 - span);
        cx.notify();
    }

    fn begin(&mut self, effect: usize, param: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(text) = (match &self.state {
            State::Ready(report) => report
                .chain
                .effects
                .get(effect)
                .and_then(|effect| effect.params.get(param))
                .map(Param::input),
            State::Empty | State::Loading(_) => None,
        }) else {
            return;
        };
        self.edit = Some(Edit {
            effect,
            param,
            text,
            fresh: true,
        });
        window.focus(&self.focus, cx);
        cx.notify();
    }

    fn key(&mut self, event: &KeyDownEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let command = event.keystroke.modifiers.platform || event.keystroke.modifiers.control;
        if command && event.keystroke.key == "z" {
            if event.keystroke.modifiers.shift {
                self.redo(cx);
            } else {
                self.undo(cx);
            }
            cx.stop_propagation();
            return;
        }
        if self.edit.is_none() {
            if command {
                match event.keystroke.key.as_str() {
                    "=" | "+" => self.scale(1.25, 0.5, cx),
                    "-" => self.scale(0.8, 0.5, cx),
                    "0" => {
                        self.zoom = 1.0;
                        self.view = 0.0;
                        cx.notify();
                    }
                    _ => {
                        cx.propagate();
                        return;
                    }
                }
                cx.stop_propagation();
                return;
            }
            cx.propagate();
            return;
        }
        cx.stop_propagation();
        match event.keystroke.key.as_str() {
            "enter" | "return" => {
                let edit = self.edit.take().expect("edit checked above");
                if let Ok(value) = edit.text.parse::<f64>() {
                    let action = if let State::Ready(report) = &mut self.state
                        && let Some(effect) = report.chain.effects.get_mut(edit.effect)
                    {
                        let effect_name = effect.name().to_owned();
                        effect.params.get_mut(edit.param).map(|param| {
                            param.set(value);
                            format!("Set {effect_name} {}", param.name)
                        })
                    } else {
                        None
                    };
                    if let Some(action) = action {
                        self.mark_dirty();
                        self.record(action, false);
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

    fn dropzone(&self, _cx: &mut Context<Self>) -> AnyElement {
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
                    .child("WAV · FLAC · MP3 · AAC · ALAC · OGG · or use +"),
            )
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
            .child(
                div()
                    .flex_1()
                    .min_h(px(270.0))
                    .w_full()
                    .flex()
                    .flex_col()
                    .child(self.overview(report, cx))
                    .child(
                        div()
                            .flex_1()
                            .min_h_0()
                            .flex()
                            .child(
                                div()
                                    .id("analysis")
                                    .flex_1()
                                    .min_w(px(500.0))
                                    .min_h_0()
                                    .overflow_y_scroll()
                                    .scrollbar_width(px(0.0))
                                    .track_scroll(&self.tracks[0])
                                    .flex()
                                    .flex_col()
                                    .child(self.profile(report, cx)),
                            )
                            .child(
                                div()
                                    .w(px(250.0))
                                    .min_w(px(250.0))
                                    .h_full()
                                    .min_h_0()
                                    .border_l_1()
                                    .border_color(theme::LINE)
                                    .bg(theme::PANEL)
                                    .flex()
                                    .flex_col()
                                    .child(spectrum_band(report))
                                    .child(
                                        div().flex_1().min_h_0().child(self.inspector(report, cx)),
                                    )
                                    .child(self.history_menu(cx)),
                            ),
                    ),
            )
            .child(self.chain(report, cx))
            .into_any_element()
    }

    fn tab(&self, index: usize, tab: &Tab, cx: &mut Context<Self>) -> AnyElement {
        let active = self.active == Some(index) && matches!(self.state, State::Ready(_));
        let dirty = if active { self.dirty } else { tab.dirty };
        let path = tab.path.clone();
        let drag = TabDrag {
            path: path.clone(),
            name: tab.report.name(),
            position: Point::default(),
        };
        div()
            .id(("tab", index))
            .h_full()
            .min_w(px(116.0))
            .max_w(px(220.0))
            .px_2()
            .border_b_2()
            .border_color(if active { theme::ACCENT } else { theme::LINE })
            .bg(if active {
                theme::SURFACE
            } else {
                theme::CANVAS
            })
            .flex()
            .items_center()
            .gap_2()
            .cursor_move()
            .hover(|node| node.bg(theme::HOVER))
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.activate(index, cx);
            }))
            .on_drag(drag.clone(), |drag: &TabDrag, position, _window, cx| {
                cx.new(|_| drag.clone().position(position))
            })
            .on_drag_move::<TabDrag>(cx.listener(
                move |this, event: &DragMoveEvent<TabDrag>, _window, cx| {
                    if !event.bounds.contains(&event.event.position) {
                        return;
                    }
                    let Some(original) = this
                        .tabs
                        .iter()
                        .position(|tab| tab.path == event.drag(cx).path)
                    else {
                        return;
                    };
                    let from = this.tab_dragging.unwrap_or(original);
                    if this.tab_dragging.is_none() {
                        this.tab_dragging = Some(from);
                    }
                    let midpoint = event.bounds.center().x;
                    let crossed = (from < index && event.event.position.x > midpoint)
                        || (from > index && event.event.position.x < midpoint);
                    if crossed {
                        this.reorder_tab(from, index, cx);
                    }
                },
            ))
            .on_drop(cx.listener(|this, _drag: &TabDrag, _window, cx| {
                this.tab_dragging = None;
                cx.notify();
            }))
            .child(
                div()
                    .id(("tab-content", index))
                    .min_w_0()
                    .flex_1()
                    .overflow_hidden()
                    .cursor_move()
                    .on_drag(drag, |drag: &TabDrag, position, _window, cx| {
                        cx.new(|_| drag.clone().position(position))
                    })
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_xs()
                            .line_height(px(12.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(if active { theme::INK } else { theme::MUTED })
                            .child(tab.report.name()),
                    )
                    .child(
                        div()
                            .mt(px(2.0))
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_size(px(9.0))
                            .line_height(px(10.0))
                            .text_color(theme::FAINT)
                            .child(format!(
                                "{} · {}",
                                tab.report.format_text(),
                                tab.report.duration_text()
                            )),
                    ),
            )
            .children(dirty.then(|| {
                div()
                    .size(px(5.0))
                    .flex_none()
                    .rounded_full()
                    .bg(theme::ACCENT)
            }))
            .child(
                div()
                    .id(("close-tab", index))
                    .size(px(20.0))
                    .flex_none()
                    .rounded(theme::RADIUS)
                    .text_sm()
                    .text_color(theme::MUTED)
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .hover(|node| node.bg(theme::TRACK).text_color(theme::INK))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_this, _event, _window, cx| cx.stop_propagation()),
                    )
                    .on_click(cx.listener(move |this, _event, window, cx| {
                        cx.stop_propagation();
                        this.close(index, window, cx);
                    }))
                    .child("×"),
            )
            .into_any_element()
    }

    fn header(&self, cx: &mut Context<Self>) -> AnyElement {
        let button = div()
            .id("add")
            .size(px(24.0))
            .rounded(theme::RADIUS)
            .text_lg()
            .text_color(theme::MUTED)
            .flex()
            .items_center()
            .justify_center()
            .cursor_pointer()
            .on_hover(cx.listener(|this, hovered: &bool, _window, cx| {
                this.hover(OPEN, *hovered, cx);
            }))
            .on_click(cx.listener(|this, _event, _window, cx| this.pick(cx)))
            .child("+");
        let button = match self.hovers[OPEN] {
            Hover::Idle => button.into_any_element(),
            Hover::Over => button
                .bg(theme::HOVER)
                .text_color(theme::INK)
                .into_any_element(),
            Hover::In => button
                .with_animation(
                    ("another-in", self.glows[OPEN]),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |node, delta| {
                        node.bg(theme::mix(theme::CANVAS, theme::HOVER, delta))
                            .text_color(theme::mix(theme::MUTED, theme::INK, delta))
                    },
                )
                .into_any_element(),
            Hover::Out => button
                .with_animation(
                    ("another-out", self.glows[OPEN]),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |node, delta| {
                        node.bg(theme::mix(theme::HOVER, theme::CANVAS, delta))
                            .text_color(theme::mix(theme::INK, theme::MUTED, delta))
                    },
                )
                .into_any_element(),
        };

        div()
            .w_full()
            .h(px(44.0))
            .flex_none()
            .px_4()
            .border_b_1()
            .border_color(theme::LINE)
            .bg(theme::CANVAS)
            .flex()
            .items_center()
            .gap_3()
            .child(
                div()
                    .size(px(22.0))
                    .rounded(theme::RADIUS)
                    .bg(theme::ACCENT)
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(theme::SURFACE)
                    .font_weight(gpui::FontWeight::BOLD)
                    .child("M"),
            )
            .child(
                div()
                    .pr_3()
                    .border_r_1()
                    .border_color(theme::LINE)
                    .text_sm()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child("Muspector"),
            )
            .child(button)
            .child(
                div()
                    .id("tabs")
                    .min_w_0()
                    .flex_1()
                    .h_full()
                    .overflow_x_scroll()
                    .flex()
                    .items_center()
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, _event, _window, cx| {
                            if this.tab_dragging.take().is_some() {
                                cx.notify();
                            }
                        }),
                    )
                    .children(
                        self.tabs
                            .iter()
                            .enumerate()
                            .map(|(index, tab)| self.tab(index, tab, cx)),
                    )
                    .children(match &self.state {
                        State::Loading(path) => Some(
                            div()
                                .h_full()
                                .min_w(px(116.0))
                                .max_w(px(220.0))
                                .px_3()
                                .border_b_2()
                                .border_color(theme::MUTED)
                                .bg(theme::SURFACE)
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_sm()
                                .text_color(theme::MUTED)
                                .flex()
                                .items_center()
                                .child(
                                    path.file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or("Inspecting…")
                                        .to_owned(),
                                ),
                        ),
                        State::Empty | State::Ready(_) => None,
                    }),
            )
            .child(
                div()
                    .w(px(126.0))
                    .min_w(px(126.0))
                    .flex_none()
                    .flex()
                    .items_center()
                    .justify_end()
                    .gap_3()
                    .child(pressure("CPU", self.pressure.cpu))
                    .child(pressure("RAM", self.pressure.ram)),
            )
            .into_any_element()
    }

    fn overview(&self, report: &Report, cx: &mut Context<Self>) -> AnyElement {
        let points = compact(&report.profile.points, 512);
        let view = self.view;
        let span = 1.0 / self.zoom;
        let bounds: Rc<RefCell<Option<Bounds<Pixels>>>> = Rc::new(RefCell::new(None));
        let capture = bounds.clone();
        let begin = bounds.clone();
        let locate = bounds.clone();

        div()
            .h(px(64.0))
            .flex_none()
            .w_full()
            .px_2()
            .py_1()
            .border_b_1()
            .border_color(theme::LINE)
            .bg(theme::PANEL)
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .h(px(18.0))
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child("Overview"),
                    )
                    .child(div().text_xs().text_color(theme::MUTED).child(format!(
                        "{:.1}× · pinch or ⌘/Ctrl-scroll to zoom",
                        self.zoom
                    ))),
            )
            .child(
                div()
                    .id("overview")
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .overflow_hidden()
                    .bg(theme::TRACK)
                    .cursor_ew_resize()
                    .child(
                        canvas(
                            move |area, _, _| {
                                *capture.borrow_mut() = Some(area);
                            },
                            move |area, _, window, _| {
                                let left = area.origin.x + px(10.0);
                                let right = area.origin.x + area.size.width - px(10.0);
                                let top = area.origin.y + px(4.0);
                                let bottom = area.origin.y + area.size.height - px(4.0);
                                let width = right - left;
                                let middle = top + (bottom - top) * 0.5;
                                let amplitude = (bottom - top) * 0.44;

                                if points.len() > 1 {
                                    let mut wave = PathBuilder::fill();
                                    for (index, item) in points.iter().enumerate() {
                                        let x = left
                                            + width * (index as f32 / (points.len() - 1) as f32);
                                        let point = point(x, middle - amplitude * item.max);
                                        if index == 0 {
                                            wave.move_to(point);
                                        } else {
                                            wave.line_to(point);
                                        }
                                    }
                                    for (index, item) in points.iter().enumerate().rev() {
                                        let x = left
                                            + width * (index as f32 / (points.len() - 1) as f32);
                                        wave.line_to(point(x, middle - amplitude * item.min));
                                    }
                                    wave.close();
                                    if let Ok(path) = wave.build() {
                                        window.paint_path(path, theme::ACCENT_SOFT);
                                    }
                                }

                                let x1 = left + width * view;
                                let x2 = left + width * (view + span).min(1.0);
                                let mut shade = PathBuilder::fill();
                                shade.add_polygon(
                                    &[
                                        point(x1, top),
                                        point(x2, top),
                                        point(x2, bottom),
                                        point(x1, bottom),
                                    ],
                                    true,
                                );
                                if let Ok(path) = shade.build() {
                                    let mut color = theme::ACCENT_SOFT;
                                    color.a = 0.42;
                                    window.paint_path(path, color);
                                }

                                let mut viewport = PathBuilder::stroke(px(1.5));
                                viewport.move_to(point(x1, top));
                                viewport.line_to(point(x2, top));
                                viewport.line_to(point(x2, bottom));
                                viewport.line_to(point(x1, bottom));
                                viewport.close();
                                if let Ok(path) = viewport.build() {
                                    window.paint_path(path, theme::ACCENT);
                                }

                                let center = top + (bottom - top) * 0.5;
                                let inset = px(5.0);
                                for (edge, direction) in [(x1, 1.0_f32), (x2, -1.0_f32)] {
                                    let mut grip = PathBuilder::stroke(px(2.0));
                                    grip.move_to(point(edge, top));
                                    grip.line_to(point(edge, bottom));
                                    if let Ok(path) = grip.build() {
                                        window.paint_path(path, theme::ACCENT_HOVER);
                                    }

                                    let tip = edge + inset * direction;
                                    let back = edge + inset * 2.0 * direction;
                                    let mut chevron = PathBuilder::stroke(px(1.5));
                                    chevron.move_to(point(back, center - px(4.0)));
                                    chevron.line_to(point(tip, center));
                                    chevron.line_to(point(back, center + px(4.0)));
                                    if let Ok(path) = chevron.build() {
                                        window.paint_path(path, theme::INK);
                                    }
                                }
                            },
                        )
                        .size_full(),
                    )
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                            let Some(area) = *begin.borrow() else {
                                return;
                            };
                            let position = horizontal(event.position.x, area);
                            let span = 1.0 / this.zoom;
                            let left = this.view;
                            let right = this.view + span;
                            let threshold = 12.0 / f32::from(area.size.width).max(1.0);
                            if (position - left).abs() <= threshold {
                                this.overview = Some(OverviewDrag::Left);
                                this.overview_anchor = right;
                            } else if (position - right).abs() <= threshold {
                                this.overview = Some(OverviewDrag::Right);
                                this.overview_anchor = left;
                            } else {
                                this.overview = Some(OverviewDrag::Pan);
                                if position < left || position > right {
                                    this.view = (position - span * 0.5).clamp(0.0, 1.0 - span);
                                    this.overview_anchor = span * 0.5;
                                } else {
                                    this.overview_anchor = position - left;
                                }
                            }
                            cx.notify();
                        }),
                    )
                    .on_mouse_move(cx.listener(move |this, event: &MouseMoveEvent, _, cx| {
                        let Some(drag) = this.overview else {
                            return;
                        };
                        let Some(area) = *locate.borrow() else {
                            return;
                        };
                        let position = horizontal(event.position.x, area);
                        match drag {
                            OverviewDrag::Pan => {
                                let span = 1.0 / this.zoom;
                                this.view =
                                    (position - this.overview_anchor).clamp(0.0, 1.0 - span);
                            }
                            OverviewDrag::Left => {
                                let right = this.overview_anchor;
                                let left = position.clamp(0.0, right - 1.0 / 32.0);
                                let span = right - left;
                                this.view = left;
                                this.zoom = 1.0 / span;
                            }
                            OverviewDrag::Right => {
                                let left = this.overview_anchor;
                                let right = position.clamp(left + 1.0 / 32.0, 1.0);
                                let span = right - left;
                                this.view = left;
                                this.zoom = 1.0 / span;
                            }
                        }
                        cx.notify();
                    }))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, _event: &MouseUpEvent, _, _cx| {
                            this.overview = None;
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, _event: &MouseUpEvent, _, _cx| {
                            this.overview = None;
                        }),
                    ),
            )
            .into_any_element()
    }

    fn inspector(&self, report: &Report, cx: &mut Context<Self>) -> AnyElement {
        let selected = selection_stats(report, self.selection);
        let peak = selected.map_or(report.peak, |selection| selection.2);
        let rms = selected.map_or(report.rms, |selection| selection.3);
        let crest = selected.map_or(report.crest, |selection| selection.4);
        let loudness = selected.map_or(report.loudness, |selection| selection.5);
        let rail = div()
            .absolute()
            .top_0()
            .right_0()
            .bottom_0()
            .left_0()
            .child(InspectorScrollbar {
                handle: self.tracks[2].clone(),
                drag: self.inspector_drag.clone(),
            });
        let rail = match self.rail {
            Hover::Idle => None,
            Hover::Over => Some(rail.into_any_element()),
            Hover::In => Some(
                rail.with_animation(
                    ("inspector-rail", self.rail_motion),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |rail, delta| rail.opacity(delta),
                )
                .into_any_element(),
            ),
            Hover::Out => Some(
                rail.with_animation(
                    ("inspector-rail", self.rail_motion),
                    Animation::new(TINT).with_easing(ease_in_out),
                    |rail, delta| rail.opacity(1.0 - delta),
                )
                .into_any_element(),
            ),
        };

        div()
            .id("inspector")
            .w_full()
            .flex_1()
            .min_h_0()
            .relative()
            .overflow_y_scroll()
            .scrollbar_width(px(0.0))
            .track_scroll(&self.tracks[2])
            .on_scroll_wheel(cx.listener(|this, _event: &ScrollWheelEvent, _, cx| {
                this.reveal_rail(cx);
            }))
            .bg(theme::PANEL)
            .p_2()
            .flex()
            .flex_col()
            .gap_2()
            .child(section(
                "Inspector",
                if selected.is_some() {
                    "Selected range"
                } else {
                    "Full-file metrics"
                },
            ))
            .children(selected.map(|selection| {
                readout(
                    "Selection",
                    format!(
                        "{} – {}  ·  {}",
                        stamp(selection.0),
                        stamp(selection.1),
                        span(selection.1 - selection.0)
                    ),
                )
            }))
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(metric("Peak", format!("{peak:.1} dB")))
                    .child(metric("RMS", format!("{rms:.1} dB"))),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(metric("Crest", format!("{crest:.1} dB")))
                    .child(metric("Clips", report.clips.to_string())),
            )
            .child(readout("Loudness", format!("{loudness:.1} LUFS")))
            .child(readout("Centroid", format!("{:.0} Hz", report.centroid)))
            .child(readout("Rolloff", format!("{:.0} Hz", report.rolloff)))
            .child(readout("Low", format!("{:.1} dB", report.low)))
            .child(readout("Mid", format!("{:.1} dB", report.mid)))
            .child(readout("High", format!("{:.1} dB", report.high)))
            .children(rail)
            .into_any_element()
    }

    fn toggle_history(&mut self, cx: &mut Context<Self>) {
        self.history_open = !self.history_open;
        self.history_motion = self.history_motion.wrapping_add(1).max(1);
        let motion = self.history_motion;
        cx.notify();

        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(CARD).await;
            let _ = view.update(cx, move |this, cx| {
                if this.history_motion == motion {
                    this.history_motion = 0;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn history_menu(&self, cx: &mut Context<Self>) -> AnyElement {
        let current = self.history.cursor;
        let total = self.history.entries.len().saturating_sub(1);
        let can_undo = current > 0;
        let can_redo = current + 1 < self.history.entries.len();
        let rows = self
            .history
            .entries
            .iter()
            .enumerate()
            .rev()
            .map(|(index, step)| {
                div()
                    .id(("history-step", index))
                    .h(px(26.0))
                    .flex_none()
                    .px_2()
                    .bg(if index == current {
                        theme::ACCENT_SOFT
                    } else {
                        theme::PANEL
                    })
                    .text_size(px(10.0))
                    .text_color(if index == current {
                        theme::INK
                    } else {
                        theme::MUTED
                    })
                    .flex()
                    .items_center()
                    .gap_2()
                    .cursor_pointer()
                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                    .on_click(cx.listener(move |this, _event, _window, cx| {
                        this.jump(index, cx);
                    }))
                    .child(
                        div()
                            .w(px(20.0))
                            .flex_none()
                            .text_color(theme::FAINT)
                            .child(format!("{index:02}")),
                    )
                    .child(
                        div()
                            .min_w_0()
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .child(step.label.clone()),
                    )
            })
            .collect::<Vec<_>>();

        let list = div()
            .id("history-list")
            .h(px(156.0))
            .flex_none()
            .overflow_y_scroll()
            .scrollbar_width(px(0.0))
            .track_scroll(&self.history_track)
            .border_b_1()
            .border_color(theme::LINE)
            .bg(theme::PANEL)
            .flex()
            .flex_col()
            .children(rows);

        let header = div()
            .id("history")
            .h(px(32.0))
            .flex_none()
            .px_2()
            .text_xs()
            .flex()
            .items_center()
            .cursor_pointer()
            .on_click(cx.listener(|this, _event, _window, cx| {
                this.toggle_history(cx);
            }))
            .child(
                div()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child("History"),
            )
            .child(
                div()
                    .ml_2()
                    .text_size(px(9.0))
                    .text_color(theme::FAINT)
                    .child(format!("{current}/{total}")),
            )
            .child(div().flex_1())
            .child(
                div()
                    .id("undo")
                    .size(px(22.0))
                    .rounded(theme::RADIUS)
                    .opacity(if can_undo { 1.0 } else { 0.35 })
                    .text_sm()
                    .text_color(theme::MUTED)
                    .flex()
                    .items_center()
                    .justify_center()
                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_this, _event, _window, cx| cx.stop_propagation()),
                    )
                    .on_click(cx.listener(|this, _event, _window, cx| {
                        cx.stop_propagation();
                        this.undo(cx);
                    }))
                    .child(svg().size(px(13.0)).path("icons/undo.svg")),
            )
            .child(
                div()
                    .id("redo")
                    .size(px(22.0))
                    .rounded(theme::RADIUS)
                    .opacity(if can_redo { 1.0 } else { 0.35 })
                    .text_sm()
                    .text_color(theme::MUTED)
                    .flex()
                    .items_center()
                    .justify_center()
                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_this, _event, _window, cx| cx.stop_propagation()),
                    )
                    .on_click(cx.listener(|this, _event, _window, cx| {
                        cx.stop_propagation();
                        this.redo(cx);
                    }))
                    .child(svg().size(px(13.0)).path("icons/redo.svg")),
            )
            .child(
                div()
                    .ml_1()
                    .size(px(14.0))
                    .text_color(theme::FAINT)
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(svg().size(px(12.0)).path(if self.history_open {
                        "icons/chevron-down.svg"
                    } else {
                        "icons/chevron-up.svg"
                    })),
            );

        let list = div().w_full().flex_none().overflow_hidden().child(list);
        let list = if self.history_motion == 0 {
            list.h(px(if self.history_open { 156.0 } else { 0.0 }))
                .opacity(if self.history_open { 1.0 } else { 0.0 })
                .into_any_element()
        } else {
            let opening = self.history_open;
            list.with_animation(
                ("history-menu", self.history_motion),
                Animation::new(CARD).with_easing(ease_in_out),
                move |list, delta| {
                    let progress = if opening { delta } else { 1.0 - delta };
                    list.h(px(156.0 * progress)).opacity(progress)
                },
            )
            .into_any_element()
        };

        div()
            .flex_none()
            .border_t_1()
            .border_color(theme::LINE)
            .bg(theme::PANEL)
            .flex()
            .flex_col()
            .child(list)
            .child(header)
            .into_any_element()
    }

    fn profile(&self, report: &Report, cx: &mut Context<Self>) -> AnyElement {
        let points = report.profile.points.clone();
        let duration = report.duration;
        let view = self.view;
        let scale = self.scale;
        let detailed = self.zoom > 1.01;
        let visible = 1.0 / self.zoom;
        let last = points.len().saturating_sub(1);
        let first = (view * last as f32).floor() as usize;
        let final_index = ((view + visible).min(1.0) * last as f32).ceil() as usize;
        let minimum = 0.5 / last.max(1) as f32;
        let display = if points.is_empty() {
            Vec::new()
        } else {
            compact(&points[first.min(last)..=final_index.min(last)], 2_048)
        };
        let cursor = self.cursor;
        let selection = self.selection;
        let bounds: Rc<RefCell<Option<Bounds<Pixels>>>> = Rc::new(RefCell::new(None));
        let capture = bounds.clone();
        let locate = bounds.clone();
        let begin = bounds.clone();
        let pan_begin = bounds.clone();
        let wheel = bounds.clone();
        let pinch = bounds.clone();

        let detail = selection
            .filter(|(start, end)| (end - start).abs() > minimum)
            .map(|(start, end)| {
                let (start, end) = ordered(start, end);
                format!(
                    "Selected {} – {}  ·  {}",
                    stamp(duration * f64::from(start)),
                    stamp(duration * f64::from(end)),
                    span(duration * f64::from(end - start))
                )
            })
            .or_else(|| {
                cursor.and_then(|position| {
                    let index = (position * points.len().saturating_sub(1) as f32).round() as usize;
                    points.get(index).map(|point| {
                        let peak = f64::from(point.min.abs().max(point.max.abs()));
                        format!(
                            "{}  ·  Peak {:.1} dB  ·  RMS {:.1} dB  ·  {:.1} LUFS",
                            stamp(duration * f64::from(position)),
                            level(peak),
                            point.level,
                            point.loudness
                        )
                    })
                })
            })
            .unwrap_or_else(|| "Drag to select · hover for values".to_owned());

        let plot = div()
            .id("profile")
            .flex_1()
            .min_w_0()
            .min_h(px(110.0))
            .relative()
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

                        if let Some((start, end)) = selection {
                            let (start, end) = ordered(start, end);
                            let x1 = left + width * ((start - view) / visible).clamp(0.0, 1.0);
                            let x2 = left + width * ((end - view) / visible).clamp(0.0, 1.0);
                            let mut fill = PathBuilder::fill();
                            fill.add_polygon(
                                &[
                                    point(x1, top),
                                    point(x2, top),
                                    point(x2, bottom),
                                    point(x1, bottom),
                                ],
                                true,
                            );
                            if let Ok(path) = fill.build() {
                                let mut color = theme::ACCENT_SOFT;
                                color.a = 0.72;
                                window.paint_path(path, color);
                            }
                            for x in [x1, x2] {
                                let mut edge = PathBuilder::stroke(px(1.5));
                                edge.move_to(point(x, top));
                                edge.line_to(point(x, bottom));
                                if let Ok(path) = edge.build() {
                                    window.paint_path(path, theme::ACCENT);
                                }
                            }
                        }

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

                        let mut baseline = PathBuilder::stroke(px(1.0));
                        baseline.move_to(point(left, bottom));
                        baseline.line_to(point(right, bottom));
                        if let Ok(path) = baseline.build() {
                            let mut color = theme::MUTED;
                            color.a = 0.42;
                            window.paint_path(path, color);
                        }

                        if display.len() > 1 {
                            let x = |index: usize| {
                                left + width * (index as f32 / (display.len() - 1) as f32)
                            };
                            if detailed {
                                let mut wave = PathBuilder::stroke(px(1.0));
                                for (index, item) in display.iter().enumerate() {
                                    let x = x(index);
                                    wave.move_to(point(
                                        x,
                                        middle - amplitude * (item.max * scale).clamp(-1.0, 1.0),
                                    ));
                                    wave.line_to(point(
                                        x,
                                        middle - amplitude * (item.min * scale).clamp(-1.0, 1.0),
                                    ));
                                }
                                if let Ok(path) = wave.build() {
                                    window.paint_path(path, theme::ACCENT);
                                }
                            } else {
                                let mut wave = PathBuilder::fill();
                                for (index, item) in display.iter().enumerate() {
                                    let point = point(
                                        x(index),
                                        middle - amplitude * (item.max * scale).clamp(-1.0, 1.0),
                                    );
                                    if index == 0 {
                                        wave.move_to(point);
                                    } else {
                                        wave.line_to(point);
                                    }
                                }
                                for (index, item) in display.iter().enumerate().rev() {
                                    wave.line_to(point(
                                        x(index),
                                        middle - amplitude * (item.min * scale).clamp(-1.0, 1.0),
                                    ));
                                }
                                wave.close();
                                if let Ok(path) = wave.build() {
                                    window.paint_path(path, theme::ACCENT_SOFT);
                                }

                                let mut peak = PathBuilder::stroke(px(1.0));
                                for (index, item) in display.iter().enumerate() {
                                    let sample = item.min.abs().max(item.max.abs());
                                    let point = point(
                                        x(index),
                                        middle - amplitude * (sample * scale).clamp(0.0, 1.0),
                                    );
                                    if index == 0 {
                                        peak.move_to(point);
                                    } else {
                                        peak.line_to(point);
                                    }
                                }
                                if let Ok(path) = peak.build() {
                                    window.paint_path(path, theme::ACCENT);
                                }
                            }

                            let mut rms = PathBuilder::stroke(px(1.5));
                            for (index, item) in display.iter().enumerate() {
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

                            let mut loudness = PathBuilder::stroke(px(1.0));
                            for (index, item) in display.iter().enumerate() {
                                let value = ((item.loudness + 72.0) / 72.0).clamp(0.0, 1.0) as f32;
                                let point = point(x(index), bottom - (bottom - level_top) * value);
                                if index == 0 {
                                    loudness.move_to(point);
                                } else {
                                    loudness.line_to(point);
                                }
                            }
                            if let Ok(path) = loudness.build() {
                                let mut color = theme::INK;
                                color.a = 0.72;
                                window.paint_path(path, color);
                            }
                        }

                        if let Some(position) = cursor {
                            let x = left + width * ((position - view) / visible).clamp(0.0, 1.0);
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
                    .right_3()
                    .bottom_1()
                    .h(px(16.0))
                    .flex()
                    .items_end()
                    .justify_between()
                    .text_size(px(8.0))
                    .text_color(theme::FAINT)
                    .children((0..=8).map(|index| {
                        let ratio = index as f64 / 8.0;
                        let position = f64::from(view) + f64::from(visible) * ratio;
                        div()
                            .h_full()
                            .flex()
                            .flex_col()
                            .items_center()
                            .justify_between()
                            .child(div().w(px(1.0)).h(px(4.0)).bg(theme::LINE))
                            .child(stamp(duration * position.min(1.0)))
                    })),
            )
            .child(
                div()
                    .absolute()
                    .left_3()
                    .top(relative(0.62))
                    .flex()
                    .gap_2()
                    .text_size(px(9.0))
                    .child(div().text_color(theme::ACCENT_HOVER).child("RMS"))
                    .child(div().text_color(theme::INK).child("LUFS")),
            )
            .on_mouse_move(cx.listener(move |this, event: &MouseMoveEvent, _, cx| {
                let Some(area) = *locate.borrow() else {
                    return;
                };
                let local = horizontal(event.position.x, area);
                if let Some((origin, initial)) = this.pan {
                    let span = 1.0 / this.zoom;
                    this.view = (initial - (local - origin) * span).clamp(0.0, 1.0 - span);
                    cx.notify();
                    return;
                }
                let position = this.view + local / this.zoom;
                if let Some(start) = this.drag {
                    this.selection = Some((start, position));
                    this.cursor = Some(position);
                    cx.notify();
                    return;
                }
                if this
                    .cursor
                    .is_none_or(|current| (current - position).abs() > minimum)
                {
                    this.cursor = Some(position);
                    cx.notify();
                }
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                    let Some(area) = *begin.borrow() else {
                        return;
                    };
                    let position = this.view + horizontal(event.position.x, area) / this.zoom;
                    this.drag = Some(position);
                    this.selection = Some((position, position));
                    this.cursor = Some(position);
                    cx.notify();
                }),
            )
            .on_mouse_down(
                MouseButton::Middle,
                cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                    let Some(area) = *pan_begin.borrow() else {
                        return;
                    };
                    this.pan = Some((horizontal(event.position.x, area), this.view));
                    cx.stop_propagation();
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(move |this, _: &MouseUpEvent, _, cx| {
                    this.drag = None;
                    if this
                        .selection
                        .is_some_and(|(start, end)| (end - start).abs() <= minimum)
                    {
                        this.selection = None;
                    }
                    cx.notify();
                }),
            )
            .on_mouse_up(
                MouseButton::Middle,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    if this.pan.take().is_some() {
                        cx.stop_propagation();
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up_out(
                MouseButton::Middle,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    if this.pan.take().is_some() {
                        cx.stop_propagation();
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    this.drag = None;
                    cx.notify();
                }),
            )
            .on_scroll_wheel(cx.listener(move |this, event: &ScrollWheelEvent, _, cx| {
                let Some(area) = *wheel.borrow() else {
                    return;
                };
                let delta = event.delta.pixel_delta(px(16.0));
                let movement = if delta.y.abs() >= delta.x.abs() {
                    f32::from(delta.y)
                } else {
                    f32::from(delta.x)
                };
                if movement.abs() <= 0.01 {
                    return;
                }
                let width = f32::from(area.size.width).max(1.0);
                let local = horizontal(event.position.x, area);
                if event.modifiers.alt {
                    const SCALES: [f32; 7] = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];
                    let current = SCALES
                        .iter()
                        .enumerate()
                        .min_by(|(_, left), (_, right)| {
                            (*left - this.scale)
                                .abs()
                                .total_cmp(&(*right - this.scale).abs())
                        })
                        .map_or(0, |(index, _)| index);
                    let next = if movement > 0.0 {
                        (current + 1).min(SCALES.len() - 1)
                    } else {
                        current.saturating_sub(1)
                    };
                    this.scale = SCALES[next];
                } else if event.modifiers.platform || event.modifiers.control {
                    let factor = if event.delta.precise() {
                        let amount = 1.0 + movement.abs() * 0.01;
                        if movement > 0.0 { amount } else { 1.0 / amount }
                    } else if movement > 0.0 {
                        1.18
                    } else {
                        1.0 / 1.18
                    };
                    this.scale(factor, local, cx);
                } else if this.zoom > 1.0 {
                    let span = 1.0 / this.zoom;
                    this.view = (this.view - movement / width * span * 3.0).clamp(0.0, 1.0 - span);
                } else {
                    return;
                }
                cx.stop_propagation();
                cx.notify();
            }))
            .on_pinch(cx.listener(move |this, event: &PinchEvent, _, cx| {
                let Some(area) = *pinch.borrow() else {
                    return;
                };
                if event.delta.abs() <= f32::EPSILON {
                    return;
                }
                let anchor = horizontal(event.position.x, area);
                this.scale((1.0 + event.delta).max(0.05), anchor, cx);
                cx.stop_propagation();
            }))
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !hovered && this.cursor.take().is_some() {
                    cx.notify();
                }
            }));

        let mut marks = Vec::new();
        for (index, db) in [
            0.0_f32, -1.0, -2.0, -3.0, -4.0, -6.0, -9.0, -12.0, -18.0, -24.0,
        ]
        .into_iter()
        .enumerate()
        {
            let level = 10.0_f32.powf(db / 20.0) * scale;
            if !(0.075..=1.001).contains(&level) {
                continue;
            }
            for (side, direction) in [(0_usize, 1.0_f32), (1, -1.0)] {
                let top = 0.31 - 0.27 * level * direction;
                marks.push(
                    div()
                        .id(("level", index * 2 + side))
                        .absolute()
                        .left_1()
                        .top(relative(top))
                        .text_color(theme::FAINT)
                        .child(format!("{db:.0}"))
                        .into_any_element(),
                );
            }
        }
        marks.push(
            div()
                .id("level-infinity")
                .absolute()
                .left_1()
                .top(relative(0.31))
                .text_color(theme::FAINT)
                .child("−∞")
                .into_any_element(),
        );

        let ruler = div()
            .w(px(24.0))
            .min_w(px(24.0))
            .h_full()
            .relative()
            .border_l_1()
            .border_color(theme::LINE)
            .text_size(px(9.0))
            .text_color(theme::FAINT)
            .child(
                div()
                    .absolute()
                    .left_1()
                    .top_0()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child("dB"),
            )
            .children(marks);

        let chart = div()
            .flex_1()
            .min_h(px(110.0))
            .w_full()
            .overflow_hidden()
            .bg(theme::TRACK)
            .flex()
            .child(plot)
            .child(ruler);

        div()
            .flex_1()
            .min_h(px(170.0))
            .w_full()
            .p_2()
            .bg(theme::SURFACE)
            .flex()
            .flex_col()
            .gap_2()
            .child(
                div()
                    .flex()
                    .items_start()
                    .justify_between()
                    .child(section(
                        "Workspace",
                        "Waveform · RMS · pinch or ⌘/Ctrl-scroll zoom · Alt-scroll scale",
                    ))
                    .child(div().text_xs().text_color(theme::MUTED).child(detail)),
            )
            .child(chart)
            .into_any_element()
    }

    fn chain(&self, report: &Report, cx: &mut Context<Self>) -> AnyElement {
        div()
            .id("chain")
            .h(px(224.0))
            .flex_none()
            .w_full()
            .relative()
            .border_t_1()
            .border_color(theme::LINE)
            .bg(theme::PANEL)
            .overflow_hidden()
            .flex()
            .flex_col()
            .child(
                div()
                    .h(px(36.0))
                    .flex_none()
                    .px_2()
                    .border_b_1()
                    .border_color(theme::LINE)
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .child("Signal Chain"),
                    )
                    .child(
                        div()
                            .ml_1()
                            .text_xs()
                            .text_color(theme::MUTED)
                            .child(format!(
                                "· {:.0}% {} candidate",
                                report.chain.score * 100.0,
                                if report.chain.blind {
                                    "hybrid"
                                } else {
                                    "heuristic"
                                }
                            )),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .items_center()
                            .justify_end()
                            .gap_1()
                            .child(
                                div()
                                    .id("reset")
                                    .h(px(24.0))
                                    .px_2()
                                    .rounded(theme::RADIUS)
                                    .text_xs()
                                    .text_color(theme::MUTED)
                                    .flex()
                                    .items_center()
                                    .cursor_pointer()
                                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                                    .on_click(cx.listener(|this, _event, window, cx| {
                                        this.ask_reset(window, cx)
                                    }))
                                    .child("Reset"),
                            )
                            .child(
                                div()
                                    .id("rescan")
                                    .h(px(24.0))
                                    .px_2()
                                    .rounded(theme::RADIUS)
                                    .text_xs()
                                    .text_color(theme::MUTED)
                                    .flex()
                                    .items_center()
                                    .cursor_pointer()
                                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                                    .on_click(cx.listener(|this, _event, window, cx| {
                                        this.ask_rescan(window, cx)
                                    }))
                                    .child("Rescan"),
                            ),
                    ),
            )
            .child(
                div()
                    .id("devices")
                    .flex_1()
                    .min_h_0()
                    .w_full()
                    .overflow_x_scroll()
                    .scrollbar_width(px(8.0))
                    .track_scroll(&self.tracks[1])
                    .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, _window, cx| {
                        this.scroll_chain(event, cx);
                    }))
                    .on_drop(cx.listener(|this, _drag: &EffectDrag, _window, cx| {
                        this.finish_drag(cx);
                    }))
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, _event, _window, cx| this.finish_drag(cx)),
                    )
                    .px_2()
                    .pt_2()
                    .pb_0()
                    .flex()
                    .gap_2()
                    .children(
                        report
                            .chain
                            .effects
                            .iter()
                            .enumerate()
                            .map(|(index, effect)| self.effect(index, effect, cx)),
                    ),
            )
            .child(hbar(&self.tracks[1]))
            .into_any_element()
    }

    fn effect(&self, index: usize, effect: &Effect, cx: &mut Context<Self>) -> AnyElement {
        let active = effect.active;
        let expanded = self.expanded[index];
        let width = if expanded {
            DEVICE_OPENED
        } else {
            DEVICE_CLOSED
        };
        let drag = EffectDrag {
            index,
            name: effect.name().to_owned(),
            named: effect.model.is_some(),
            kind: effect.kind.name(),
            score: effect.score,
            evidence: effect.evidence.clone(),
            params: effect.params.clone(),
            active,
            expanded,
            position: Point::default(),
        };
        let details = div()
            .id(("details", index))
            .w(px(288.0))
            .min_w(px(288.0))
            .h_full()
            .px_1()
            .cursor_move()
            .on_drag(drag.clone(), |drag: &EffectDrag, position, _window, cx| {
                cx.new(|_| drag.clone().position(position))
            })
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .id(("evidence", index))
                    .h(px(32.0))
                    .min_h(px(32.0))
                    .overflow_hidden()
                    .text_size(px(10.0))
                    .line_height(px(12.0))
                    .text_color(theme::MUTED)
                    .cursor_move()
                    .on_drag(drag.clone(), |drag: &EffectDrag, position, _window, cx| {
                        cx.new(|_| drag.clone().position(position))
                    })
                    .child(effect.evidence.clone()),
            )
            .child(div().w_full().flex_1().min_h_0().flex().gap_1().children(
                effect.params.iter().enumerate().map(|(param, value)| {
                    let default = self.baseline.as_ref().and_then(|chain| {
                        chain
                            .effects
                            .iter()
                            .find(|candidate| {
                                candidate.kind == effect.kind && candidate.model == effect.model
                            })
                            .and_then(|candidate| {
                                candidate
                                    .params
                                    .iter()
                                    .find(|candidate| candidate.name == value.name)
                            })
                            .map(Param::normal)
                    });
                    let edit = self
                        .edit
                        .as_ref()
                        .filter(|edit| edit.effect == index && edit.param == param)
                        .map(|edit| edit.text.clone());
                    knob(
                        index,
                        param,
                        value,
                        Control {
                            default,
                            edit,
                            active,
                            drag: drag.clone(),
                        },
                        cx,
                    )
                }),
            ));
        let details = if self.folds[index] == 0 {
            details
                .opacity(if expanded { 1.0 } else { 0.0 })
                .into_any_element()
        } else {
            let start = if expanded { 0.0 } else { 1.0 };
            let end = if expanded { 1.0 } else { 0.0 };
            let token =
                self.job.wrapping_mul(10_000) + index as u64 * 100 + self.folds[index] as u64;
            details
                .with_animation(
                    ("details", token),
                    Animation::new(CARD).with_easing(ease_in_out),
                    move |details, delta| details.opacity(start + (end - start) * delta),
                )
                .into_any_element()
        };
        let card = div()
            .id(("device", index))
            .flex_none()
            .w(px(width))
            .min_w(px(width))
            .h_full()
            .p_1()
            .rounded(theme::RADIUS)
            .border_1()
            .border_color(if active {
                theme::ACCENT_SOFT
            } else {
                theme::LINE
            })
            .bg(theme::SURFACE)
            .flex()
            .flex_row()
            .overflow_hidden()
            .cursor_move()
            .on_drag_move::<EffectDrag>(cx.listener(
                move |this, event: &DragMoveEvent<EffectDrag>, _window, cx| {
                    if !event.bounds.contains(&event.event.position) {
                        return;
                    }
                    let original = event.drag(cx).index;
                    let from = this.dragging.unwrap_or(original);
                    if this.dragging.is_none() {
                        this.dragging = Some(from);
                        cx.notify();
                    }
                    let midpoint = event.bounds.center().x;
                    let crossed = (from < index && event.event.position.x > midpoint)
                        || (from > index && event.event.position.x < midpoint);
                    if crossed {
                        this.reorder(from, index, cx);
                    }
                },
            ))
            .child(
                div()
                    .id(("drag", index))
                    .w(px(DEVICE_RAIL))
                    .min_w(px(DEVICE_RAIL))
                    .h_full()
                    .flex()
                    .flex_col()
                    .gap(px(3.0))
                    .cursor_move()
                    .on_drag(drag.clone(), |drag: &EffectDrag, position, _window, cx| {
                        cx.new(|_| drag.clone().position(position))
                    })
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .w_full()
                            .justify_center()
                            .gap_1()
                            .child(
                                div()
                                    .id(("toggle", index))
                                    .size(px(16.0))
                                    .rounded_full()
                                    .bg(if active { theme::ACCENT } else { theme::FAINT })
                                    .cursor_pointer()
                                    .hover(|node| node.shadow_sm())
                                    .on_click(cx.listener(move |this, _event, _window, cx| {
                                        this.toggle(index, cx);
                                    })),
                            )
                            .child(
                                div()
                                    .id(("expand", index))
                                    .size(px(16.0))
                                    .rounded_full()
                                    .bg(if expanded {
                                        theme::ACCENT_SOFT
                                    } else {
                                        theme::TRACK
                                    })
                                    .text_base()
                                    .text_color(if expanded {
                                        theme::ACCENT
                                    } else {
                                        theme::MUTED
                                    })
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .cursor_pointer()
                                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                                    .on_click(cx.listener(move |this, _event, _window, cx| {
                                        this.expand(index, cx);
                                    }))
                                    .child(if expanded { "‹" } else { "›" }),
                            ),
                    )
                    .child(
                        div()
                            .text_size(px(10.0))
                            .text_color(theme::MUTED)
                            .child(format!("{:02}", index + 1)),
                    )
                    .child(
                        div()
                            .w_full()
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_size(px(10.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(theme::INK)
                            .child(effect.kind.name()),
                    )
                    .child(
                        div()
                            .w_full()
                            .text_size(px(10.0))
                            .text_color(theme::FAINT)
                            .child(format!("{:.0}%", effect.score * 100.0)),
                    )
                    .child(
                        div().w_full().flex_1().min_h_0().relative().children(
                            effect
                                .model
                                .as_deref()
                                .map(|name| vertical(name, theme::INK, -20.0, 70.0, true)),
                        ),
                    ),
            )
            .child(details);
        let card = if self.folds[index] == 0 {
            card.into_any_element()
        } else {
            let start = if expanded {
                DEVICE_CLOSED
            } else {
                DEVICE_OPENED
            };
            let end = width;
            let token =
                self.job.wrapping_mul(10_000) + index as u64 * 100 + self.folds[index] as u64;
            card.with_animation(
                ("fold", token),
                Animation::new(CARD).with_easing(ease_in_out),
                move |card, delta| {
                    let width = start + (end - start) * delta;
                    card.w(px(width)).min_w(px(width))
                },
            )
            .into_any_element()
        };
        let end = if active { 1.0 } else { 0.58 };
        let opacity = if self.dragging == Some(index) {
            0.12
        } else {
            end
        };
        let shell = div()
            .h_full()
            .flex_none()
            .relative()
            .opacity(opacity)
            .child(card);
        let shell = if self.moves[index] == 0 {
            shell.into_any_element()
        } else {
            let shift = self.shifts[index];
            let token =
                self.job.wrapping_mul(100_000) + index as u64 * 1_000 + self.moves[index] as u64;
            shell
                .with_animation(
                    ("move", token),
                    Animation::new(CARD).with_easing(ease_in_out),
                    move |shell, delta| shell.left(px(shift * (1.0 - delta))),
                )
                .into_any_element()
        };
        if self.cards[index] == 0 {
            return shell;
        }

        let start = if active { 0.58 } else { 1.0 };
        let token = self.job.wrapping_mul(10_000) + index as u64 * 100 + self.cards[index] as u64;
        div()
            .h_full()
            .flex_none()
            .child(shell)
            .with_animation(
                ("effect", token),
                Animation::new(CARD).with_easing(ease_in_out),
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
                .unwrap_or(size(px(1_440.0), px(820.0)));
            let width = current
                .width
                .max(px(1_440.0))
                .min((available.width - px(40.0)).max(px(1_050.0)));
            let height = current
                .height
                .max(px(820.0))
                .min((available.height - px(80.0)).max(px(680.0)));
            let fitted = size(width, height);
            if fitted != current {
                window.resize(fitted);
            }
        }

        let content = self.inspect(cx);

        div()
            .id("root")
            .size_full()
            .min_w(px(1050.0))
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
            .child(self.header(cx))
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .flex()
                    .flex_col()
                    .child(content)
                    .child(bar(&self.tracks[0])),
            )
            .children(self.toast())
    }
}

fn scroll_size(handle: &ScrollHandle, height: f32) -> Option<(f32, f32)> {
    let maximum = f32::from(handle.max_offset().y).max(0.0);
    if maximum <= 0.5 || height <= 0.0 {
        return None;
    }
    let viewport = f32::from(handle.bounds().size.height).max(1.0);
    let thumb = (height * viewport / (viewport + maximum)).clamp(28.0, height);
    Some((maximum, thumb))
}

fn set_scroll(handle: &ScrollHandle, height: f32, local: f32, anchor: f32) {
    let Some((maximum, thumb)) = scroll_size(handle, height) else {
        return;
    };
    let travel = (height - thumb).max(1.0);
    let progress = ((local - anchor) / travel).clamp(0.0, 1.0);
    handle.set_offset(point(handle.offset().x, px(-maximum * progress)));
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
                    let maximum = f32::from(handle.max_offset().y).max(0.0);
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

fn hbar(handle: &ScrollHandle) -> AnyElement {
    let handle = handle.clone();
    div()
        .absolute()
        .left_3()
        .right_3()
        .bottom(px(4.0))
        .h(px(4.0))
        .child(
            canvas(
                move |_, _, _| {},
                move |bounds, _, window, _| {
                    let maximum = f32::from(handle.max_offset().x).max(0.0);
                    if maximum <= 0.5 {
                        return;
                    }
                    let width = f32::from(bounds.size.width);
                    let viewport = f32::from(handle.bounds().size.width).max(1.0);
                    let thumb = (width * viewport / (viewport + maximum)).clamp(40.0, width);
                    let progress = (-f32::from(handle.offset().x) / maximum).clamp(0.0, 1.0);
                    let left = f32::from(bounds.origin.x) + (width - thumb) * progress;
                    let top = bounds.origin.y;
                    let bottom = bounds.origin.y + bounds.size.height;
                    let left = px(left);
                    let right = left + px(thumb);
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

fn drag_knob(index: usize, param: &Param) -> Div {
    let normal = param.normal();
    div()
        .flex_1()
        .min_w(px(90.0))
        .h_full()
        .p_1()
        .when(index > 0, |node| {
            node.border_l_1().border_color(theme::LINE)
        })
        .flex()
        .flex_col()
        .items_center()
        .gap_1()
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
                            window.paint_path(path, theme::LINE);
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
                            window.paint_path(path, theme::ACCENT);
                        }
                    },
                )
                .size_full(),
            ),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme::MUTED)
                .whitespace_nowrap()
                .child(param.name),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme::INK)
                .whitespace_nowrap()
                .child(param.text()),
        )
}

fn knob(
    effect: usize,
    index: usize,
    param: &Param,
    control: Control,
    cx: &mut Context<Muspector>,
) -> AnyElement {
    let Control {
        default,
        edit,
        active,
        drag,
    } = control;
    let normal = param.normal();
    let label = param.name;
    let editing = edit.is_some();
    let value = edit
        .map(|text| format!("{text}| "))
        .unwrap_or_else(|| param.text());
    div()
        .id(("knob", effect * 100 + index))
        .flex_1()
        .min_w(px(90.0))
        .h_full()
        .p_1()
        .when(index > 0, |node| {
            node.border_l_1().border_color(theme::LINE)
        })
        .flex()
        .flex_col()
        .items_center()
        .gap_1()
        .cursor_ns_resize()
        .on_drag(drag, |drag: &EffectDrag, position, _window, cx| {
            cx.new(|_| drag.clone().position(position))
        })
        .on_click(cx.listener(move |this, event: &ClickEvent, _window, cx| {
            if event.click_count() == 2 {
                this.reset_param(effect, index, cx);
                cx.stop_propagation();
            }
        }))
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

                        if let Some(default) = default {
                            let angle = (-135.0_f32 + default * 270.0).to_radians();
                            let start = point(
                                center.x + px(angle.sin() * 15.0),
                                center.y - px(angle.cos() * 15.0),
                            );
                            let end = point(
                                center.x + px(angle.sin() * 20.0),
                                center.y - px(angle.cos() * 20.0),
                            );
                            let mut marker = PathBuilder::stroke(px(3.0));
                            marker.move_to(start);
                            marker.line_to(end);
                            if let Ok(path) = marker.build() {
                                let mut color = theme::FAINT;
                                color.a = if active { 0.82 } else { 0.46 };
                                window.paint_path(path, color);
                            }
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
        .child(
            div()
                .text_xs()
                .text_color(theme::MUTED)
                .whitespace_nowrap()
                .child(label),
        )
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
                        .overflow_hidden()
                        .whitespace_nowrap()
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
        .px_2()
        .py_2()
        .border_b_1()
        .border_color(theme::LINE)
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

fn readout(label: &'static str, value: String) -> Div {
    div()
        .w_full()
        .px_2()
        .py_2()
        .border_b_1()
        .border_color(theme::LINE)
        .flex()
        .items_center()
        .justify_between()
        .child(div().text_xs().text_color(theme::MUTED).child(label))
        .child(
            div()
                .text_xs()
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

fn spectrum_band(report: &Report) -> Div {
    let curve = report.spectrum.clone();
    let high = (report.rate as f64 / 2.0).clamp(40.0, 20_000.0);
    let centroid = position(report.centroid, high);
    let rolloff = position(report.rolloff, high);
    let chart = div()
        .h(px(92.0))
        .flex_none()
        .w_full()
        .relative()
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
        .h(px(142.0))
        .flex_none()
        .w_full()
        .pt_2()
        .border_t_1()
        .border_color(theme::LINE)
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .px_2()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .text_sm()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(theme::INK)
                                .child("Spectrum"),
                        )
                        .child(
                            div()
                                .text_size(px(9.0))
                                .text_color(theme::FAINT)
                                .child("Full file"),
                        ),
                )
                .child(
                    div()
                        .w_full()
                        .flex()
                        .gap_2()
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .items_baseline()
                                .gap_1()
                                .child(
                                    div()
                                        .text_size(px(9.0))
                                        .text_color(theme::FAINT)
                                        .child("Centroid"),
                                )
                                .child(
                                    div()
                                        .ml_auto()
                                        .text_xs()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .text_color(theme::MUTED)
                                        .child(format!("{:.0} Hz", report.centroid)),
                                ),
                        )
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .flex()
                                .items_baseline()
                                .gap_1()
                                .child(
                                    div()
                                        .text_size(px(9.0))
                                        .text_color(theme::FAINT)
                                        .child("Rolloff"),
                                )
                                .child(
                                    div()
                                        .ml_auto()
                                        .text_xs()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .text_color(theme::MUTED)
                                        .child(format!("{:.0} Hz", report.rolloff)),
                                ),
                        ),
                ),
        )
        .child(chart)
}

fn pressure_level(value: f32) -> usize {
    if value < 55.0 {
        1
    } else if value < 80.0 {
        2
    } else {
        3
    }
}

fn pressure(label: &'static str, active: usize) -> AnyElement {
    let colors = [theme::GOOD, theme::WARN, theme::ERROR];

    div()
        .w(px(48.0))
        .min_w(px(48.0))
        .flex_none()
        .flex()
        .items_center()
        .justify_end()
        .gap_1()
        .child(
            div()
                .text_size(px(9.0))
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(theme::FAINT)
                .child(label),
        )
        .child(div().flex().items_center().gap(px(2.0)).children(
            colors.into_iter().enumerate().map(|(index, color)| {
                div().size(px(6.0)).rounded(px(1.0)).bg(if index < active {
                    color
                } else {
                    theme::LINE
                })
            }),
        ))
        .into_any_element()
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

fn horizontal(position: Pixels, area: Bounds<Pixels>) -> f32 {
    let left = area.origin.x + px(10.0);
    let width = f32::from((area.size.width - px(20.0)).max(px(1.0)));
    (f32::from(position - left) / width).clamp(0.0, 1.0)
}

fn vertical(text: &str, color: gpui::Rgba, left: f32, width: f32, title: bool) -> AnyElement {
    svg()
        .absolute()
        .left(px(left))
        .top(px(if title { 20.0 } else { 44.0 }))
        .w(px(width))
        .h(px(if title { 28.0 } else { 16.0 }))
        .text_color(color)
        .path(label(if title { "title" } else { "meta" }, text))
        .with_transformation(Transformation::rotate(radians(std::f32::consts::FRAC_PI_2)))
        .into_any_element()
}

fn label(style: &str, text: &str) -> String {
    let mut path = format!("labels/{style}/");
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            path.push(char::from(byte));
        } else {
            path.push('%');
            path.push_str(&format!("{byte:02X}"));
        }
    }
    path.push_str(".svg");
    path
}

fn ordered(left: f32, right: f32) -> (f32, f32) {
    (left.min(right), left.max(right))
}

fn compact(points: &[analysis::Point], limit: usize) -> Vec<analysis::Point> {
    if points.len() <= limit {
        return points.to_vec();
    }
    let size = points.len().div_ceil(limit);
    points
        .chunks(size)
        .map(|chunk| {
            let min = chunk
                .iter()
                .map(|point| point.min)
                .fold(f32::INFINITY, f32::min);
            let max = chunk
                .iter()
                .map(|point| point.max)
                .fold(f32::NEG_INFINITY, f32::max);
            let power = chunk
                .iter()
                .map(|point| 10.0_f64.powf(point.level / 10.0))
                .sum::<f64>()
                / chunk.len() as f64;
            let loudness = chunk
                .iter()
                .map(|point| 10.0_f64.powf(point.loudness / 10.0))
                .sum::<f64>()
                / chunk.len() as f64;
            analysis::Point {
                min,
                max,
                level: 10.0 * power.max(1.0e-12).log10(),
                loudness: 10.0 * loudness.max(1.0e-12).log10(),
            }
        })
        .collect()
}

fn selection_stats(
    report: &Report,
    selection: Option<(f32, f32)>,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let (start, end) = selection?;
    let (start, end) = ordered(start, end);
    if report.profile.points.is_empty() {
        return None;
    }
    let last = report.profile.points.len().saturating_sub(1);
    if end - start <= 0.5 / last.max(1) as f32 {
        return None;
    }
    let from = (start * last as f32).floor() as usize;
    let to = ((end * last as f32).ceil() as usize).min(last);
    let points = &report.profile.points[from..=to];
    let peak = points
        .iter()
        .map(|point| f64::from(point.min.abs().max(point.max.abs())))
        .fold(0.0_f64, f64::max);
    let power = points
        .iter()
        .map(|point| 10.0_f64.powf(point.level / 10.0))
        .sum::<f64>()
        / points.len() as f64;
    let peak = level(peak);
    let rms = 10.0 * power.max(1.0e-12).log10();
    let loudness = 10.0
        * (points
            .iter()
            .map(|point| 10.0_f64.powf(point.loudness / 10.0))
            .sum::<f64>()
            / points.len() as f64)
            .max(1.0e-12)
            .log10();
    Some((
        report.duration * f64::from(start),
        report.duration * f64::from(end),
        peak,
        rms,
        peak - rms,
        loudness,
    ))
}

fn span(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1} s", seconds.max(0.0))
    } else {
        stamp(seconds)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::{Fingerprint, infer};

    fn chain() -> Chain {
        infer(Fingerprint {
            peak: -1.0,
            crest: 12.0,
            range: 16.0,
            floor: -48.0,
            silence: 0.04,
            transient: 0.5,
            flatness: 0.12,
            low: -6.0,
            mid: -4.0,
            high: -14.0,
            echo: 0.2,
            echo_ms: 180.0,
            tail: 0.12,
        })
    }

    #[test]
    fn history_merges_gestures_and_discards_redo() {
        let baseline = chain();
        let mut history = History::detected(baseline.clone(), baseline.clone(), [false; 6]);
        let mut adjusted = baseline.clone();
        adjusted.effects[0].params[0].shift(1.0);
        history.record(
            "Adjust Gate Threshold".to_owned(),
            Snapshot {
                chain: adjusted.clone(),
                baseline: baseline.clone(),
                dirty: true,
                expanded: [false; 6],
            },
            true,
        );
        adjusted.effects[0].params[0].shift(1.0);
        history.record(
            "Adjust Gate Threshold".to_owned(),
            Snapshot {
                chain: adjusted,
                baseline: baseline.clone(),
                dirty: true,
                expanded: [false; 6],
            },
            true,
        );
        assert_eq!(history.entries.len(), 2);
        assert_eq!(history.cursor, 1);

        history.cursor = 0;
        history.merge = None;
        let mut toggled = baseline.clone();
        toggled.effects[0].active = !toggled.effects[0].active;
        history.record(
            "Enable Gate".to_owned(),
            Snapshot {
                chain: toggled,
                baseline,
                dirty: true,
                expanded: [false; 6],
            },
            false,
        );
        assert_eq!(history.entries.len(), 2);
        assert_eq!(history.entries[1].label, "Enable Gate");
    }
}
