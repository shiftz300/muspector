use crate::{
    analysis::{self, Report},
    audio::{self, Audio},
    blind,
    chain::{Chain, Effect, Param},
    icon::Icon,
    theme,
};
use anyhow::Context as AnyhowContext;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use gpui::PinchEvent;
use gpui::{
    Animation, AnimationExt, AnyElement, App, AppContext, BorderStyle, Bounds, ClickEvent, Context,
    Corners, DispatchPhase, Div, DragMoveEvent, Edges, Element, ElementId, ExternalPaths,
    FocusHandle, GlobalElementId, Hitbox, HitboxBehavior, InspectorElementId, IntoElement,
    KeyDownEvent, LayoutId, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathBuilder,
    PathPromptOptions, Pixels, Point, Position, PromptLevel, Render, ScrollHandle,
    ScrollWheelEvent, Style, Styled, Transformation, Window, canvas, div, ease_in_out,
    linear_color_stop, linear_gradient, point, prelude::*, px, quad, radians, relative, size, svg,
    transparent_black,
};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

const TINT: Duration = Duration::from_millis(140);
const CARD: Duration = Duration::from_millis(180);
const TAB: Duration = Duration::from_millis(170);
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
enum ToastKind {
    Success,
    Info,
    Warning,
    Error,
}

impl ToastKind {
    fn color(self) -> gpui::Rgba {
        match self {
            Self::Success => theme::GOOD,
            Self::Info => theme::ACCENT,
            Self::Warning => theme::WARN,
            Self::Error => theme::ERROR,
        }
    }

    fn icon(self) -> Icon {
        match self {
            Self::Success => Icon::Check,
            Self::Info => Icon::Info,
            Self::Warning => Icon::TriangleAlert,
            Self::Error => Icon::CircleX,
        }
    }
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
    kind: ToastKind,
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

#[derive(Clone)]
struct Revision {
    report: Box<Report>,
    source: PathBuf,
    audio_dirty: bool,
}

#[derive(Clone)]
struct Snapshot {
    chain: Chain,
    revision: Rc<Revision>,
    baseline: Chain,
    dirty: bool,
    expanded: [bool; 6],
    selection: Option<(f32, f32)>,
    playhead: Option<f32>,
    looped: bool,
}

impl PartialEq for Snapshot {
    fn eq(&self, other: &Self) -> bool {
        self.chain == other.chain
            && Rc::ptr_eq(&self.revision, &other.revision)
            && self.baseline == other.baseline
            && self.dirty == other.dirty
            && self.expanded == other.expanded
            && self.selection == other.selection
            && self.playhead == other.playhead
            && self.looped == other.looped
    }
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
    fn detected(snapshot: Snapshot) -> Self {
        Self {
            entries: vec![Step {
                label: "Detected".to_owned(),
                snapshot,
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

fn history_sources(history: &History) -> Vec<PathBuf> {
    history
        .entries
        .iter()
        .map(|step| step.snapshot.revision.source.clone())
        .collect()
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
    source: PathBuf,
    project: Option<PathBuf>,
    report: Box<Report>,
    baseline: Chain,
    dirty: bool,
    audio_dirty: bool,
    revision: Rc<Revision>,
    expanded: [bool; 6],
    history: History,
    motion: usize,
    shift: f32,
}

#[derive(Clone)]
struct Pending {
    id: u64,
    path: PathBuf,
    progress: f32,
    previous: f32,
    stage: &'static str,
    motion: usize,
}

#[derive(Clone, Copy)]
struct SelectionMenu {
    position: Point<Pixels>,
}

#[derive(Clone, Copy)]
enum AudioEdit {
    Delete,
    Paste,
}

#[derive(Clone)]
struct SaveData {
    path: PathBuf,
    source: PathBuf,
    project: Option<PathBuf>,
    audio_dirty: bool,
    chain: Chain,
}

#[derive(Clone)]
struct Closing {
    path: PathBuf,
    token: usize,
}

#[derive(Clone)]
struct TabDrag {
    path: PathBuf,
    name: String,
    meta: String,
    active: bool,
    dirty: bool,
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
            .pl(self.position.x - px(80.0))
            .pt(self.position.y - px(22.0))
            .child(
                div()
                    .w(px(160.0))
                    .h(px(44.0))
                    .px_2()
                    .border_b_2()
                    .border_color(if self.active {
                        theme::ACCENT
                    } else {
                        theme::LINE
                    })
                    .bg(if self.active {
                        theme::SURFACE
                    } else {
                        theme::CANVAS
                    })
                    .shadow_md()
                    .overflow_hidden()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .overflow_hidden()
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .min_w_0()
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .text_xs()
                                    .line_height(px(12.0))
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .flex()
                                    .items_center()
                                    .gap(px(3.0))
                                    .children(self.dirty.then(|| {
                                        div().flex_none().text_color(theme::ACCENT).child("*")
                                    }))
                                    .child(
                                        div()
                                            .min_w_0()
                                            .overflow_hidden()
                                            .text_color(if self.active {
                                                theme::INK
                                            } else {
                                                theme::MUTED
                                            })
                                            .child(self.name.clone()),
                                    ),
                            )
                            .child(
                                div()
                                    .mt(px(2.0))
                                    .overflow_hidden()
                                    .whitespace_nowrap()
                                    .text_size(px(9.0))
                                    .line_height(px(10.0))
                                    .text_color(theme::FAINT)
                                    .child(self.meta.clone()),
                            ),
                    )
                    .child(
                        div()
                            .size(px(20.0))
                            .flex_none()
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(Icon::Close.draw(px(12.0), theme::MUTED)),
                    ),
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
                                            .child(
                                                if self.expanded {
                                                    Icon::Left
                                                } else {
                                                    Icon::Right
                                                }
                                                .draw(px(11.0), theme::MUTED),
                                            ),
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
    pending: Vec<Pending>,
    active: Option<usize>,
    pending_active: Option<u64>,
    dirty: bool,
    audio_dirty: bool,
    tab_dragging: Option<usize>,
    tab_motion: usize,
    tab_close: usize,
    closing: Vec<Closing>,
    job: u64,
    inspect_job: u64,
    hovers: [Hover; 1],
    glows: [usize; 1],
    alert: Option<Alert>,
    notice: usize,
    cursor: Option<f32>,
    playhead: Option<f32>,
    audio: Option<Audio>,
    source: Option<PathBuf>,
    revision: Option<Rc<Revision>>,
    clipboard: Option<crate::clip::Clip>,
    selection_menu: Option<SelectionMenu>,
    looped: bool,
    playback: usize,
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
    tab_track: ScrollHandle,
    tracks: [ScrollHandle; 3],
    training: blind::Training,
    training_open: bool,
    training_motion: usize,
    outputs: Vec<audio::Output>,
    output: Option<String>,
    settings_open: bool,
    settings_motion: usize,
    closing_app: bool,
}

impl Muspector {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let outputs = audio::outputs();
        let output = audio::load_output();
        let mut this = Self {
            state: State::Empty,
            tabs: Vec::new(),
            pending: Vec::new(),
            active: None,
            pending_active: None,
            dirty: false,
            audio_dirty: false,
            tab_dragging: None,
            tab_motion: 0,
            tab_close: 0,
            closing: Vec::new(),
            job: 0,
            inspect_job: 0,
            hovers: [Hover::Idle; 1],
            glows: [0; 1],
            alert: None,
            notice: 0,
            cursor: None,
            playhead: None,
            audio: None,
            source: None,
            revision: None,
            clipboard: None,
            selection_menu: None,
            looped: false,
            playback: 0,
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
            tab_track: ScrollHandle::new(),
            tracks: [
                ScrollHandle::new(),
                ScrollHandle::new(),
                ScrollHandle::new(),
            ],
            training: blind::Training::load_active(),
            training_open: false,
            training_motion: 0,
            outputs,
            output,
            settings_open: false,
            settings_motion: 0,
            closing_app: false,
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
                system.refresh_memory();
                refresh_process(&mut system);
                system
            });
            let mut system = task.await;

            loop {
                cx.background_executor().timer(Duration::from_secs(1)).await;
                let task = cx.background_spawn(async move {
                    system.refresh_memory();
                    let pressure = refresh_process(&mut system);
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

    fn notify(&mut self, kind: ToastKind, text: String, cx: &mut Context<Self>) {
        self.notice = self.notice.wrapping_add(1);
        let notice = self.notice;
        self.alert = Some(Alert {
            text,
            kind,
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

    fn success(&mut self, text: String, cx: &mut Context<Self>) {
        self.notify(ToastKind::Success, text, cx);
    }

    fn info(&mut self, text: String, cx: &mut Context<Self>) {
        self.notify(ToastKind::Info, text, cx);
    }

    fn warn(&mut self, text: String, cx: &mut Context<Self>) {
        self.notify(ToastKind::Warning, text, cx);
    }

    fn error(&mut self, text: String, cx: &mut Context<Self>) {
        self.notify(ToastKind::Error, text, cx);
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

    fn import_clean(&mut self, cx: &mut Context<Self>) {
        self.set_training_open(false, cx);
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Import Clean".into()),
        });
        cx.spawn(async move |view, cx| {
            if let Ok(Ok(Some(paths))) = receiver.await
                && let Some(path) = paths.into_iter().next()
            {
                let task = cx.background_spawn(async move { analysis::training_from_clean(&path) });
                let result = task.await;
                let _ = view.update(cx, |this, cx| match result {
                    Ok(training) => match training.save_active() {
                        Ok(()) => {
                            this.training = training;
                            this.success("Clean reference active".to_owned(), cx);
                        }
                        Err(error) => this.error(format!("Could not save training: {error:#}"), cx),
                    },
                    Err(error) => this.error(format!("Could not import clean: {error:#}"), cx),
                });
            }
        })
        .detach();
    }

    fn import_training(&mut self, cx: &mut Context<Self>) {
        self.set_training_open(false, cx);
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Import Training".into()),
        });
        cx.spawn(async move |view, cx| {
            if let Ok(Ok(Some(paths))) = receiver.await
                && let Some(path) = paths.into_iter().next()
            {
                let task = cx.background_spawn(async move {
                    let bytes = std::fs::read(&path)
                        .map_err(anyhow::Error::from)
                        .with_context(|| format!("could not read {}", path.display()))?;
                    blind::Training::import(&bytes)
                });
                let result = task.await;
                let _ = view.update(cx, |this, cx| match result {
                    Ok(training) => match training.save_active() {
                        Ok(()) => {
                            this.training = training;
                            this.success("Training profile active".to_owned(), cx);
                        }
                        Err(error) => this.error(format!("Could not save training: {error:#}"), cx),
                    },
                    Err(error) => this.error(format!("Could not import training: {error:#}"), cx),
                });
            }
        })
        .detach();
    }

    fn export_training(&mut self, cx: &mut Context<Self>) {
        self.set_training_open(false, cx);
        let receiver =
            cx.prompt_for_new_path(Path::new("."), Some("muspector-profile.musp-training"));
        let bytes = self.training.export();
        cx.spawn(async move |view, cx| {
            if let Ok(Ok(Some(path))) = receiver.await {
                let result = cx
                    .background_spawn(async move { std::fs::write(path, bytes) })
                    .await;
                let _ = view.update(cx, |this, cx| match result {
                    Ok(()) => this.success("Training profile exported".to_owned(), cx),
                    Err(error) => this.error(format!("Could not export training: {error}"), cx),
                });
            }
        })
        .detach();
    }

    fn restore_default_training(&mut self, cx: &mut Context<Self>) {
        self.set_training_open(false, cx);
        let training = blind::Training::embedded();
        match training.save_active() {
            Ok(()) => {
                self.training = training;
                self.success("Default clean restored".to_owned(), cx);
            }
            Err(error) => self.error(format!("Could not restore default clean: {error:#}"), cx),
        }
    }

    fn set_training_open(&mut self, open: bool, cx: &mut Context<Self>) {
        if self.training_open == open {
            return;
        }
        self.training_open = open;
        if open {
            self.settings_open = false;
        }
        self.training_motion = self.training_motion.wrapping_add(1).max(1);
        let motion = self.training_motion;
        cx.notify();
        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(TINT).await;
            let _ = view.update(cx, |this, cx| {
                if this.training_motion == motion {
                    this.training_motion = 0;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn set_settings_open(&mut self, open: bool, cx: &mut Context<Self>) {
        if self.settings_open == open {
            return;
        }
        if open {
            self.outputs = audio::outputs();
            self.training_open = false;
        }
        self.settings_open = open;
        self.settings_motion = self.settings_motion.wrapping_add(1).max(1);
        let motion = self.settings_motion;
        cx.notify();
        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(TINT).await;
            let _ = view.update(cx, |this, cx| {
                if this.settings_motion == motion {
                    this.settings_motion = 0;
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn select_output(&mut self, output: Option<String>, cx: &mut Context<Self>) {
        if self.output == output {
            self.set_settings_open(false, cx);
            return;
        }
        if let Err(error) = audio::save_output(output.as_deref()) {
            self.error(format!("Could not save audio output: {error:#}"), cx);
            return;
        }
        self.output = output;
        self.audio = None;
        self.playback = self.playback.wrapping_add(1);
        self.set_settings_open(false, cx);
        self.success("Audio output updated".to_owned(), cx);
    }

    fn reset_editor(&mut self) {
        self.playback = self.playback.wrapping_add(1);
        self.audio = None;
        self.cursor = None;
        self.playhead = None;
        self.looped = false;
        self.selection = None;
        self.drag = None;
        self.pan = None;
        self.selection_menu = None;
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

    fn seek(&mut self, position: f32, cx: &mut Context<Self>) {
        let position = position.clamp(0.0, 1.0);
        self.playhead = Some(position);
        let Some((path, duration)) = (match (&self.state, &self.source) {
            (State::Ready(report), Some(source)) => Some((source.clone(), report.duration)),
            _ => None,
        }) else {
            return;
        };
        let result = self.audio.as_ref().and_then(|audio| {
            audio
                .matches(&path)
                .then(|| audio.seek(Duration::from_secs_f64(duration * f64::from(position))))
        });
        if let Some(Err(error)) = result {
            self.audio = None;
            self.error(format!("Could not seek: {error}"), cx);
        } else {
            cx.notify();
        }
    }

    fn toggle_play(&mut self, cx: &mut Context<Self>) {
        let Some((path, duration)) = (match (&self.state, &self.source) {
            (State::Ready(report), Some(source)) => Some((source.clone(), report.duration)),
            _ => None,
        }) else {
            return;
        };
        if duration <= f64::EPSILON {
            self.warn("This file has no playable duration".to_owned(), cx);
            return;
        }

        if let Some(audio) = &self.audio
            && audio.matches(&path)
            && !audio.paused()
        {
            audio.pause();
            self.playback = self.playback.wrapping_add(1);
            cx.notify();
            return;
        }

        let replace = self
            .audio
            .as_ref()
            .is_none_or(|audio| !audio.matches(&path) || audio.empty());
        if replace {
            let mut position = self.playhead.unwrap_or(0.0).clamp(0.0, 1.0);
            if self.looped
                && let Some((start, end)) = self.selection.map(|range| ordered(range.0, range.1))
                && (position < start || position >= end)
            {
                position = start;
            }
            match Audio::open(
                &path,
                Duration::from_secs_f64(duration * f64::from(position)),
                self.output.as_deref(),
            ) {
                Ok(audio) => self.audio = Some(audio),
                Err(error) => {
                    self.error(format!("Could not play: {error}"), cx);
                    return;
                }
            }
        }

        if let Some(audio) = &self.audio {
            audio.play();
            self.playhead =
                Some((audio.position().as_secs_f64() / duration).clamp(0.0, 1.0) as f32);
        }
        self.watch(cx);
        cx.notify();
    }

    fn toggle_loop(&mut self, cx: &mut Context<Self>) {
        let Some((start, end)) = self.selection.map(|range| ordered(range.0, range.1)) else {
            self.warn("Drag across the waveform to select a loop".to_owned(), cx);
            return;
        };
        if end - start <= f32::EPSILON {
            self.warn("Select a longer range to loop".to_owned(), cx);
            return;
        }
        self.looped = !self.looped;
        if self.looped
            && self
                .playhead
                .is_none_or(|position| position < start || position >= end)
        {
            self.seek(start, cx);
        } else {
            cx.notify();
        }
    }

    fn watch(&mut self, cx: &mut Context<Self>) {
        self.playback = self.playback.wrapping_add(1);
        let playback = self.playback;
        cx.spawn(async move |view, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(33))
                    .await;
                let active = view
                    .update(cx, |this, cx| {
                        if this.playback != playback {
                            return false;
                        }
                        let State::Ready(report) = &this.state else {
                            return false;
                        };
                        let duration = report.duration;
                        let Some(path) = this.source.clone() else {
                            return false;
                        };
                        let Some(audio) = &this.audio else {
                            return false;
                        };
                        if !audio.matches(&path) || audio.paused() {
                            return false;
                        }

                        let mut seconds = audio.position().as_secs_f64();
                        if this.looped
                            && let Some((start, end)) =
                                this.selection.map(|range| ordered(range.0, range.1))
                        {
                            let start = duration * f64::from(start);
                            let end = duration * f64::from(end);
                            if seconds >= end || seconds < start {
                                if audio.seek(Duration::from_secs_f64(start)).is_err() {
                                    this.looped = false;
                                } else {
                                    seconds = start;
                                }
                            }
                        }

                        let ended = audio.empty() || seconds >= duration;
                        this.playhead = Some(if ended {
                            1.0
                        } else {
                            (seconds / duration).clamp(0.0, 1.0) as f32
                        });
                        cx.notify();
                        !ended
                    })
                    .unwrap_or(false);
                if !active {
                    break;
                }
            }
        })
        .detach();
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
        if let Some(source) = &self.source {
            tab.source = source.clone();
        }
        if let Some(revision) = &self.revision {
            tab.revision = revision.clone();
        }
        if let Some(baseline) = &self.baseline {
            tab.baseline = baseline.clone();
        }
        tab.dirty = self.dirty;
        tab.audio_dirty = self.audio_dirty;
        tab.expanded = self.expanded;
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
        self.pending_active = None;
        self.state = State::Ready(tab.report);
        self.source = Some(tab.source);
        self.revision = Some(tab.revision);
        self.baseline = Some(tab.baseline);
        self.dirty = tab.dirty;
        self.audio_dirty = tab.audio_dirty;
        self.history = tab.history;
        self.reset_editor();
        self.expanded = tab.expanded;
        cx.notify();
    }

    fn activate_pending(&mut self, id: u64, cx: &mut Context<Self>) {
        let Some(path) = self
            .pending
            .iter()
            .find(|pending| pending.id == id)
            .map(|pending| pending.path.clone())
        else {
            return;
        };
        self.sync_active();
        self.active = None;
        self.pending_active = Some(id);
        self.audio = None;
        self.playback = self.playback.wrapping_add(1);
        self.reset_editor();
        self.state = State::Loading(path);
        cx.notify();
    }

    fn close_pending(&mut self, id: u64, cx: &mut Context<Self>) {
        let Some(index) = self.pending.iter().position(|pending| pending.id == id) else {
            return;
        };
        self.pending.remove(index);
        if self.pending_active != Some(id) {
            cx.notify();
            return;
        }
        self.pending_active = None;
        if self.tabs.is_empty() {
            if let Some(next) = self.pending.first().cloned() {
                self.pending_active = Some(next.id);
                self.state = State::Loading(next.path);
            } else {
                self.state = State::Empty;
            }
            cx.notify();
        } else {
            self.activate(0, cx);
        }
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
            revision: self.revision.clone()?,
            baseline: self.baseline.clone()?,
            dirty: self.dirty,
            expanded: self.expanded,
            selection: self.selection,
            playhead: self.playhead,
            looped: self.looped,
        })
    }

    fn refresh_history_cursor(&mut self) {
        let Some(snapshot) = self.snapshot() else {
            return;
        };
        if let Some(step) = self.history.entries.get_mut(self.history.cursor) {
            step.snapshot = snapshot;
        }
        if let Some(index) = self.active
            && let Some(tab) = self.tabs.get_mut(index)
        {
            tab.history = self.history.clone();
        }
    }

    fn record(&mut self, label: impl Into<String>, merge: bool) {
        let Some(snapshot) = self.snapshot() else {
            return;
        };
        let previous = history_sources(&self.history);
        self.history.record(label.into(), snapshot, merge);
        if let Some(index) = self.active
            && let Some(tab) = self.tabs.get_mut(index)
        {
            tab.expanded = self.expanded;
            tab.history = self.history.clone();
        }
        self.cleanup_unreferenced(previous);
    }

    fn cleanup_unreferenced(&self, sources: Vec<PathBuf>) {
        for source in sources {
            let retained = self.source.as_ref() == Some(&source)
                || self
                    .clipboard
                    .as_ref()
                    .is_some_and(|clip| clip.path == source)
                || self.tabs.iter().any(|tab| {
                    tab.source == source
                        || tab
                            .history
                            .entries
                            .iter()
                            .any(|step| step.snapshot.revision.source == source)
                });
            if !retained {
                crate::clip::cleanup(&source);
            }
        }
    }

    fn restore(&mut self, cursor: usize, cx: &mut Context<Self>) {
        let Some(step) = self.history.entries.get(cursor).cloned() else {
            return;
        };
        self.history.cursor = cursor;
        self.history.merge = None;
        self.audio = None;
        self.playback = self.playback.wrapping_add(1);
        let mut report = step.snapshot.revision.report.clone();
        report.chain = step.snapshot.chain.clone();
        self.state = State::Ready(report.clone());
        self.source = Some(step.snapshot.revision.source.clone());
        self.revision = Some(step.snapshot.revision.clone());
        self.baseline = Some(step.snapshot.baseline.clone());
        self.audio_dirty = step.snapshot.revision.audio_dirty;
        self.dirty = step.snapshot.dirty;
        self.expanded = step.snapshot.expanded;
        self.selection = step.snapshot.selection;
        self.playhead = step.snapshot.playhead;
        self.looped = step.snapshot.looped && self.selection.is_some();
        self.edit = None;
        self.cards = [0; 6];
        self.folds = [0; 6];
        self.moves = [0; 6];
        self.shifts = [0.0; 6];
        self.dragging = None;
        if let Some(index) = self.active
            && let Some(tab) = self.tabs.get_mut(index)
        {
            tab.report = report;
            tab.source = step.snapshot.revision.source.clone();
            tab.revision = step.snapshot.revision;
            tab.baseline = step.snapshot.baseline;
            tab.dirty = self.dirty;
            tab.audio_dirty = self.audio_dirty;
            tab.expanded = self.expanded;
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

    fn remove_tab(&mut self, path: &Path, cx: &mut Context<Self>) {
        let Some(index) = self.tabs.iter().position(|tab| tab.path == path) else {
            return;
        };
        if index >= self.tabs.len() {
            return;
        }
        let was_active = self.active == Some(index);
        let removed = self.tabs.remove(index);
        if was_active {
            self.audio = None;
        }
        crate::clip::cleanup(&removed.source);
        for step in &removed.history.entries {
            crate::clip::cleanup(&step.snapshot.revision.source);
        }
        self.closing.retain(|closing| closing.path != path);
        self.tab_dragging = None;
        if self.tabs.is_empty() {
            self.active = None;
            self.source = None;
            self.revision = None;
            self.baseline = None;
            self.dirty = false;
            self.audio_dirty = false;
            self.history = History::default();
            self.reset_editor();
            if let Some(pending) = self.pending.first() {
                self.pending_active = Some(pending.id);
                self.state = State::Loading(pending.path.clone());
            } else {
                self.pending_active = None;
                self.state = State::Empty;
            }
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

    fn close_now(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(path) = self.tabs.get(index).map(|tab| tab.path.clone()) else {
            return;
        };
        if self.closing.iter().any(|closing| closing.path == path) {
            return;
        }
        self.tab_close = self.tab_close.wrapping_add(1).max(1);
        let token = self.tab_close;
        self.closing.push(Closing {
            path: path.clone(),
            token,
        });
        cx.notify();

        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(TAB).await;
            let _ = view.update(cx, move |this, cx| {
                let still_closing = this
                    .closing
                    .iter()
                    .any(|closing| closing.path == path && closing.token == token);
                if still_closing {
                    this.remove_tab(&path, cx);
                }
            });
        })
        .detach();
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
            &format!("Save changes to {name}?"),
            Some("This document has unsaved audio or signal-chain changes."),
            &["Save", "Don’t Save", "Cancel"],
            cx,
        );
        cx.spawn(async move |view, cx| match answer.await.unwrap_or(2) {
            0 => {
                let _ = view.update(cx, |this, cx| this.save_and_close(path, cx));
            }
            1 => {
                let _ = view.update(cx, |this, cx| {
                    if let Some(index) = this.tabs.iter().position(|tab| tab.path == path) {
                        this.close_now(index, cx);
                    }
                });
            }
            _ => {}
        })
        .detach();
    }

    fn save_data(&self, path: &Path) -> Option<SaveData> {
        let tab = self.tabs.iter().find(|tab| tab.path == path)?;
        Some(SaveData {
            path: tab.path.clone(),
            source: tab.source.clone(),
            project: tab.project.clone(),
            audio_dirty: tab.audio_dirty,
            chain: tab.report.chain.clone(),
        })
    }

    fn apply_saved(&mut self, path: &Path, saved: crate::project::Saved) {
        let Some(index) = self.tabs.iter().position(|tab| tab.path == path) else {
            return;
        };
        let old_source = self.tabs[index].source.clone();
        let revision = Rc::new(Revision {
            report: self.tabs[index].report.clone(),
            source: saved.audio.clone(),
            audio_dirty: false,
        });
        self.tabs[index].project = Some(saved.project);
        self.tabs[index].source = saved.audio.clone();
        self.tabs[index].dirty = false;
        self.tabs[index].audio_dirty = false;
        self.tabs[index].revision = revision.clone();
        let cursor = self.tabs[index].history.cursor;
        if let Some(step) = self.tabs[index].history.entries.get_mut(cursor) {
            step.snapshot.revision = revision.clone();
            step.snapshot.dirty = false;
        }
        if self.active == Some(index) {
            self.audio = None;
            self.source = Some(saved.audio);
            self.revision = Some(revision);
            self.dirty = false;
            self.audio_dirty = false;
            self.history = self.tabs[index].history.clone();
        }
        let retained = self.tabs.iter().any(|tab| {
            tab.source == old_source
                || tab
                    .history
                    .entries
                    .iter()
                    .any(|step| step.snapshot.revision.source == old_source)
        });
        if !retained {
            crate::clip::cleanup(&old_source);
        }
    }

    fn cleanup(&mut self) {
        self.audio = None;
        for tab in &self.tabs {
            crate::clip::cleanup(&tab.source);
            for step in &tab.history.entries {
                crate::clip::cleanup(&step.snapshot.revision.source);
            }
        }
        if let Some(clipboard) = self.clipboard.take() {
            crate::clip::cleanup(&clipboard.path);
        }
    }

    fn save_and_close(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.sync_active();
        let Some(data) = self.save_data(&path) else {
            return;
        };
        let task = cx.background_spawn(async move {
            crate::project::save(
                data.project.as_deref(),
                &data.path,
                &data.source,
                data.audio_dirty,
                &data.chain,
            )
        });
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| match result {
                Ok(saved) => {
                    this.apply_saved(&path, saved);
                    this.success("Document saved".to_owned(), cx);
                    if let Some(index) = this.tabs.iter().position(|tab| tab.path == path) {
                        this.close_now(index, cx);
                    }
                }
                Err(error) => this.error(format!("Could not save: {error:#}"), cx),
            });
        })
        .detach();
    }

    fn save_active(&mut self, cx: &mut Context<Self>) {
        self.sync_active();
        let Some(path) = self
            .active
            .and_then(|index| self.tabs.get(index))
            .map(|tab| tab.path.clone())
        else {
            return;
        };
        let Some(data) = self.save_data(&path) else {
            return;
        };
        let task = cx.background_spawn(async move {
            crate::project::save(
                data.project.as_deref(),
                &data.path,
                &data.source,
                data.audio_dirty,
                &data.chain,
            )
        });
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| match result {
                Ok(saved) => {
                    this.apply_saved(&path, saved);
                    this.success("Document saved".to_owned(), cx);
                }
                Err(error) => this.error(format!("Could not save: {error:#}"), cx),
            });
        })
        .detach();
    }

    pub fn window_should_close(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        self.sync_active();
        if !self.tabs.iter().any(|tab| tab.dirty) {
            self.cleanup();
            return true;
        }
        if !self.closing_app {
            self.ask_quit(window, cx);
        }
        false
    }

    fn ask_quit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.closing_app = true;
        let dirty = self.tabs.iter().filter(|tab| tab.dirty).count();
        let detail = if dirty == 1 {
            "One document has unsaved changes.".to_owned()
        } else {
            format!("{dirty} documents have unsaved changes.")
        };
        let handle = window.window_handle();
        let answer = window.prompt(
            PromptLevel::Warning,
            "Save changes before quitting?",
            Some(&detail),
            &["Save All", "Don’t Save", "Cancel"],
            cx,
        );
        cx.spawn(async move |view, cx| match answer.await.unwrap_or(2) {
            0 => {
                let Ok(data) = view.update(cx, |this, _cx| {
                    this.sync_active();
                    this.tabs
                        .iter()
                        .filter(|tab| tab.dirty)
                        .map(|tab| SaveData {
                            path: tab.path.clone(),
                            source: tab.source.clone(),
                            project: tab.project.clone(),
                            audio_dirty: tab.audio_dirty,
                            chain: tab.report.chain.clone(),
                        })
                        .collect::<Vec<_>>()
                }) else {
                    return;
                };
                let task = cx.background_spawn(async move {
                    let mut saved = Vec::with_capacity(data.len());
                    for item in data {
                        let result = crate::project::save(
                            item.project.as_deref(),
                            &item.path,
                            &item.source,
                            item.audio_dirty,
                            &item.chain,
                        )?;
                        saved.push((item.path, result));
                    }
                    Ok::<_, anyhow::Error>(saved)
                });
                let result = task.await;
                let close = view
                    .update(cx, |this, cx| match result {
                        Ok(saved) => {
                            for (path, saved) in saved {
                                this.apply_saved(&path, saved);
                            }
                            this.cleanup();
                            true
                        }
                        Err(error) => {
                            this.closing_app = false;
                            this.error(format!("Could not save: {error:#}"), cx);
                            false
                        }
                    })
                    .unwrap_or(false);
                if close {
                    let _ = handle.update(cx, |_root, window, _cx| window.remove_window());
                }
            }
            1 => {
                let _ = view.update(cx, |this, _cx| this.cleanup());
                let _ = handle.update(cx, |_root, window, _cx| window.remove_window());
            }
            _ => {
                let _ = view.update(cx, |this, cx| {
                    this.closing_app = false;
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn reorder_tab(&mut self, from: usize, to: usize, distance: f32, cx: &mut Context<Self>) {
        if from == to || from >= self.tabs.len() || to >= self.tabs.len() {
            return;
        }
        if self.closing.iter().any(|closing| {
            closing.path == self.tabs[from].path || closing.path == self.tabs[to].path
        }) {
            return;
        }
        self.tab_motion = self.tab_motion.wrapping_add(1).max(1);
        let motion = self.tab_motion;
        for tab in &mut self.tabs {
            tab.motion = 0;
            tab.shift = 0.0;
        }
        let tab = self.tabs.remove(from);
        self.tabs.insert(to, tab);
        if from < to {
            for tab in &mut self.tabs[from..to] {
                tab.motion = motion;
                tab.shift = distance;
            }
        } else {
            for tab in &mut self.tabs[(to + 1)..=from] {
                tab.motion = motion;
                tab.shift = -distance;
            }
        }
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

        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(TAB).await;
            let _ = view.update(cx, move |this, cx| {
                if this.tab_motion == motion {
                    for tab in &mut this.tabs {
                        tab.motion = 0;
                        tab.shift = 0.0;
                    }
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn finish_tab_drag(&mut self, cx: &mut Context<Self>) {
        if self.tab_dragging.take().is_some() {
            cx.notify();
        }
    }

    fn start(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        if !supported(&path) {
            self.warn("Unsupported audio format".to_owned(), cx);
            return;
        }

        if let Some(index) = self.tabs.iter().position(|tab| tab.path == path) {
            self.activate(index, cx);
            self.info("This file is already open".to_owned(), cx);
            return;
        }
        if let Some(id) = self
            .pending
            .iter()
            .find(|pending| pending.path == path)
            .map(|pending| pending.id)
        {
            self.activate_pending(id, cx);
            self.info("This file is already being inspected".to_owned(), cx);
            return;
        }

        self.sync_active();
        self.inspect_job = self.inspect_job.wrapping_add(1).max(1);
        let job = self.inspect_job;
        self.pending.push(Pending {
            id: job,
            path: path.clone(),
            progress: 0.0,
            previous: 0.0,
            stage: "Queued",
            motion: 0,
        });
        let foreground = self.active.is_none() && self.pending_active.is_none();
        if foreground {
            self.pending_active = Some(job);
            self.state = State::Loading(path.clone());
            self.reset_editor();
            self.fit = false;
        }
        cx.notify();

        let training = self.training.clone();
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let task = cx.background_spawn(async move {
            analysis::inspect_with_progress(&path, training, |progress| {
                let _ = progress_tx.send(progress);
            })
        });
        cx.spawn(async move |view, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(33))
                    .await;
                let mut latest = None;
                let mut disconnected = false;
                loop {
                    match progress_rx.try_recv() {
                        Ok(progress) => latest = Some(progress),
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }
                let pending = view
                    .update(cx, |this, cx| {
                        let Some(pending) =
                            this.pending.iter_mut().find(|pending| pending.id == job)
                        else {
                            return false;
                        };
                        if let Some(progress) = latest {
                            pending.previous = pending.progress;
                            pending.progress = progress.value.clamp(0.0, 1.0);
                            pending.stage = progress.stage;
                            pending.motion = pending.motion.wrapping_add(1).max(1);
                            cx.notify();
                        }
                        true
                    })
                    .unwrap_or(false);
                if !pending || disconnected {
                    break;
                }
            }
        })
        .detach();
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| {
                let Some(pending) = this.pending.iter().position(|pending| pending.id == job)
                else {
                    return;
                };
                let foreground = this.pending_active == Some(job);
                this.pending.remove(pending);
                match result {
                    Ok(report) => {
                        let report = Box::new(report);
                        let baseline = report.chain.clone();
                        let expanded_state = [false; 6];
                        let revision = Rc::new(Revision {
                            report: report.clone(),
                            source: report.path.clone(),
                            audio_dirty: false,
                        });
                        let history = History::detected(Snapshot {
                            chain: report.chain.clone(),
                            revision: revision.clone(),
                            baseline: baseline.clone(),
                            dirty: false,
                            expanded: expanded_state,
                            selection: None,
                            playhead: None,
                            looped: false,
                        });
                        this.tabs.push(Tab {
                            path: report.path.clone(),
                            source: report.path.clone(),
                            project: None,
                            report: report.clone(),
                            baseline: baseline.clone(),
                            dirty: false,
                            audio_dirty: false,
                            revision: revision.clone(),
                            expanded: expanded_state,
                            history: history.clone(),
                            motion: 0,
                            shift: 0.0,
                        });
                        if foreground {
                            this.pending_active = None;
                            this.fit = false;
                            this.activate(this.tabs.len() - 1, cx);
                        } else {
                            cx.notify();
                        }
                    }
                    Err(error) => {
                        if foreground {
                            this.pending_active = None;
                            if this.tabs.is_empty() {
                                if let Some(next) = this.pending.first().cloned() {
                                    this.pending_active = Some(next.id);
                                    this.state = State::Loading(next.path);
                                } else {
                                    this.state = State::Empty;
                                }
                            } else {
                                this.activate(0, cx);
                            }
                        }
                        this.error(format!("{error:#}"), cx);
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
            let audio_dirty = self.audio_dirty;
            report.chain = baseline.clone();
            self.edit = None;
            self.cards = [0; 6];
            self.folds = [0; 6];
            self.expanded = [false; 6];
            self.moves = [0; 6];
            self.shifts = [0.0; 6];
            self.dragging = None;
            self.dirty = audio_dirty;
            if let Some(index) = self.active
                && let Some(tab) = self.tabs.get_mut(index)
            {
                tab.dirty = audio_dirty;
                tab.expanded = self.expanded;
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
        let source = previous.source.clone();
        let from = previous.report.duration * f64::from(selection.0);
        let to = previous.report.duration * f64::from(selection.1);
        self.job = self.job.wrapping_add(1);
        let job = self.job;

        let training = self.training.clone();
        let task = cx.background_spawn(async move {
            analysis::inspect_range_with_training(&source, from, to, training)
        });
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
                        let expanded = [false; 6];

                        this.tabs[index].report.chain = chain.clone();
                        this.tabs[index].baseline = baseline.clone();
                        let audio_dirty = previous.audio_dirty;
                        this.tabs[index].dirty = audio_dirty;
                        this.tabs[index].audio_dirty = audio_dirty;
                        this.tabs[index].expanded = expanded;

                        if active {
                            if let State::Ready(report) = &mut this.state {
                                report.chain = chain.clone();
                            }
                            this.baseline = Some(baseline.clone());
                            this.dirty = audio_dirty;
                            this.audio_dirty = audio_dirty;
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
                            let previous = history_sources(&this.tabs[index].history);
                            let snapshot = Snapshot {
                                chain,
                                revision: this.tabs[index].revision.clone(),
                                baseline,
                                dirty: audio_dirty,
                                expanded,
                                selection: None,
                                playhead: None,
                                looped: false,
                            };
                            this.tabs[index].history.record(
                                "Rescan selection".to_owned(),
                                snapshot,
                                false,
                            );
                            this.cleanup_unreferenced(previous);
                        }
                        cx.notify();
                    }
                    Err(error) => {
                        this.error(format!("{error:#}"), cx);
                    }
                }
            });
        })
        .detach();
    }

    fn selected_range(&self) -> Option<(PathBuf, PathBuf, f64, f64)> {
        let index = self.active?;
        let tab = self.tabs.get(index)?;
        let (start, end) = self.selection.map(|range| ordered(range.0, range.1))?;
        let bins = tab.report.profile.points.len().saturating_sub(1).max(1);
        if end - start <= 0.5 / bins as f32 {
            return None;
        }
        Some((
            tab.path.clone(),
            tab.source.clone(),
            tab.report.duration * f64::from(start),
            tab.report.duration * f64::from(end),
        ))
    }

    fn copy_selection(&mut self, cx: &mut Context<Self>) {
        let Some((_logical, source, start, end)) = self.selected_range() else {
            self.warn("Select a non-empty audio range".to_owned(), cx);
            return;
        };
        self.selection_menu = None;
        let task = cx.background_spawn(async move { crate::clip::copy(&source, start, end) });
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| match result {
                Ok(clip) => {
                    let duration = clip.duration();
                    this.clipboard = Some(clip);
                    this.success(format!("Copied {}", span(duration)), cx);
                }
                Err(error) => this.error(format!("Could not copy audio: {error:#}"), cx),
            });
        })
        .detach();
    }

    fn edit_audio(&mut self, edit: AudioEdit, cx: &mut Context<Self>) {
        let Some(index) = self.active else {
            return;
        };
        let Some(tab) = self.tabs.get(index).cloned() else {
            return;
        };
        let logical = tab.path.clone();
        let source = tab.source.clone();
        let duration = tab.report.duration;
        let chain = tab.report.chain.clone();
        let baseline = tab.baseline.clone();
        let (start, end) = match edit {
            AudioEdit::Delete => {
                let Some((_logical, _source, start, end)) = self.selected_range() else {
                    self.warn("Select a non-empty audio range".to_owned(), cx);
                    return;
                };
                (start, end)
            }
            AudioEdit::Paste => {
                let position = self
                    .selection
                    .map(|range| ordered(range.0, range.1))
                    .unwrap_or_else(|| {
                        let position = self.playhead.or(self.cursor).unwrap_or(0.0);
                        (position, position)
                    });
                (
                    duration * f64::from(position.0),
                    duration * f64::from(position.1),
                )
            }
        };
        let clip = match edit {
            AudioEdit::Paste => {
                let Some(clip) = self.clipboard.clone() else {
                    self.warn("Copy an audio range before pasting".to_owned(), cx);
                    return;
                };
                Some(clip)
            }
            AudioEdit::Delete => None,
        };
        self.refresh_history_cursor();
        self.selection_menu = None;
        self.audio = None;
        self.playback = self.playback.wrapping_add(1);
        self.job = self.job.wrapping_add(1);
        let job = self.job;
        let training = self.training.clone();
        let task = cx.background_spawn(async move {
            let edited = match edit {
                AudioEdit::Delete => crate::clip::delete(&source, start, end),
                AudioEdit::Paste => crate::clip::paste(
                    &source,
                    clip.as_ref().expect("paste clipboard checked"),
                    start,
                    end,
                ),
            }?;
            let mut report = analysis::inspect_with_training(&edited.path, training)?;
            report.path = logical.clone();
            report.chain = chain;
            Ok::<_, anyhow::Error>((logical, edited.path, Box::new(report)))
        });
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| {
                if this.job != job {
                    if let Ok((_, source, _)) = &result {
                        crate::clip::cleanup(source);
                    }
                    return;
                }
                match result {
                    Ok((logical, source, report)) => {
                        let Some(index) = this.tabs.iter().position(|tab| tab.path == logical)
                        else {
                            crate::clip::cleanup(&source);
                            return;
                        };
                        let active = this.active == Some(index);
                        let revision = Rc::new(Revision {
                            report: report.clone(),
                            source: source.clone(),
                            audio_dirty: true,
                        });
                        this.tabs[index].source = source.clone();
                        this.tabs[index].report = report.clone();
                        this.tabs[index].baseline = baseline.clone();
                        this.tabs[index].dirty = true;
                        this.tabs[index].audio_dirty = true;
                        this.tabs[index].revision = revision.clone();
                        if active {
                            this.source = Some(source);
                            this.state = State::Ready(report);
                            this.revision = Some(revision);
                            this.baseline = Some(baseline);
                            this.dirty = true;
                            this.audio_dirty = true;
                            this.selection = None;
                            this.looped = false;
                            this.playhead = Some((start / duration.max(f64::EPSILON)) as f32);
                            this.record(
                                match edit {
                                    AudioEdit::Delete => "Delete selection",
                                    AudioEdit::Paste => "Paste audio",
                                },
                                false,
                            );
                        } else {
                            let previous = history_sources(&this.tabs[index].history);
                            let snapshot = Snapshot {
                                chain: this.tabs[index].report.chain.clone(),
                                revision,
                                baseline,
                                dirty: true,
                                expanded: this.tabs[index].expanded,
                                selection: None,
                                playhead: Some((start / duration.max(f64::EPSILON)) as f32),
                                looped: false,
                            };
                            this.tabs[index].history.record(
                                match edit {
                                    AudioEdit::Delete => "Delete selection",
                                    AudioEdit::Paste => "Paste audio",
                                }
                                .to_owned(),
                                snapshot,
                                false,
                            );
                            this.cleanup_unreferenced(previous);
                        }
                        this.success(
                            match edit {
                                AudioEdit::Delete => "Selection deleted",
                                AudioEdit::Paste => "Audio pasted",
                            }
                            .to_owned(),
                            cx,
                        );
                    }
                    Err(error) => this.error(format!("Could not edit audio: {error:#}"), cx),
                }
            });
        })
        .detach();
    }

    fn quick_export(&mut self, cx: &mut Context<Self>) {
        let Some((logical, source, start, end)) = self.selected_range() else {
            self.warn("Select a non-empty audio range".to_owned(), cx);
            return;
        };
        let target = selection_path(&logical);
        self.selection_menu = None;
        let task =
            cx.background_spawn(async move { crate::clip::export(&source, &target, start, end) });
        cx.spawn(async move |view, cx| {
            let result = task.await;
            let _ = view.update(cx, |this, cx| match result {
                Ok(()) => this.success("Selection exported".to_owned(), cx),
                Err(error) => this.error(format!("Could not export WAV: {error:#}"), cx),
            });
        })
        .detach();
    }

    fn select_all(&mut self, cx: &mut Context<Self>) {
        if matches!(self.state, State::Ready(_)) {
            self.selection = Some((0.0, 1.0));
            self.cursor = Some(1.0);
            self.selection_menu = None;
            cx.notify();
        }
    }

    fn expand(&mut self, effect: usize, cx: &mut Context<Self>) {
        if effect >= self.expanded.len() {
            return;
        }
        self.expanded[effect] = !self.expanded[effect];
        if let Some(index) = self.active
            && let Some(tab) = self.tabs.get_mut(index)
        {
            tab.expanded = self.expanded;
        }
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
            if !command
                && !event.keystroke.modifiers.alt
                && !event.keystroke.modifiers.shift
                && matches!(event.keystroke.key.as_str(), "space" | " ")
            {
                self.toggle_play(cx);
                cx.stop_propagation();
                return;
            }
            if command {
                match event.keystroke.key.as_str() {
                    "a" => self.select_all(cx),
                    "c" => self.copy_selection(cx),
                    "v" => self.edit_audio(AudioEdit::Paste, cx),
                    "e" => self.quick_export(cx),
                    "s" => self.save_active(cx),
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
            if matches!(
                event.keystroke.key.as_str(),
                "backspace" | "delete" | "forwarddelete"
            ) {
                self.edit_audio(AudioEdit::Delete, cx);
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
        let color = alert.kind.color();
        let node = div()
            .absolute()
            .top(px(76.0))
            .left_5()
            .right_5()
            .flex()
            .justify_center()
            .child(
                div()
                    .w_full()
                    .max_w(px(400.0))
                    .px_4()
                    .py_3()
                    .rounded(theme::RADIUS)
                    .border_1()
                    .border_color(theme::mix(theme::LINE, color, 0.62))
                    .bg(theme::mix(theme::PANEL, color, 0.07))
                    .text_sm()
                    .text_color(theme::INK)
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .size(px(24.0))
                            .flex_none()
                            .rounded_full()
                            .bg(theme::mix(theme::PANEL, color, 0.18))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(alert.kind.icon().draw(px(15.0), color)),
                    )
                    .child(
                        div()
                            .min_w_0()
                            .flex_1()
                            .truncate()
                            .child(alert.text.clone()),
                    ),
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
        let pending = self
            .pending_active
            .and_then(|id| self.pending.iter().find(|pending| pending.id == id))
            .or_else(|| self.pending.iter().find(|pending| pending.path == path));
        let progress = pending.map_or(0.0, |pending| pending.progress);
        let previous = pending.map_or(progress, |pending| pending.previous);
        let stage = pending.map_or("Preparing", |pending| pending.stage);
        let motion = pending.map_or(0, |pending| pending.motion);
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("Audio")
            .to_owned();
        let bar = div()
            .w(relative(progress))
            .h_full()
            .rounded_full()
            .bg(linear_gradient(
                90.0,
                linear_color_stop(theme::ACCENT_SOFT, 0.0),
                linear_color_stop(theme::ACCENT, 1.0),
            ));
        let bar = if motion == 0 {
            bar.into_any_element()
        } else {
            bar.with_animation(
                ("inspect-progress", motion),
                Animation::new(TINT).with_easing(ease_in_out),
                move |bar, delta| bar.w(relative(previous + (progress - previous) * delta)),
            )
            .into_any_element()
        };
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
                    .w(px(240.0))
                    .h(px(6.0))
                    .relative()
                    .overflow_hidden()
                    .rounded_full()
                    .bg(theme::TRACK)
                    .border_1()
                    .border_color(theme::LINE)
                    .child(bar),
            )
            .child(
                div()
                    .mt_2()
                    .text_base()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::INK)
                    .child("Inspecting"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme::MUTED)
                    .child(format!("{name}  ·  {stage}  ·  {:.0}%", progress * 100.0)),
            )
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
            .child(self.transport(report, cx))
            .child(self.chain(report, cx))
            .into_any_element()
    }

    fn pending_tab(&self, pending: &Pending, cx: &mut Context<Self>) -> AnyElement {
        let id = pending.id;
        let active = self.pending_active == Some(id);
        let name = pending
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Audio")
            .to_owned();
        let progress = pending.progress;
        let previous = pending.previous;
        let motion = pending.motion;
        let bar = div()
            .absolute()
            .bottom_0()
            .left_0()
            .w(relative(progress))
            .h(px(2.0))
            .rounded_full()
            .bg(theme::ACCENT);
        let bar = if motion == 0 {
            bar.into_any_element()
        } else {
            let token = (id as usize).wrapping_mul(10_000).wrapping_add(motion);
            bar.with_animation(
                ("pending-progress", token),
                Animation::new(TINT).with_easing(ease_in_out),
                move |bar, delta| bar.w(relative(previous + (progress - previous) * delta)),
            )
            .into_any_element()
        };
        div()
            .id(("pending-tab", id as usize))
            .h_full()
            .min_w(px(116.0))
            .max_w(px(220.0))
            .relative()
            .overflow_hidden()
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
            .cursor_pointer()
            .hover(|node| node.bg(theme::HOVER))
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.activate_pending(id, cx);
            }))
            .child(
                div()
                    .min_w_0()
                    .flex_1()
                    .overflow_hidden()
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
                            .child(name),
                    )
                    .child(
                        div()
                            .mt(px(2.0))
                            .text_size(px(9.0))
                            .line_height(px(10.0))
                            .text_color(theme::FAINT)
                            .child(format!("{} · {:.0}%", pending.stage, progress * 100.0)),
                    ),
            )
            .child(
                div()
                    .id(("close-pending", id as usize))
                    .size(px(20.0))
                    .flex_none()
                    .rounded(theme::RADIUS)
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .hover(|node| node.bg(theme::TRACK))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|_this, _event, _window, cx| cx.stop_propagation()),
                    )
                    .on_click(cx.listener(move |this, _event, _window, cx| {
                        cx.stop_propagation();
                        this.close_pending(id, cx);
                    }))
                    .child(Icon::Close.draw(px(12.0), theme::MUTED)),
            )
            .child(bar)
            .into_any_element()
    }

    fn tab(&self, index: usize, tab: &Tab, cx: &mut Context<Self>) -> AnyElement {
        let active = self.active == Some(index) && matches!(self.state, State::Ready(_));
        let dirty = if active { self.dirty } else { tab.dirty };
        let closing = self
            .closing
            .iter()
            .find(|closing| closing.path == tab.path)
            .map(|closing| closing.token);
        let dragging = self.tab_dragging == Some(index);
        let path = tab.path.clone();
        let drag = TabDrag {
            path: path.clone(),
            name: tab.report.name(),
            meta: format!(
                "{} · {}",
                tab.report.format_text(),
                tab.report.duration_text()
            ),
            active,
            dirty,
            position: Point::default(),
        };
        let node =
            div()
                .id(("tab", index))
                .h_full()
                .min_w(px(116.0))
                .max_w(px(220.0))
                .relative()
                .overflow_hidden()
                .opacity(if dragging { 0.62 } else { 1.0 })
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
                            this.reorder_tab(from, index, f32::from(event.bounds.size.width), cx);
                        }
                    },
                ))
                .on_drop(cx.listener(|this, _drag: &TabDrag, _window, cx| {
                    this.finish_tab_drag(cx);
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
                                .min_w_0()
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_xs()
                                .line_height(px(12.0))
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .flex()
                                .items_center()
                                .gap(px(3.0))
                                .children(dirty.then(|| {
                                    div().flex_none().text_color(theme::ACCENT).child("*")
                                }))
                                .child(
                                    div()
                                        .min_w_0()
                                        .overflow_hidden()
                                        .text_color(if active { theme::INK } else { theme::MUTED })
                                        .child(tab.report.name()),
                                ),
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
                        .child(
                            Icon::Close
                                .draw(px(12.0), theme::MUTED)
                                .hover(|icon| icon.text_color(theme::INK)),
                        ),
                );
        let node = if tab.motion == 0 {
            node.into_any_element()
        } else {
            let shift = tab.shift;
            let token = tab.motion.wrapping_mul(1_000).wrapping_add(index);
            node.with_animation(
                ("tab-move", token),
                Animation::new(TAB).with_easing(ease_in_out),
                move |node, delta| node.left(px(shift * (1.0 - delta))),
            )
            .into_any_element()
        };
        let Some(token) = closing else {
            return node;
        };
        div()
            .h_full()
            .min_w(px(116.0))
            .max_w(px(220.0))
            .flex_none()
            .overflow_hidden()
            .child(node)
            .with_animation(
                ("tab-close", token),
                Animation::new(TAB).with_easing(ease_in_out),
                |shell, delta| {
                    let remaining = 1.0 - delta;
                    shell
                        .min_w(px(116.0 * remaining))
                        .max_w(px(220.0 * remaining))
                        .opacity(remaining)
                },
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
            .child(Icon::Add.draw(px(14.0), theme::MUTED));
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
        let tab_max = f32::from(self.tab_track.max_offset().x).max(0.0);
        let tab_offset = (-f32::from(self.tab_track.offset().x)).clamp(0.0, tab_max);
        let hidden_right = tab_max > 0.5 && tab_offset + 0.5 < tab_max;
        let tabs = div()
            .id("tabs")
            .min_w_0()
            .flex_1()
            .h_full()
            .overflow_x_scroll()
            .scrollbar_width(px(0.0))
            .track_scroll(&self.tab_track)
            .flex()
            .items_center()
            .on_scroll_wheel(cx.listener(|_this, _event, _window, cx| cx.notify()))
            .on_drop(cx.listener(|this, _drag: &TabDrag, _window, cx| {
                this.finish_tab_drag(cx);
            }))
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _event, _window, cx| this.finish_tab_drag(cx)),
            )
            .children(
                self.tabs
                    .iter()
                    .enumerate()
                    .map(|(index, tab)| self.tab(index, tab, cx)),
            )
            .children(
                self.pending
                    .iter()
                    .map(|pending| self.pending_tab(pending, cx)),
            );

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
                    .relative()
                    .min_w_0()
                    .flex_1()
                    .h_full()
                    .overflow_hidden()
                    .child(tabs)
                    .children(hidden_right.then(|| {
                        div()
                            .absolute()
                            .top_0()
                            .right_0()
                            .w(px(30.0))
                            .h_full()
                            .bg(linear_gradient(
                                90.0,
                                linear_color_stop(theme::CANVAS, 1.0),
                                linear_color_stop(
                                    gpui::Rgba {
                                        a: 0.0,
                                        ..theme::CANVAS
                                    },
                                    0.0,
                                ),
                            ))
                    })),
            )
            .child(
                div()
                    .id("training")
                    .size(px(26.0))
                    .flex_none()
                    .rounded(theme::RADIUS)
                    .text_color(if self.training_open {
                        theme::ACCENT
                    } else {
                        theme::MUTED
                    })
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                    .on_click(cx.listener(|this, _event, _window, cx| {
                        this.set_training_open(!this.training_open, cx);
                    }))
                    .child(Icon::Wave.draw(
                        px(15.0),
                        if self.training_open {
                            theme::ACCENT
                        } else {
                            theme::MUTED
                        },
                    )),
            )
            .child(
                div()
                    .id("settings")
                    .size(px(26.0))
                    .flex_none()
                    .rounded(theme::RADIUS)
                    .text_color(if self.settings_open {
                        theme::ACCENT
                    } else {
                        theme::MUTED
                    })
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .hover(|node| node.bg(theme::HOVER).text_color(theme::INK))
                    .on_click(cx.listener(|this, _event, _window, cx| {
                        this.set_settings_open(!this.settings_open, cx);
                    }))
                    .child(Icon::Settings.draw(
                        px(15.0),
                        if self.settings_open {
                            theme::ACCENT
                        } else {
                            theme::MUTED
                        },
                    )),
            )
            .child(
                div()
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(pressure("CPU", self.pressure.cpu))
                    .child(pressure("RAM", self.pressure.ram)),
            )
            .into_any_element()
    }

    fn settings_menu(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        (self.settings_open || self.settings_motion != 0).then(|| {
            let current = self
                .output
                .as_ref()
                .and_then(|id| self.outputs.iter().find(|output| &output.id == id));
            let summary = current
                .map(|output| format!("{} · {}", output.backend, output.name))
                .unwrap_or_else(|| "Operating system default".to_owned());

            let default_selected = self.output.is_none();
            let mut menu = div()
                .id("settings-menu")
                .absolute()
                .top(px(38.0))
                .right(px(112.0))
                .w(px(300.0))
                .max_h(px(430.0))
                .rounded(theme::RADIUS)
                .border_1()
                .border_color(theme::LINE)
                .bg(theme::PANEL)
                .shadow_md()
                .flex()
                .flex_col()
                .overflow_hidden()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|_this, _event, _window, cx| cx.stop_propagation()),
                )
                .child(
                    div()
                        .px_3()
                        .py_3()
                        .border_b_1()
                        .border_color(theme::LINE)
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_size(px(9.0))
                                .text_color(theme::FAINT)
                                .child("AUDIO OUTPUT"),
                        )
                        .child(
                            div()
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_sm()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(theme::INK)
                                .child(summary),
                        )
                        .child(
                            div()
                                .text_size(px(9.0))
                                .text_color(theme::FAINT)
                                .child("Playback uses rodio and CPAL"),
                        ),
                )
                .child(output_row(
                    "system-output",
                    "System Default".to_owned(),
                    "Follow the operating system output".to_owned(),
                    default_selected,
                    cx.listener(|this, _event, _window, cx| {
                        this.select_output(None, cx);
                    }),
                ));

            let list = div()
                .id("audio-output-list")
                .max_h(px(280.0))
                .overflow_y_scroll()
                .children(self.outputs.iter().enumerate().map(|(index, output)| {
                    let id = output.id.clone();
                    let selected = self.output.as_deref() == Some(output.id.as_str());
                    let detail = if output.default {
                        format!("{} · OS default", output.backend)
                    } else {
                        output.backend.clone()
                    };
                    output_row(
                        ("audio-output", index),
                        output.name.clone(),
                        detail,
                        selected,
                        cx.listener(move |this, _event, _window, cx| {
                            this.select_output(Some(id.clone()), cx);
                        }),
                    )
                }));
            menu = menu.child(list).child(
                div()
                    .px_3()
                    .py_2()
                    .border_t_1()
                    .border_color(theme::LINE)
                    .text_size(px(9.0))
                    .line_height(px(12.0))
                    .text_color(theme::FAINT)
                    .child(if cfg!(target_os = "windows") {
                        "WASAPI is built in. ASIO devices appear in builds made with --features asio."
                    } else if cfg!(target_os = "macos") {
                        "CoreAudio devices are available directly."
                    } else {
                        "Available CPAL output backends are shown above."
                    }),
            );
            let menu = if self.settings_motion == 0 {
                menu.into_any_element()
            } else {
                let opening = self.settings_open;
                menu.with_animation(
                    ("settings-menu", self.settings_motion),
                    Animation::new(TINT).with_easing(ease_in_out),
                    move |menu, delta| menu.opacity(if opening { delta } else { 1.0 - delta }),
                )
                .into_any_element()
            };
            div()
                .id("settings-dismiss")
                .absolute()
                .top_0()
                .right_0()
                .bottom_0()
                .left_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _event, _window, cx| {
                        this.set_settings_open(false, cx);
                    }),
                )
                .child(menu)
                .into_any_element()
        })
    }

    fn training_menu(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        (self.training_open || self.training_motion != 0).then(|| {
            let menu = div()
                .id("training-menu")
                .absolute()
                .top(px(38.0))
                .right(px(150.0))
                .w(px(264.0))
                .rounded(theme::RADIUS)
                .border_1()
                .border_color(theme::LINE)
                .bg(theme::PANEL)
                .shadow_md()
                .flex()
                .flex_col()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|_this, _event, _window, cx| cx.stop_propagation()),
                )
                .child(
                    div()
                        .px_3()
                        .py_3()
                        .border_b_1()
                        .border_color(theme::LINE)
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_size(px(9.0))
                                .text_color(theme::FAINT)
                                .child("ACTIVE INSPECTOR TRAINING"),
                        )
                        .child(
                            div()
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_sm()
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(theme::INK)
                                .child(self.training.name().to_owned()),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(if self.training.calibrated() {
                                    theme::ACCENT
                                } else {
                                    theme::MUTED
                                })
                                .child(self.training.summary()),
                        ),
                )
                .child(
                    div()
                        .id("import-clean")
                        .h(px(42.0))
                        .px_3()
                        .flex()
                        .items_center()
                        .cursor_pointer()
                        .hover(|node| node.bg(theme::HOVER))
                        .on_click(cx.listener(|this, _event, _window, cx| {
                            this.import_clean(cx);
                        }))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(div().text_sm().child("Import Clean Audio"))
                                .child(
                                    div()
                                        .text_size(px(9.0))
                                        .text_color(theme::FAINT)
                                        .child("Build and replace the clean reference"),
                                ),
                        ),
                )
                .child(
                    div()
                        .id("import-training")
                        .h(px(42.0))
                        .px_3()
                        .flex()
                        .items_center()
                        .cursor_pointer()
                        .hover(|node| node.bg(theme::HOVER))
                        .on_click(cx.listener(|this, _event, _window, cx| {
                            this.import_training(cx);
                        }))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(div().text_sm().child("Import Training File"))
                                .child(
                                    div()
                                        .text_size(px(9.0))
                                        .text_color(theme::FAINT)
                                        .child("Restore a portable .musp-training profile"),
                                ),
                        ),
                )
                .child(
                    div()
                        .id("restore-default-training")
                        .h(px(42.0))
                        .px_3()
                        .flex()
                        .items_center()
                        .cursor_pointer()
                        .hover(|node| node.bg(theme::HOVER))
                        .on_click(cx.listener(|this, _event, _window, cx| {
                            this.restore_default_training(cx);
                        }))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(div().text_sm().child("Restore Default Clean"))
                                .child(
                                    div()
                                        .text_size(px(9.0))
                                        .text_color(theme::FAINT)
                                        .child("Use the bundled clean reference"),
                                ),
                        ),
                )
                .child(
                    div()
                        .id("export-training")
                        .h(px(42.0))
                        .px_3()
                        .border_t_1()
                        .border_color(theme::LINE)
                        .flex()
                        .items_center()
                        .cursor_pointer()
                        .hover(|node| node.bg(theme::HOVER))
                        .on_click(cx.listener(|this, _event, _window, cx| {
                            this.export_training(cx);
                        }))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(div().text_sm().child("Export Current Training"))
                                .child(
                                    div()
                                        .text_size(px(9.0))
                                        .text_color(theme::FAINT)
                                        .child("Copy this profile to another computer"),
                                ),
                        ),
                );
            let menu = if self.training_motion == 0 {
                menu.into_any_element()
            } else {
                let opening = self.training_open;
                menu.with_animation(
                    ("training-menu", self.training_motion),
                    Animation::new(TINT).with_easing(ease_in_out),
                    move |menu, delta| menu.opacity(if opening { delta } else { 1.0 - delta }),
                )
                .into_any_element()
            };
            div()
                .id("training-dismiss")
                .absolute()
                .top_0()
                .right_0()
                .bottom_0()
                .left_0()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _event, _window, cx| {
                        this.set_training_open(false, cx);
                    }),
                )
                .child(menu)
                .into_any_element()
        })
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
                    .child(
                        Icon::Undo
                            .draw(px(13.0), theme::MUTED)
                            .hover(|icon| icon.text_color(theme::INK)),
                    ),
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
                    .child(
                        Icon::Redo
                            .draw(px(13.0), theme::MUTED)
                            .hover(|icon| icon.text_color(theme::INK)),
                    ),
            )
            .child(
                div()
                    .ml_1()
                    .size(px(14.0))
                    .text_color(theme::FAINT)
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(
                        if self.history_open {
                            Icon::Down
                        } else {
                            Icon::Up
                        }
                        .draw(px(12.0), theme::FAINT),
                    ),
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

    fn selection_context_menu(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let menu = self.selection_menu?;
        let can_paste = self.clipboard.is_some();
        Some(
            div()
                .id("selection-menu")
                .absolute()
                .left(menu.position.x)
                .top(menu.position.y)
                .w(px(196.0))
                .py_1()
                .rounded(theme::RADIUS)
                .border_1()
                .border_color(theme::LINE)
                .bg(theme::PANEL)
                .shadow_md()
                .flex()
                .flex_col()
                .cursor_default()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|_this, _event, _window, cx| cx.stop_propagation()),
                )
                .child(context_row(
                    "selection-copy",
                    "Copy",
                    shortcut("C"),
                    true,
                    cx.listener(|this, _event, _window, cx| this.copy_selection(cx)),
                ))
                .child(context_row(
                    "selection-paste",
                    "Paste",
                    shortcut("V"),
                    can_paste,
                    cx.listener(|this, _event, _window, cx| this.edit_audio(AudioEdit::Paste, cx)),
                ))
                .child(context_row(
                    "selection-delete",
                    "Delete",
                    "⌫".to_owned(),
                    true,
                    cx.listener(|this, _event, _window, cx| this.edit_audio(AudioEdit::Delete, cx)),
                ))
                .child(div().h(px(1.0)).my_1().bg(theme::LINE))
                .child(context_row(
                    "selection-export",
                    "Quick Export WAV",
                    shortcut("E"),
                    true,
                    cx.listener(|this, _event, _window, cx| this.quick_export(cx)),
                ))
                .child(context_row(
                    "selection-all",
                    "Select All",
                    shortcut("A"),
                    true,
                    cx.listener(|this, _event, _window, cx| this.select_all(cx)),
                ))
                .into_any_element(),
        )
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
        let playhead = self.playhead;
        let selection = self.selection;
        let bounds: Rc<RefCell<Option<Bounds<Pixels>>>> = Rc::new(RefCell::new(None));
        let capture = bounds.clone();
        let locate = bounds.clone();
        let begin = bounds.clone();
        let context = bounds.clone();
        let pan_begin = bounds.clone();
        let wheel = bounds.clone();
        #[cfg(any(target_os = "linux", target_os = "macos"))]
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

                        if let Some(position) = playhead
                            && (view..=view + visible).contains(&position)
                        {
                            let x = left + width * ((position - view) / visible);
                            let mut marker = PathBuilder::stroke(px(1.5));
                            marker.move_to(point(x, top));
                            marker.line_to(point(x, bottom));
                            if let Ok(path) = marker.build() {
                                window.paint_path(path, theme::ACCENT);
                            }
                            let mut head = PathBuilder::fill();
                            head.add_polygon(
                                &[
                                    point(x - px(4.0), top),
                                    point(x + px(4.0), top),
                                    point(x, top + px(6.0)),
                                ],
                                true,
                            );
                            if let Ok(path) = head.build() {
                                window.paint_path(path, theme::ACCENT);
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
                    this.selection_menu = None;
                    this.drag = Some(position);
                    this.selection = Some((position, position));
                    this.cursor = Some(position);
                    cx.notify();
                }),
            )
            .on_mouse_down(
                MouseButton::Right,
                cx.listener(move |this, event: &MouseDownEvent, _, cx| {
                    let Some(area) = *context.borrow() else {
                        return;
                    };
                    let position = this.view + horizontal(event.position.x, area) / this.zoom;
                    let inside = this
                        .selection
                        .map(|range| ordered(range.0, range.1))
                        .is_some_and(|range| (range.0..=range.1).contains(&position));
                    if !inside {
                        this.selection_menu = None;
                        cx.notify();
                        return;
                    }
                    let x = (f32::from(event.position.x - area.origin.x))
                        .clamp(4.0, (f32::from(area.size.width) - 200.0).max(4.0));
                    let y = (f32::from(event.position.y - area.origin.y))
                        .clamp(4.0, (f32::from(area.size.height) - 152.0).max(4.0));
                    this.selection_menu = Some(SelectionMenu {
                        position: point(px(x), px(y)),
                    });
                    cx.stop_propagation();
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
                    if let Some((start, end)) = this.selection
                        && (end - start).abs() <= minimum
                    {
                        this.selection = None;
                        this.looped = false;
                        this.seek((start + end) * 0.5, cx);
                        return;
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
            .children(self.selection_context_menu(cx));

        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let plot = plot.on_pinch(cx.listener(move |this, event: &PinchEvent, _, cx| {
            let Some(area) = *pinch.borrow() else {
                return;
            };
            if event.delta.abs() <= f32::EPSILON {
                return;
            }
            let anchor = horizontal(event.position.x, area);
            this.scale((1.0_f32 + event.delta).max(0.05_f32), anchor, cx);
            cx.stop_propagation();
        }));

        let plot = plot.on_hover(cx.listener(|this, hovered: &bool, _, cx| {
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

    fn transport(&self, report: &Report, cx: &mut Context<Self>) -> AnyElement {
        let playing = self
            .audio
            .as_ref()
            .is_some_and(|audio| !audio.paused() && !audio.empty());
        let loop_available = self
            .selection
            .is_some_and(|(start, end)| (end - start).abs() > f32::EPSILON);
        let position = self.playhead.unwrap_or(0.0).clamp(0.0, 1.0);
        let selection = self
            .selection
            .filter(|(start, end)| (end - start).abs() > f32::EPSILON)
            .map(|range| ordered(range.0, range.1))
            .map(|(start, end)| {
                format!(
                    "Loop {} – {}",
                    stamp(report.duration * f64::from(start)),
                    stamp(report.duration * f64::from(end))
                )
            });

        div()
            .id("transport")
            .h(px(34.0))
            .min_h(px(34.0))
            .w_full()
            .relative()
            .border_t_1()
            .border_b_1()
            .border_color(theme::LINE)
            .bg(theme::PANEL)
            .child(
                div()
                    .absolute()
                    .left_2()
                    .top_0()
                    .bottom_0()
                    .whitespace_nowrap()
                    .text_xs()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(theme::INK)
                    .flex()
                    .items_center()
                    .child(format!(
                        "{} / {}",
                        stamp(report.duration * f64::from(position)),
                        stamp(report.duration)
                    )),
            )
            .child(
                div()
                    .absolute()
                    .left(relative(0.5))
                    .top(px(4.0))
                    .ml(px(-26.0))
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(
                        div()
                            .id("playback")
                            .size(px(24.0))
                            .rounded(theme::RADIUS)
                            .bg(if playing {
                                theme::ACCENT_SOFT
                            } else {
                                theme::TRACK
                            })
                            .text_color(theme::INK)
                            .cursor_pointer()
                            .flex()
                            .items_center()
                            .justify_center()
                            .hover(|style| style.bg(theme::HOVER))
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_play(cx)))
                            .child(
                                if playing { Icon::Pause } else { Icon::Play }
                                    .draw(px(13.0), theme::INK),
                            ),
                    )
                    .child(
                        div()
                            .id("loop")
                            .size(px(24.0))
                            .rounded(theme::RADIUS)
                            .opacity(if loop_available { 1.0 } else { 0.42 })
                            .bg(if self.looped {
                                theme::ACCENT_SOFT
                            } else {
                                theme::TRACK
                            })
                            .text_color(theme::INK)
                            .cursor_pointer()
                            .flex()
                            .items_center()
                            .justify_center()
                            .hover(|style| style.bg(theme::HOVER))
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_loop(cx)))
                            .child(Icon::Loop.draw(px(13.0), theme::INK)),
                    ),
            )
            .children(selection.map(|selection| {
                div()
                    .absolute()
                    .right_2()
                    .top_0()
                    .bottom_0()
                    .text_xs()
                    .text_color(if self.looped {
                        theme::ACCENT
                    } else {
                        theme::MUTED
                    })
                    .flex()
                    .items_center()
                    .child(selection)
            }))
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
                                    .child(
                                        if expanded { Icon::Left } else { Icon::Right }
                                            .draw(
                                                px(11.0),
                                                if expanded {
                                                    theme::ACCENT
                                                } else {
                                                    theme::MUTED
                                                },
                                            )
                                            .hover(|icon| icon.text_color(theme::INK)),
                                    ),
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
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _event, _window, cx| {
                    if this.selection_menu.take().is_some() {
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _event, _window, cx| this.finish_tab_drag(cx)),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _event, _window, cx| this.finish_tab_drag(cx)),
            )
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
            .children(self.training_menu(cx))
            .children(self.settings_menu(cx))
    }
}

fn output_row(
    id: impl Into<ElementId>,
    name: String,
    detail: String,
    selected: bool,
    click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    div()
        .id(id)
        .h(px(44.0))
        .px_3()
        .flex_none()
        .flex()
        .items_center()
        .gap_2()
        .cursor_pointer()
        .hover(|node| node.bg(theme::HOVER))
        .on_click(click)
        .child(
            div()
                .size(px(8.0))
                .rounded_full()
                .flex_none()
                .bg(if selected { theme::ACCENT } else { theme::LINE }),
        )
        .child(
            div()
                .min_w_0()
                .flex_1()
                .flex()
                .flex_col()
                .child(
                    div()
                        .overflow_hidden()
                        .whitespace_nowrap()
                        .text_sm()
                        .text_color(if selected { theme::INK } else { theme::MUTED })
                        .child(name),
                )
                .child(
                    div()
                        .text_size(px(9.0))
                        .text_color(theme::FAINT)
                        .child(detail),
                ),
        )
        .into_any_element()
}

fn context_row(
    id: &'static str,
    label: &'static str,
    key: String,
    enabled: bool,
    click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let row = div()
        .id(id)
        .h(px(28.0))
        .px_2()
        .opacity(if enabled { 1.0 } else { 0.35 })
        .flex()
        .items_center()
        .justify_between()
        .text_xs()
        .text_color(theme::INK)
        .child(label)
        .child(div().text_size(px(9.0)).text_color(theme::FAINT).child(key));
    if enabled {
        row.cursor_pointer()
            .hover(|node| node.bg(theme::HOVER))
            .on_click(click)
            .into_any_element()
    } else {
        row.into_any_element()
    }
}

fn shortcut(key: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("⌘{key}")
    } else {
        format!("Ctrl+{key}")
    }
}

fn selection_path(source: &Path) -> PathBuf {
    let parent = source.parent().unwrap_or_else(|| Path::new("."));
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    for suffix in 1..=9_999 {
        let name = if suffix == 1 {
            format!("{stem}-selection.wav")
        } else {
            format!("{stem}-selection-{suffix}.wav")
        };
        let target = parent.join(name);
        if !target.exists() {
            return target;
        }
    }
    parent.join(format!("{stem}-selection-{}.wav", std::process::id()))
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
                .child(step(effect, index, -1.0, cx))
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
                .child(step(effect, index, 1.0, cx)),
        )
        .into_any_element()
}

fn step(effect: usize, param: usize, direction: f64, cx: &mut Context<Muspector>) -> AnyElement {
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
        .child(
            if direction > 0.0 {
                Icon::Add
            } else {
                Icon::Remove
            }
            .draw(px(10.0), theme::MUTED)
            .hover(|icon| icon.text_color(theme::INK)),
        )
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

fn refresh_process(system: &mut System) -> Pressure {
    let pid = Pid::from_u32(std::process::id());
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        false,
        ProcessRefreshKind::nothing().with_cpu().with_memory(),
    );
    let Some(process) = system.process(pid) else {
        return Pressure::default();
    };

    let cores = std::thread::available_parallelism()
        .map(|cores| cores.get())
        .unwrap_or(1) as f32;
    let cpu = (process.cpu_usage() / cores).clamp(0.0, 100.0);
    let (available, capacity) = memory_headroom(system, process.memory());

    Pressure {
        cpu: pressure_level(cpu),
        ram: memory_level(available, capacity),
    }
}

#[cfg(target_os = "macos")]
fn memory_headroom(system: &System, process_memory: u64) -> (u64, u64) {
    // XNU's available-memory metric includes reclaimable pages. Adding the
    // process RSS yields the capacity the process could occupy before that
    // current headroom is exhausted.
    let available = system.available_memory();
    (available, available.saturating_add(process_memory))
}

#[cfg(not(target_os = "macos"))]
fn memory_headroom(system: &System, process_memory: u64) -> (u64, u64) {
    #[cfg(target_os = "linux")]
    if let Some(limits) = system.cgroup_limits()
        && limits.total_memory > 0
        && limits.total_memory < system.total_memory()
    {
        return (
            limits.total_memory.saturating_sub(process_memory),
            limits.total_memory,
        );
    }

    let total = system.total_memory();
    (total.saturating_sub(process_memory), total)
}

fn memory_level(available: u64, total: u64) -> usize {
    const GIB: u64 = 1_024 * 1_024 * 1_024;

    if total == 0 {
        return 0;
    }

    // Base memory pressure on the process's remaining headroom instead of
    // system-wide "used" RAM. The absolute caps prevent large-memory machines
    // from turning red while several GiB remain.
    let available = available.min(total);
    let critical = (total / 16).min(GIB);
    let warning = (total / 5).min(4 * GIB);

    if available <= critical {
        3
    } else if available <= warning {
        2
    } else {
        1
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

    fn revision(chain: Chain) -> Rc<Revision> {
        Rc::new(Revision {
            report: Box::new(Report {
                path: PathBuf::from("test.wav"),
                codec: "WAV".to_owned(),
                rate: 48_000,
                channels: 1,
                duration: 1.0,
                peak: -1.0,
                rms: -12.0,
                loudness: -13.0,
                crest: 11.0,
                centroid: 1_000.0,
                rolloff: 4_000.0,
                low: -6.0,
                mid: -4.0,
                high: -14.0,
                clips: 0,
                spectrum: Vec::new(),
                profile: analysis::Profile { points: Vec::new() },
                chain,
            }),
            source: PathBuf::from("test.wav"),
            audio_dirty: false,
        })
    }

    fn snapshot(chain: Chain, baseline: Chain, revision: Rc<Revision>, dirty: bool) -> Snapshot {
        Snapshot {
            chain,
            revision,
            baseline,
            dirty,
            expanded: [false; 6],
            selection: None,
            playhead: None,
            looped: false,
        }
    }

    #[test]
    fn history_merges_gestures_and_discards_redo() {
        let baseline = chain();
        let revision = revision(baseline.clone());
        let mut history = History::detected(snapshot(
            baseline.clone(),
            baseline.clone(),
            revision.clone(),
            false,
        ));
        let mut adjusted = baseline.clone();
        adjusted.effects[0].params[0].shift(1.0);
        history.record(
            "Adjust Gate Threshold".to_owned(),
            snapshot(adjusted.clone(), baseline.clone(), revision.clone(), true),
            true,
        );
        adjusted.effects[0].params[0].shift(1.0);
        history.record(
            "Adjust Gate Threshold".to_owned(),
            snapshot(adjusted, baseline.clone(), revision.clone(), true),
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
            snapshot(toggled, baseline, revision, true),
            false,
        );
        assert_eq!(history.entries.len(), 2);
        assert_eq!(history.entries[1].label, "Enable Gate");
    }

    #[test]
    fn history_keeps_audio_revisions_for_undo_and_redo() {
        let baseline = chain();
        let original = revision(baseline.clone());
        let mut history = History::detected(snapshot(
            baseline.clone(),
            baseline.clone(),
            original.clone(),
            false,
        ));
        let edited = Rc::new(Revision {
            report: original.report.clone(),
            source: PathBuf::from("edited.wav"),
            audio_dirty: true,
        });

        history.record(
            "Delete selection".to_owned(),
            snapshot(baseline.clone(), baseline, edited.clone(), true),
            false,
        );

        assert_eq!(history.entries.len(), 2);
        assert!(Rc::ptr_eq(&history.entries[0].snapshot.revision, &original));
        assert!(Rc::ptr_eq(&history.entries[1].snapshot.revision, &edited));
        assert_eq!(history.entries[1].label, "Delete selection");
        assert!(history.entries[1].snapshot.revision.audio_dirty);
    }

    #[test]
    fn memory_pressure_uses_remaining_headroom() {
        const GIB: u64 = 1_024 * 1_024 * 1_024;

        assert_eq!(memory_level(10 * GIB, 16 * GIB), 1);
        assert_eq!(memory_level(2 * GIB, 16 * GIB), 2);
        assert_eq!(memory_level(768 * 1_024 * 1_024, 16 * GIB), 3);
    }

    #[test]
    fn large_memory_systems_keep_absolute_reserve() {
        const GIB: u64 = 1_024 * 1_024 * 1_024;

        assert_eq!(memory_level(5 * GIB, 64 * GIB), 1);
        assert_eq!(memory_level(3 * GIB, 64 * GIB), 2);
        assert_eq!(memory_level(768 * 1_024 * 1_024, 64 * GIB), 3);
        assert_eq!(memory_level(0, 0), 0);
    }
}
