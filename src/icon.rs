use gpui::{Hsla, Pixels, Styled, Svg, svg};

#[derive(Clone, Copy)]
pub enum Icon {
    Add,
    Close,
    Down,
    Left,
    Loop,
    Pause,
    Play,
    Redo,
    Remove,
    Right,
    Undo,
    Up,
}

impl Icon {
    pub fn draw(self, size: Pixels, color: impl Into<Hsla>) -> Svg {
        svg()
            .size(size)
            .flex_none()
            .text_color(color)
            .path(self.path())
    }

    fn path(self) -> &'static str {
        match self {
            Self::Add => "icons/plus.svg",
            Self::Close => "icons/x.svg",
            Self::Down => "icons/chevron-down.svg",
            Self::Left => "icons/chevron-left.svg",
            Self::Loop => "icons/rotate-cw.svg",
            Self::Pause => "icons/pause.svg",
            Self::Play => "icons/play.svg",
            Self::Redo => "icons/redo.svg",
            Self::Remove => "icons/minus.svg",
            Self::Right => "icons/chevron-right.svg",
            Self::Undo => "icons/undo.svg",
            Self::Up => "icons/chevron-up.svg",
        }
    }
}
