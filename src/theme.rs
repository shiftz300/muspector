use gpui::{Pixels, Rgba, px};

pub const RADIUS: Pixels = px(10.0);

pub const CANVAS: Rgba = color(0x141414);
pub const PANEL: Rgba = color(0x1a1a1a);
pub const SURFACE: Rgba = color(0x212121);
pub const TRACK: Rgba = color(0x191919);
pub const HOVER: Rgba = color(0x2a2a2a);
pub const INK: Rgba = color(0xf0eee9);
pub const MUTED: Rgba = color(0xa19e97);
pub const FAINT: Rgba = color(0x6f6c66);
pub const LINE: Rgba = color(0x353535);
pub const ACCENT: Rgba = color(0xe58aae);
pub const ACCENT_HOVER: Rgba = color(0xf09bbb);
pub const ACCENT_SOFT: Rgba = color(0x38232e);
pub const GOOD: Rgba = color(0x70b58a);
pub const WARN: Rgba = color(0xd2aa65);
pub const ERROR: Rgba = color(0xe0746a);

const fn color(hex: u32) -> Rgba {
    Rgba {
        r: ((hex >> 16) & 0xff) as f32 / 255.0,
        g: ((hex >> 8) & 0xff) as f32 / 255.0,
        b: (hex & 0xff) as f32 / 255.0,
        a: 1.0,
    }
}

pub fn mix(from: Rgba, to: Rgba, amount: f32) -> Rgba {
    let amount = amount.clamp(0.0, 1.0);
    Rgba {
        r: from.r + (to.r - from.r) * amount,
        g: from.g + (to.g - from.g) * amount,
        b: from.b + (to.b - from.b) * amount,
        a: from.a + (to.a - from.a) * amount,
    }
}
