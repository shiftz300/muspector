use anyhow::Result;
use gpui::{AssetSource, SharedString};
use std::borrow::Cow;

pub struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        if let Some(path) = path
            .strip_prefix("labels/")
            .and_then(|path| path.strip_suffix(".svg"))
            && let Some((style, encoded)) = path.split_once('/')
            && let Some(label) = decode(encoded)
        {
            let svg = if style == "title" {
                let lines = wrap(&label);
                let width = lines
                    .iter()
                    .map(|line| line.chars().count() * 7 + 8)
                    .max()
                    .unwrap_or(64)
                    .clamp(58, 98);
                let center = width / 2;
                let spans = lines
                    .iter()
                    .enumerate()
                    .map(|(index, line)| {
                        format!(
                            r#"<tspan x="{center}" y="{}">{}</tspan>"#,
                            11 + index * 13,
                            escape(line)
                        )
                    })
                    .collect::<String>();
                let height = if lines.len() > 1 { 27 } else { 16 };
                format!(
                    r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><text text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="700" fill="black">{spans}</text></svg>"#
                )
            } else {
                format!(
                    r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 124 18"><text x="62" y="13" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="12" font-weight="500" fill="black">{}</text></svg>"#,
                    escape(&label)
                )
            };
            return Ok(Some(Cow::Owned(svg.into_bytes())));
        }
        match path {
            "icons/audio.svg" => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/audio.svg"
            )))),
            "icons/undo.svg" => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/undo.svg"
            )))),
            "icons/redo.svg" => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/redo.svg"
            )))),
            "icons/chevron-up.svg" => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/chevron-up.svg"
            )))),
            "icons/chevron-down.svg" => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/chevron-down.svg"
            )))),
            _ => Ok(None),
        }
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        Ok(match path {
            "icons" => vec![
                "icons/audio.svg".into(),
                "icons/undo.svg".into(),
                "icons/redo.svg".into(),
                "icons/chevron-up.svg".into(),
                "icons/chevron-down.svg".into(),
            ],
            _ => Vec::new(),
        })
    }
}

fn wrap(label: &str) -> Vec<String> {
    let words = label.split_whitespace().collect::<Vec<_>>();
    if words.len() < 2 {
        return vec![label.to_owned()];
    }
    let total = words.iter().map(|word| word.len()).sum::<usize>() + words.len() - 1;
    let mut split = 1;
    let mut best = usize::MAX;
    for index in 1..words.len() {
        let left = words[..index].join(" ").len();
        let right = total.saturating_sub(left + 1);
        let balance = left.abs_diff(right);
        if balance < best {
            best = balance;
            split = index;
        }
    }
    vec![words[..split].join(" "), words[split..].join(" ")]
}

fn escape(label: &str) -> String {
    label
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn decode(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = *bytes.get(index + 1)?;
            let low = *bytes.get(index + 2)?;
            output.push(hex(high)? << 4 | hex(low)?);
            index += 3;
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(output).ok()
}

fn hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}
