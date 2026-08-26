use anyhow::Result;
use gpui::{AssetSource, SharedString};
use std::borrow::Cow;

pub struct Assets;

impl AssetSource for Assets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        match path {
            "icons/audio.svg" => Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/audio.svg"
            )))),
            _ => Ok(None),
        }
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        Ok(match path {
            "icons" => vec!["icons/audio.svg".into()],
            _ => Vec::new(),
        })
    }
}
