use anyhow::{Context, Result};
use rodio::cpal::{
    DeviceId,
    traits::{DeviceTrait, HostTrait},
};
use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub id: String,
    pub name: String,
    pub backend: String,
    pub default: bool,
}

pub struct Audio {
    _sink: MixerDeviceSink,
    player: Player,
    path: PathBuf,
}

impl Audio {
    pub fn open(path: &Path, position: Duration, output: Option<&str>) -> Result<Self> {
        let sink = match output {
            Some(output) => {
                let id = DeviceId::from_str(output).context("saved audio device is invalid")?;
                let host = rodio::cpal::host_from_id(id.0)
                    .context("selected audio backend is unavailable")?;
                let device = host
                    .device_by_id(&id)
                    .context("selected audio output is unavailable")?;
                DeviceSinkBuilder::from_device(device)
                    .context("selected audio output has no compatible format")?
                    .open_stream()
                    .context("could not open the selected audio output")?
            }
            None => {
                DeviceSinkBuilder::open_default_sink().context("no audio output is available")?
            }
        };
        let player = Player::connect_new(sink.mixer());
        let file =
            File::open(path).with_context(|| format!("could not open {}", path.display()))?;
        player.append(Decoder::try_from(file).context("could not decode audio for playback")?);
        player.pause();
        if !position.is_zero() {
            player
                .try_seek(position)
                .context("could not seek to the playback position")?;
        }
        Ok(Self {
            _sink: sink,
            player,
            path: path.to_owned(),
        })
    }

    pub fn matches(&self, path: &Path) -> bool {
        self.path == path
    }

    pub fn play(&self) {
        self.player.play();
    }

    pub fn pause(&self) {
        self.player.pause();
    }

    pub fn paused(&self) -> bool {
        self.player.is_paused()
    }

    pub fn empty(&self) -> bool {
        self.player.empty()
    }

    pub fn position(&self) -> Duration {
        self.player.get_pos()
    }

    pub fn seek(&self, position: Duration) -> Result<()> {
        self.player
            .try_seek(position)
            .context("could not seek during playback")
    }
}

pub fn outputs() -> Vec<Output> {
    let mut result = Vec::new();
    for host_id in rodio::cpal::available_hosts() {
        let Ok(host) = rodio::cpal::host_from_id(host_id) else {
            continue;
        };
        let default = host
            .default_output_device()
            .and_then(|device| device.id().ok());
        let Ok(devices) = host.output_devices() else {
            continue;
        };
        for device in devices {
            let Ok(id) = device.id() else {
                continue;
            };
            let name = device
                .description()
                .map(|description| description.name().to_owned())
                .unwrap_or_else(|_| "Audio output".to_owned());
            result.push(Output {
                id: id.to_string(),
                name,
                backend: backend(host_id),
                default: default.as_ref() == Some(&id),
            });
        }
    }
    result.sort_by(|left, right| {
        right
            .default
            .cmp(&left.default)
            .then_with(|| left.backend.cmp(&right.backend))
            .then_with(|| left.name.cmp(&right.name))
    });
    result
}

fn backend(id: rodio::cpal::HostId) -> String {
    match id.to_string().as_str() {
        "CoreAudio" => "CoreAudio".to_owned(),
        "Wasapi" => "WASAPI".to_owned(),
        "Asio" => "ASIO".to_owned(),
        "Alsa" => "ALSA".to_owned(),
        "Jack" => "JACK".to_owned(),
        other => other.to_owned(),
    }
}

pub fn load_output() -> Option<String> {
    fs::read_to_string(output_path()?).ok().and_then(|value| {
        let value = value.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

pub fn save_output(output: Option<&str>) -> Result<()> {
    let path = output_path().context("could not locate the audio settings directory")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("could not create the audio settings directory")?;
    }
    fs::write(path, output.unwrap_or_default()).context("could not save the audio output setting")
}

fn output_path() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    let root = std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library/Application Support/Muspector"));
    #[cfg(target_os = "windows")]
    let root = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .map(|path| path.join("Muspector"));
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let root = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        })
        .map(|path| path.join("muspector"));
    root.map(|path| path.join("audio-output"))
}
