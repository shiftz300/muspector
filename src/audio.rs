use anyhow::{Context, Result};
use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct Audio {
    _sink: MixerDeviceSink,
    player: Player,
    path: PathBuf,
}

impl Audio {
    pub fn open(path: &Path, position: Duration) -> Result<Self> {
        let sink =
            DeviceSinkBuilder::open_default_sink().context("no audio output is available")?;
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
