use anyhow::{Context, Result, bail};
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{Decoder as AudioDecoder, DecoderOptions},
    errors::Error,
    formats::{FormatOptions, FormatReader},
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
use tempfile::{Builder, TempPath};

#[derive(Clone)]
pub struct TemporaryAudio {
    _path: Arc<TempPath>,
}

#[derive(Clone)]
pub struct Clip {
    pub path: PathBuf,
    pub rate: u32,
    pub channels: usize,
    pub frames: u64,
    _owner: TemporaryAudio,
}

impl Clip {
    pub fn duration(&self) -> f64 {
        self.frames as f64 / f64::from(self.rate)
    }
}

pub struct Edit {
    pub path: PathBuf,
    pub(crate) owner: TemporaryAudio,
}

struct Source {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track: u32,
    rate: u32,
    channels: usize,
    buffer: Option<SampleBuffer<f32>>,
}

impl Source {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("could not open audio source {}", path.display()))?;
        let stream = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
            hint.with_extension(extension);
        }
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                stream,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .context("unsupported audio source")?;
        let format = probed.format;
        let track = format
            .default_track()
            .context("audio source has no track")?;
        let id = track.id;
        let params = &track.codec_params;
        let rate = params
            .sample_rate
            .context("audio source has no sample rate")?;
        let channels = params
            .channels
            .map(|channels| channels.count())
            .context("audio source has no channel layout")?;
        let decoder = symphonia::default::get_codecs()
            .make(params, &DecoderOptions::default())
            .context("audio source has no decoder")?;
        Ok(Self {
            format,
            decoder,
            track: id,
            rate,
            channels,
            buffer: None,
        })
    }

    fn next(&mut self) -> Result<Option<Vec<f32>>> {
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                    return Ok(None);
                }
                Err(Error::ResetRequired) => bail!("audio format changed while editing"),
                Err(error) => return Err(error).context("could not read audio while editing"),
            };
            if packet.track_id() != self.track {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(Error::DecodeError(_)) => continue,
                Err(error) => return Err(error).context("could not decode audio while editing"),
            };
            let spec = *decoded.spec();
            let capacity = decoded.capacity() as u64;
            let buffer = self
                .buffer
                .get_or_insert_with(|| SampleBuffer::<f32>::new(capacity, spec));
            buffer.copy_interleaved_ref(decoded);
            return Ok(Some(buffer.samples().to_vec()));
        }
    }
}

struct Wav {
    file: File,
    rate: u32,
    channels: usize,
    samples: u64,
}

impl Wav {
    fn new(mut file: File, rate: u32, channels: usize) -> Result<Self> {
        if channels == 0 || channels > u16::MAX as usize {
            bail!("unsupported channel count");
        }
        file.write_all(&[0; 44])?;
        Ok(Self {
            file,
            rate,
            channels,
            samples: 0,
        })
    }

    fn write(&mut self, samples: &[f32]) -> Result<()> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            bail!("refusing to write non-finite audio samples");
        }
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        self.file.write_all(&bytes)?;
        self.samples = self.samples.saturating_add(samples.len() as u64);
        Ok(())
    }

    fn finish(mut self) -> Result<u64> {
        let data = self.samples.checked_mul(4).context("WAV is too large")?;
        let data = u32::try_from(data).context("WAV exceeds the 4 GB RIFF limit")?;
        let channels = self.channels as u16;
        let align = channels
            .checked_mul(4)
            .context("invalid WAV block alignment")?;
        let byte_rate = self
            .rate
            .checked_mul(u32::from(align))
            .context("invalid WAV byte rate")?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(b"RIFF")?;
        self.file.write_all(
            &36_u32
                .checked_add(data)
                .context("WAV exceeds the RIFF size limit")?
                .to_le_bytes(),
        )?;
        self.file.write_all(b"WAVEfmt ")?;
        self.file.write_all(&16_u32.to_le_bytes())?;
        self.file.write_all(&3_u16.to_le_bytes())?;
        self.file.write_all(&channels.to_le_bytes())?;
        self.file.write_all(&self.rate.to_le_bytes())?;
        self.file.write_all(&byte_rate.to_le_bytes())?;
        self.file.write_all(&align.to_le_bytes())?;
        self.file.write_all(&32_u16.to_le_bytes())?;
        self.file.write_all(b"data")?;
        self.file.write_all(&data.to_le_bytes())?;
        self.file.flush()?;
        Ok(self.samples / self.channels as u64)
    }
}

fn temporary() -> Result<(PathBuf, TemporaryAudio, File)> {
    let named = Builder::new()
        .prefix("muspector-")
        .suffix(".wav")
        .tempfile()
        .context("could not create a temporary audio file")?;
    let (file, path) = named.into_parts();
    let location = path.to_path_buf();
    Ok((
        location,
        TemporaryAudio {
            _path: Arc::new(path),
        },
        file,
    ))
}

fn frame(time: f64, rate: u32) -> u64 {
    (time.max(0.0) * f64::from(rate)).round() as u64
}

fn write_range(source: &Path, file: File, start: f64, end: f64) -> Result<(u32, usize, u64)> {
    let mut source = Source::open(source)?;
    let from = frame(start, source.rate);
    let to = frame(end, source.rate).max(from);
    if to <= from {
        bail!("select a non-empty audio range");
    }
    let mut writer = Wav::new(file, source.rate, source.channels)?;
    let mut position = 0_u64;
    while let Some(samples) = source.next()? {
        let count = samples.len() / source.channels;
        let packet_end = position.saturating_add(count as u64);
        if packet_end > from && position < to {
            let local_from = from.saturating_sub(position).min(count as u64) as usize;
            let local_to = to
                .min(packet_end)
                .saturating_sub(position)
                .min(count as u64) as usize;
            writer.write(&samples[local_from * source.channels..local_to * source.channels])?;
        }
        position = packet_end;
        if position >= to {
            break;
        }
    }
    let rate = source.rate;
    let channels = source.channels;
    let frames = writer.finish()?;
    if frames == 0 {
        bail!("selected range contains no decodable audio");
    }
    Ok((rate, channels, frames))
}

fn append(writer: &mut Wav, path: &Path, rate: u32, channels: usize) -> Result<()> {
    let mut source = Source::open(path)?;
    if source.rate != rate || source.channels != channels {
        bail!("clipboard sample rate or channel layout does not match this file");
    }
    while let Some(samples) = source.next()? {
        writer.write(&samples)?;
    }
    Ok(())
}

fn rewrite(source_path: &Path, start: f64, end: f64, insert: Option<&Clip>) -> Result<Edit> {
    let mut source = Source::open(source_path)?;
    if let Some(insert) = insert
        && (insert.rate != source.rate || insert.channels != source.channels)
    {
        bail!("clipboard sample rate or channel layout does not match this file");
    }
    let from = frame(start, source.rate);
    let to = frame(end, source.rate).max(from);
    let (target, owner, file) = temporary()?;
    let mut writer = Wav::new(file, source.rate, source.channels)?;
    let mut position = 0_u64;
    let mut inserted = false;
    while let Some(samples) = source.next()? {
        let count = samples.len() / source.channels;
        let packet_end = position.saturating_add(count as u64);
        let before = from.min(packet_end).saturating_sub(position) as usize;
        if before > 0 {
            writer.write(&samples[..before * source.channels])?;
        }
        if !inserted && packet_end >= from {
            if let Some(insert) = insert {
                append(&mut writer, &insert.path, source.rate, source.channels)?;
            }
            inserted = true;
        }
        if packet_end > to {
            let after = to.saturating_sub(position).min(count as u64) as usize;
            writer.write(&samples[after * source.channels..])?;
        }
        position = packet_end;
    }
    if !inserted && let Some(insert) = insert {
        append(&mut writer, &insert.path, source.rate, source.channels)?;
    }
    let frames = writer.finish()?;
    if frames == 0 {
        bail!("an audio document cannot be empty");
    }
    Ok(Edit {
        path: target,
        owner,
    })
}

pub fn copy(source: &Path, start: f64, end: f64) -> Result<Clip> {
    let (path, owner, file) = temporary()?;
    let (rate, channels, frames) = write_range(source, file, start, end)?;
    Ok(Clip {
        path,
        rate,
        channels,
        frames,
        _owner: owner,
    })
}

pub fn export(source: &Path, target: &Path, start: f64, end: f64) -> Result<()> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    let named = Builder::new()
        .prefix(".muspector-export-")
        .suffix(".wav")
        .tempfile_in(parent)
        .with_context(|| format!("could not create an export beside {}", target.display()))?;
    let (file, path) = named.into_parts();
    write_range(source, file, start, end)?;
    path.persist_noclobber(target)
        .map_err(|error| error.error)
        .with_context(|| format!("could not create export {}", target.display()))
}

pub fn delete(source: &Path, start: f64, end: f64) -> Result<Edit> {
    rewrite(source, start, end, None)
}

pub fn paste(source: &Path, clip: &Clip, start: f64, end: f64) -> Result<Edit> {
    rewrite(source, start, end, Some(clip))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(path: &Path) -> u64 {
        let mut source = Source::open(path).expect("open generated WAV");
        let channels = source.channels;
        let mut total = 0_u64;
        while let Some(samples) = source.next().expect("decode generated WAV") {
            total += (samples.len() / channels) as u64;
        }
        total
    }

    fn samples(path: &Path) -> Vec<f32> {
        let mut source = Source::open(path).expect("open generated WAV");
        let mut result = Vec::new();
        while let Some(packet) = source.next().expect("decode generated WAV") {
            result.extend(packet);
        }
        result
    }

    #[test]
    fn range_edits_preserve_frame_boundaries() {
        let (source, _source_owner, file) = temporary().expect("create source");
        let mut writer = Wav::new(file, 1_000, 1).expect("create source");
        writer
            .write(
                &(0..1_000)
                    .map(|index| index as f32 / 1_000.0)
                    .collect::<Vec<_>>(),
            )
            .expect("write source");
        writer.finish().expect("finish source");

        let copied = copy(&source, 0.2, 0.4).expect("copy range");
        assert_eq!(copied.frames, 200);
        assert_eq!(frames(&copied.path), 200);

        let deleted = delete(&source, 0.2, 0.4).expect("delete range");
        assert_eq!(frames(&deleted.path), 800);

        let pasted = paste(&source, &copied, 0.5, 0.5).expect("paste range");
        assert_eq!(frames(&pasted.path), 1_200);
    }

    #[test]
    fn float_edits_preserve_samples_without_normalizing_or_clipping() {
        let (source, _source_owner, file) = temporary().expect("create source");
        let original = [-1.25_f32, -0.5, 0.0, 0.5, 1.25];
        let mut writer = Wav::new(file, 48_000, 1).expect("create source");
        writer.write(&original).expect("write source");
        writer.finish().expect("finish source");
        let source_bytes = std::fs::read(&source).expect("snapshot source");

        let copied =
            copy(&source, 0.0, original.len() as f64 / 48_000.0).expect("copy whole source");
        assert_eq!(samples(&copied.path), original);
        assert_eq!(std::fs::read(&source).expect("reread source"), source_bytes);

        let (_invalid, _invalid_owner, file) = temporary().expect("create invalid WAV");
        let mut invalid_writer = Wav::new(file, 48_000, 1).expect("create invalid WAV");
        assert!(invalid_writer.write(&[f32::NAN]).is_err());
    }

    #[test]
    fn temporary_audio_lives_until_the_last_owner_is_dropped() {
        let (source, _source_owner, file) = temporary().expect("create source");
        let mut writer = Wav::new(file, 48_000, 1).expect("create source");
        writer.write(&[0.0, 0.25, -0.25]).expect("write source");
        writer.finish().expect("finish source");

        let clip = copy(&source, 0.0, 3.0 / 48_000.0).expect("copy source");
        let path = clip.path.clone();
        let retained = clip.clone();
        drop(clip);
        assert!(path.exists());
        drop(retained);
        assert!(!path.exists());
    }
}
