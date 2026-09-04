use anyhow::{Context, Result, bail};
use crossbeam_queue::ArrayQueue;
use rodio::cpal::{
    BufferSize, DeviceId, SupportedBufferSize,
    traits::{DeviceTrait, HostTrait},
};
use rodio::{
    ChannelCount, Decoder, DeviceSinkBuilder, MixerDeviceSink, Player, SampleRate, Source,
    source::{SeekError, UniformSourceIterator},
};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;

pub const COMMON_BUFFERS: [u32; 5] = [128, 256, 512, 1_024, 2_048];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub id: String,
    pub name: String,
    pub backend: String,
    pub default: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputTiming {
    pub sample_rate: u32,
    pub automatic_buffer: u32,
}

const NO_SEEK: u64 = u64::MAX;
const MAX_OUTPUT_CHANNELS: usize = 32;

#[derive(Clone, Copy)]
struct QueuedFrame {
    epoch: u64,
    samples: [f32; MAX_OUTPUT_CHANNELS],
}

struct PlaybackClock {
    rate: u32,
    channels: usize,
    queue: Arc<ArrayQueue<QueuedFrame>>,
    position: AtomicU64,
    seek: AtomicU64,
    epoch: AtomicU64,
    loop_start: AtomicU64,
    loop_end: AtomicU64,
    looped: AtomicBool,
    ended: AtomicBool,
}

impl PlaybackClock {
    fn new(rate: u32, channels: usize, position: Duration, queue_frames: usize) -> Self {
        Self {
            rate,
            channels,
            queue: Arc::new(ArrayQueue::new(queue_frames)),
            position: AtomicU64::new(frames(position, rate)),
            seek: AtomicU64::new(NO_SEEK),
            epoch: AtomicU64::new(1),
            loop_start: AtomicU64::new(0),
            loop_end: AtomicU64::new(0),
            looped: AtomicBool::new(false),
            ended: AtomicBool::new(false),
        }
    }

    fn position(&self) -> Duration {
        duration(self.position.load(Ordering::Relaxed), self.rate)
    }

    fn set_loop(&self, range: Option<(Duration, Duration)>) {
        let range = range.and_then(|(start, end)| {
            let start = frames(start, self.rate);
            let end = frames(end, self.rate);
            (end > start).then_some((start, end))
        });
        let enabled = range.is_some();
        let start = range.map_or(0, |range| range.0);
        let end = range.map_or(0, |range| range.1);
        if self.looped.load(Ordering::Acquire) == enabled
            && self.loop_start.load(Ordering::Relaxed) == start
            && self.loop_end.load(Ordering::Relaxed) == end
        {
            return;
        }
        self.loop_start.store(start, Ordering::Relaxed);
        self.loop_end.store(end, Ordering::Relaxed);
        self.looped.store(enabled, Ordering::Release);

        let current = self.position.load(Ordering::Relaxed);
        let position = if enabled && (current < start || current >= end) {
            start
        } else {
            current
        };
        self.request_seek(position);
    }

    fn request_seek(&self, position: u64) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        while self.queue.pop().is_some() {}
        self.position.store(position, Ordering::Relaxed);
        self.ended.store(false, Ordering::Release);
        self.seek.store(position, Ordering::Release);
    }
}

struct RingSource {
    clock: Arc<PlaybackClock>,
    frame: QueuedFrame,
    sample: usize,
    real: bool,
}

impl RingSource {
    fn new(clock: Arc<PlaybackClock>) -> Self {
        Self {
            clock,
            frame: QueuedFrame {
                epoch: 0,
                samples: [0.0; MAX_OUTPUT_CHANNELS],
            },
            sample: 0,
            real: false,
        }
    }

    fn next_frame(&mut self) -> bool {
        let epoch = self.clock.epoch.load(Ordering::Acquire);
        while let Some(frame) = self.clock.queue.pop() {
            if frame.epoch == epoch {
                self.frame = frame;
                self.real = true;
                self.sample = 0;
                return true;
            }
        }
        if self.clock.ended.load(Ordering::Acquire) {
            return false;
        }
        self.frame.samples.fill(0.0);
        self.real = false;
        self.sample = 0;
        true
    }
}

impl Iterator for RingSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.sample == 0 && !self.next_frame() {
            return None;
        }
        let sample = self.frame.samples[self.sample];
        self.sample += 1;
        if self.sample == self.clock.channels {
            self.sample = 0;
            if self.real {
                let next = self.clock.position.load(Ordering::Relaxed) + 1;
                let position = if self.clock.looped.load(Ordering::Acquire)
                    && next >= self.clock.loop_end.load(Ordering::Relaxed)
                {
                    self.clock.loop_start.load(Ordering::Relaxed)
                } else {
                    next
                };
                self.clock.position.store(position, Ordering::Relaxed);
            }
        }
        Some(sample)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl Source for RingSource {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> ChannelCount {
        ChannelCount::new(u16::try_from(self.clock.channels).expect("validated output channels"))
            .expect("output has channels")
    }

    fn sample_rate(&self) -> SampleRate {
        SampleRate::new(self.clock.rate).expect("output has a sample rate")
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }

    fn try_seek(&mut self, position: Duration) -> std::result::Result<(), SeekError> {
        self.clock.request_seek(frames(position, self.clock.rate));
        self.sample = 0;
        Ok(())
    }
}

pub struct PreparedAudio {
    path: PathBuf,
    position: Duration,
    decoder: Decoder<BufReader<File>>,
}

impl PreparedAudio {
    pub fn open(path: &Path, position: Duration) -> Result<Self> {
        let decoder = decoder(path, position)?;
        Ok(Self {
            path: path.to_owned(),
            position,
            decoder,
        })
    }
}

struct DecodeWorker {
    cancelled: Arc<AtomicBool>,
}

impl Drop for DecodeWorker {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

pub struct Audio {
    sink: MixerDeviceSink,
    player: Option<Player>,
    worker: Option<DecodeWorker>,
    path: Option<PathBuf>,
    clock: Option<Arc<PlaybackClock>>,
}

impl Audio {
    pub fn open_output(output: Option<&str>, buffer: Option<u32>) -> Result<Self> {
        if buffer.is_some_and(|frames| !COMMON_BUFFERS.contains(&frames)) {
            bail!("unsupported audio buffer size");
        }
        let mut sink = open_device(output_device(output)?, buffer)?;
        sink.log_on_drop(false);
        Ok(Self {
            sink,
            player: None,
            worker: None,
            path: None,
            clock: None,
        })
    }

    pub fn load(&mut self, prepared: PreparedAudio) -> Result<()> {
        self.clear();
        let config = self.sink.config();
        let channels = usize::from(config.channel_count().get());
        if channels > MAX_OUTPUT_CHANNELS {
            bail!("audio output has too many channels");
        }
        let rate = config.sample_rate().get();
        let buffer = match config.buffer_size() {
            BufferSize::Fixed(frames) => usize::try_from(*frames).unwrap_or(2_048),
            BufferSize::Default => 2_048,
        };
        let queue_frames = buffer.saturating_mul(4).clamp(512, 8_192);
        let clock = Arc::new(PlaybackClock::new(
            rate,
            channels,
            prepared.position,
            queue_frames,
        ));
        let source = RingSource::new(clock.clone());
        let player = Player::connect_new(self.sink.mixer());
        player.append(source);
        player.pause();
        let cancelled = Arc::new(AtomicBool::new(false));
        spawn_decode(
            prepared.path.clone(),
            prepared.decoder,
            prepared.position,
            clock.clone(),
            cancelled.clone(),
        )?;
        self.path = Some(prepared.path);
        self.clock = Some(clock);
        self.player = Some(player);
        self.worker = Some(DecodeWorker { cancelled });
        Ok(())
    }

    pub fn clear(&mut self) {
        self.worker = None;
        self.player = None;
        self.path = None;
        self.clock = None;
    }

    pub fn matches(&self, path: &Path) -> bool {
        self.path.as_deref() == Some(path)
    }

    pub fn play(&self) {
        if let Some(player) = &self.player {
            player.play();
        }
    }

    pub fn pause(&self) {
        if let Some(player) = &self.player {
            player.pause();
        }
    }

    pub fn paused(&self) -> bool {
        self.player.as_ref().is_none_or(Player::is_paused)
    }

    pub fn empty(&self) -> bool {
        self.player.as_ref().is_none_or(Player::empty)
    }

    pub fn position(&self) -> Duration {
        self.clock
            .as_ref()
            .map_or(Duration::ZERO, |clock| clock.position())
    }

    pub fn seek(&self, position: Duration) -> Result<()> {
        let clock = self.clock.as_ref().context("no audio is loaded")?;
        clock.request_seek(frames(position, clock.rate));
        Ok(())
    }

    pub fn set_loop(&self, range: Option<(Duration, Duration)>) {
        if let Some(clock) = &self.clock {
            clock.set_loop(range);
        }
    }
}

type UniformDecoder = UniformSourceIterator<Decoder<BufReader<File>>>;

fn decoder(path: &Path, position: Duration) -> Result<Decoder<BufReader<File>>> {
    let file = File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let mut decoder = Decoder::try_from(file).context("could not decode audio for playback")?;
    if !position.is_zero() {
        decoder
            .try_seek(position)
            .context("could not seek to the playback position")?;
    }
    Ok(decoder)
}

fn uniform(decoder: Decoder<BufReader<File>>, channels: usize, rate: u32) -> UniformDecoder {
    UniformSourceIterator::new(
        decoder,
        ChannelCount::new(u16::try_from(channels).expect("validated output channels"))
            .expect("output has channels"),
        SampleRate::new(rate).expect("output has a sample rate"),
    )
}

fn spawn_decode(
    path: PathBuf,
    decoder: Decoder<BufReader<File>>,
    position: Duration,
    clock: Arc<PlaybackClock>,
    cancelled: Arc<AtomicBool>,
) -> Result<()> {
    std::thread::Builder::new()
        .name("muspector-decode".to_owned())
        .spawn(move || decode_loop(&path, decoder, position, &clock, &cancelled))
        .context("could not start the audio decoder")?;
    Ok(())
}

fn decode_loop(
    path: &Path,
    initial_decoder: Decoder<BufReader<File>>,
    position: Duration,
    clock: &PlaybackClock,
    cancelled: &AtomicBool,
) {
    let mut source = uniform(initial_decoder, clock.channels, clock.rate);
    let mut source_frame = frames(position, clock.rate);
    let mut epoch = clock.epoch.load(Ordering::Acquire);

    while !cancelled.load(Ordering::Acquire) {
        let requested = clock.seek.swap(NO_SEEK, Ordering::AcqRel);
        if requested != NO_SEEK {
            let Ok(next) = decoder(path, duration(requested, clock.rate)) else {
                clock.ended.store(true, Ordering::Release);
                break;
            };
            source = uniform(next, clock.channels, clock.rate);
            source_frame = requested;
            epoch = clock.epoch.load(Ordering::Acquire);
        }

        if clock.looped.load(Ordering::Acquire)
            && source_frame >= clock.loop_end.load(Ordering::Relaxed)
        {
            let start = clock.loop_start.load(Ordering::Relaxed);
            let Ok(next) = decoder(path, duration(start, clock.rate)) else {
                clock.ended.store(true, Ordering::Release);
                break;
            };
            source = uniform(next, clock.channels, clock.rate);
            source_frame = start;
        }

        let mut queued = QueuedFrame {
            epoch,
            samples: [0.0; MAX_OUTPUT_CHANNELS],
        };
        let mut complete = true;
        for sample in &mut queued.samples[..clock.channels] {
            let Some(value) = source.next() else {
                complete = false;
                break;
            };
            *sample = value;
        }
        if !complete {
            clock.ended.store(true, Ordering::Release);
            break;
        }

        loop {
            match clock.queue.push(queued) {
                Ok(()) => break,
                Err(returned) => queued = returned,
            }
            if cancelled.load(Ordering::Acquire) || clock.seek.load(Ordering::Acquire) != NO_SEEK {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        source_frame += 1;
    }
}

fn output_device(output: Option<&str>) -> Result<rodio::cpal::Device> {
    let Some(output) = output else {
        return rodio::cpal::default_host()
            .default_output_device()
            .context("no audio output is available");
    };
    let id = DeviceId::from_str(output).context("saved audio device is invalid")?;
    let host = rodio::cpal::host_from_id(id.0).context("selected audio backend is unavailable")?;
    host.device_by_id(&id)
        .context("selected audio output is unavailable")
}

/// Device timing and a supported power-of-two buffer targeting roughly 10 ms.
pub fn output_timing(output: Option<&str>) -> Option<OutputTiming> {
    output_timing_for_device(&output_device(output).ok()?)
}

fn output_timing_for_device(device: &rodio::cpal::Device) -> Option<OutputTiming> {
    let config = device.default_output_config().ok()?;
    let target = nearest_power_of_two(config.sample_rate() / 100);
    let automatic_buffer = match config.buffer_size() {
        SupportedBufferSize::Range { min, max } => target.clamp(*min, *max),
        SupportedBufferSize::Unknown => target,
    };
    Some(OutputTiming {
        sample_rate: config.sample_rate(),
        automatic_buffer,
    })
}

fn nearest_power_of_two(value: u32) -> u32 {
    if value <= 1 {
        return 1;
    }
    let next = value.next_power_of_two();
    let previous = next >> 1;
    if value - previous <= next - value {
        previous
    } else {
        next
    }
}

fn open_device(device: rodio::cpal::Device, buffer: Option<u32>) -> Result<MixerDeviceSink> {
    let builder = DeviceSinkBuilder::from_device(device.clone())
        .context("audio output has no compatible format")?;
    let buffer =
        buffer.or_else(|| output_timing_for_device(&device).map(|timing| timing.automatic_buffer));
    let builder = match buffer {
        Some(frames) => builder.with_buffer_size(BufferSize::Fixed(frames)),
        None => builder,
    };
    builder
        .open_stream()
        .context("could not open the audio output with this buffer size")
}

fn frames(duration: Duration, rate: u32) -> u64 {
    (duration.as_secs_f64() * f64::from(rate)).round() as u64
}

fn duration(frames: u64, rate: u32) -> Duration {
    Duration::from_secs_f64(frames as f64 / f64::from(rate))
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
        left.backend
            .cmp(&right.backend)
            .then_with(|| right.default.cmp(&left.default))
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
    load("audio-output")
}

pub fn save_output(output: Option<&str>) -> Result<()> {
    save("audio-output", output.unwrap_or_default())
}

pub fn load_buffer() -> Option<u32> {
    load("audio-buffer")
        .and_then(|value| value.parse().ok())
        .filter(|frames| COMMON_BUFFERS.contains(frames))
}

pub fn save_buffer(buffer: Option<u32>) -> Result<()> {
    save(
        "audio-buffer",
        &buffer.map_or_else(String::new, |frames| frames.to_string()),
    )
}

fn load(name: &str) -> Option<String> {
    fs::read_to_string(settings_path(name)?)
        .ok()
        .and_then(|value| {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_owned())
        })
}

fn save(name: &str, value: &str) -> Result<()> {
    let path = settings_path(name).context("could not locate the audio settings directory")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("could not create the audio settings directory")?;
    }
    fs::write(path, value).context("could not save the audio output setting")
}

fn settings_path(name: &str) -> Option<PathBuf> {
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
    root.map(|path| path.join(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn mono_wav(samples: &[i16], rate: u32) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().expect("temporary wav");
        let data_bytes = u32::try_from(std::mem::size_of_val(samples)).expect("small fixture");
        file.write_all(b"RIFF").expect("riff");
        file.write_all(&(36 + data_bytes).to_le_bytes())
            .expect("riff length");
        file.write_all(b"WAVEfmt ").expect("wave fmt");
        file.write_all(&16_u32.to_le_bytes()).expect("fmt length");
        file.write_all(&1_u16.to_le_bytes()).expect("pcm");
        file.write_all(&1_u16.to_le_bytes()).expect("mono");
        file.write_all(&rate.to_le_bytes()).expect("sample rate");
        file.write_all(&(rate * 2).to_le_bytes())
            .expect("byte rate");
        file.write_all(&2_u16.to_le_bytes()).expect("block align");
        file.write_all(&16_u16.to_le_bytes()).expect("bits");
        file.write_all(b"data").expect("data");
        file.write_all(&data_bytes.to_le_bytes())
            .expect("data length");
        for sample in samples {
            file.write_all(&sample.to_le_bytes()).expect("sample");
        }
        file.flush().expect("flush wav");
        file
    }

    #[test]
    fn loop_wraps_on_an_exact_frame_boundary() {
        let fixture = mono_wav(&[0, 1_000, 2_000, 3_000], 1_000);
        let prepared = PreparedAudio::open(fixture.path(), Duration::ZERO).expect("open wav");
        let clock = Arc::new(PlaybackClock::new(1_000, 1, Duration::ZERO, 8));
        clock.set_loop(Some((Duration::from_millis(1), Duration::from_millis(3))));
        let cancelled = Arc::new(AtomicBool::new(false));
        spawn_decode(
            prepared.path,
            prepared.decoder,
            prepared.position,
            clock.clone(),
            cancelled.clone(),
        )
        .expect("spawn decoder");
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while clock.queue.len() < 4 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            clock.queue.len() >= 4,
            "decoder did not prepare loop frames"
        );
        let mut source = RingSource::new(clock.clone());

        let positions = (0..4)
            .map(|_| {
                source.next().expect("looped sample");
                frames(clock.position(), 1_000)
            })
            .collect::<Vec<_>>();
        cancelled.store(true, Ordering::Release);

        assert_eq!(positions, [2, 1, 2, 1]);
    }

    #[test]
    fn invalid_loop_range_is_disabled() {
        let clock = PlaybackClock::new(48_000, 2, Duration::ZERO, 512);
        clock.set_loop(Some((Duration::from_millis(20), Duration::from_millis(20))));
        assert!(!clock.looped.load(Ordering::Acquire));
    }

    #[test]
    fn automatic_buffer_targets_about_ten_milliseconds() {
        assert_eq!(nearest_power_of_two(441), 512);
        assert_eq!(nearest_power_of_two(480), 512);
        assert_eq!(nearest_power_of_two(960), 1_024);
    }
}
