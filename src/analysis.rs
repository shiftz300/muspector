use crate::{
    blind,
    chain::{self, Chain, Fingerprint},
};
use anyhow::{Context, Result, bail};
use ebur128::{EbuR128, Mode};
use realfft::{RealFftPlanner, RealToComplex};
use std::{
    collections::VecDeque,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{
        CODEC_TYPE_AAC, CODEC_TYPE_ALAC, CODEC_TYPE_FLAC, CODEC_TYPE_MP3, CODEC_TYPE_PCM_F32BE,
        CODEC_TYPE_PCM_F32LE, CODEC_TYPE_PCM_F64BE, CODEC_TYPE_PCM_F64LE, CODEC_TYPE_PCM_S8,
        CODEC_TYPE_PCM_S16BE, CODEC_TYPE_PCM_S16LE, CODEC_TYPE_PCM_S24BE, CODEC_TYPE_PCM_S24LE,
        CODEC_TYPE_PCM_S32BE, CODEC_TYPE_PCM_S32LE, CODEC_TYPE_PCM_U8, CODEC_TYPE_PCM_U16BE,
        CODEC_TYPE_PCM_U16LE, CODEC_TYPE_PCM_U24BE, CODEC_TYPE_PCM_U24LE, CODEC_TYPE_PCM_U32BE,
        CODEC_TYPE_PCM_U32LE, CODEC_TYPE_VORBIS, CodecType, DecoderOptions,
    },
    errors::Error,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

const FFT: usize = 4096;
const HOP: usize = 2048;
const FLOOR: f64 = 1.0e-12;
const CHART: usize = 64;
const LIMIT: usize = 32_768;

#[derive(Clone, Debug)]
pub struct Point {
    pub min: f32,
    pub max: f32,
    pub level: f64,
    pub loudness: f64,
}

#[derive(Clone, Debug)]
pub struct Profile {
    pub points: Vec<Point>,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub path: PathBuf,
    pub codec: String,
    pub rate: u32,
    pub channels: usize,
    pub duration: f64,
    pub peak: f64,
    pub rms: f64,
    pub loudness: f64,
    pub crest: f64,
    pub centroid: f64,
    pub rolloff: f64,
    pub low: f64,
    pub mid: f64,
    pub high: f64,
    pub clips: u64,
    pub spectrum: Vec<f64>,
    pub profile: Profile,
    pub chain: Chain,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progress {
    pub value: f32,
    pub stage: &'static str,
}

impl Report {
    pub fn name(&self) -> String {
        self.path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Audio")
            .to_owned()
    }

    pub fn duration_text(&self) -> String {
        let total = self.duration.round() as u64;
        format!("{}:{:02}", total / 60, total % 60)
    }

    pub fn format_text(&self) -> String {
        format!(
            "{}  ·  {:.1} kHz  ·  {}",
            self.codec,
            self.rate as f64 / 1000.0,
            channel_text(self.channels)
        )
    }
}

#[cfg(test)]
pub fn inspect(path: &Path) -> Result<Report> {
    inspect_with_training(path, blind::Training::embedded())
}

pub fn inspect_with_training(path: &Path, training: blind::Training) -> Result<Report> {
    inspect_span(path, None, training, |_| {})
}

pub fn inspect_with_progress(
    path: &Path,
    training: blind::Training,
    progress: impl FnMut(Progress),
) -> Result<Report> {
    inspect_span(path, None, training, progress)
}

#[cfg(test)]
pub fn inspect_range(path: &Path, start: f64, end: f64) -> Result<Report> {
    inspect_range_with_training(path, start, end, blind::Training::embedded())
}

pub fn inspect_range_with_training(
    path: &Path,
    start: f64,
    end: f64,
    training: blind::Training,
) -> Result<Report> {
    if !start.is_finite() || !end.is_finite() || start < 0.0 || end <= start {
        bail!("Invalid analysis range");
    }
    inspect_span(path, Some((start, end)), training, |_| {})
}

fn inspect_span(
    path: &Path,
    span: Option<(f64, f64)>,
    training: blind::Training,
    mut progress: impl FnMut(Progress),
) -> Result<Report> {
    progress(Progress {
        value: 0.02,
        stage: "Opening audio",
    });
    if !path.is_file() {
        bail!("请选择一个音频文件");
    }

    let source = File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(source), Default::default());
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
        .context("不支持或无法识别这个音频文件")?;
    let mut format = probed.format;
    let track = format.default_track().context("文件中没有可解码的音轨")?;
    let track_id = track.id;
    let params = &track.codec_params;
    let rate = params.sample_rate.context("音频缺少采样率信息")?;
    let channels = params
        .channels
        .map(|channels| channels.count())
        .context("音频缺少声道信息")?;
    let codec = codec_name(params.codec, path);
    let total_frames = span
        .map(|(start, end)| ((end - start) * f64::from(rate)).ceil() as u64)
        .or(params.n_frames)
        .filter(|frames| *frames > 0);
    let mut decoder = symphonia::default::get_codecs()
        .make(params, &DecoderOptions::default())
        .context("没有适合这个音轨的解码器")?;

    let mut signal = Signal::new(rate);
    let mut timeline = Timeline::new(rate);
    let mut space = Space::new(rate);
    let mut sample_buffer = None;
    let mut frames = 0_u64;
    let mut count = 0_u64;
    let mut sum = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut clips = 0_u64;
    let mut scan = blind::Scan::with_training(rate, training);
    let mut loudness =
        EbuR128::new(channels as u32, rate, Mode::I).context("无法初始化响度分析器")?;
    let from = span
        .map(|(start, _)| (start * f64::from(rate)).floor() as u64)
        .unwrap_or(0);
    let to = span
        .map(|(_, end)| (end * f64::from(rate)).ceil() as u64)
        .unwrap_or(u64::MAX);
    let mut decoded_frames = 0_u64;
    let mut reported = 0.08_f32;
    progress(Progress {
        value: reported,
        stage: "Decoding audio",
    });

    'packets: loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(Error::ResetRequired) => bail!("音轨在文件中途发生变化，暂不支持"),
            Err(error) => return Err(error).context("读取音频数据失败"),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(Error::DecodeError(_)) => continue,
            Err(error) => return Err(error).context("解码音频失败"),
        };
        let spec = *decoded.spec();
        let capacity = decoded.capacity() as u64;
        let buffer = sample_buffer.get_or_insert_with(|| SampleBuffer::<f32>::new(capacity, spec));
        buffer.copy_interleaved_ref(decoded);
        let samples = buffer.samples();
        let packet_frames = samples.len() / channels;
        let packet_start = decoded_frames;
        let packet_end = packet_start.saturating_add(packet_frames as u64);
        decoded_frames = packet_end;
        if let Some(total) = total_frames {
            let selected_frames = packet_end.saturating_sub(from).min(total);
            let value = 0.08 + 0.62 * (selected_frames as f32 / total as f32).clamp(0.0, 1.0);
            if value - reported >= 0.005 {
                reported = value;
                progress(Progress {
                    value,
                    stage: "Decoding audio",
                });
            }
        }
        if packet_end <= from {
            continue;
        }
        if packet_start >= to {
            break;
        }
        let local_start = from.saturating_sub(packet_start).min(packet_frames as u64) as usize;
        let local_end = to
            .min(packet_end)
            .saturating_sub(packet_start)
            .min(packet_frames as u64) as usize;
        let selected = &samples[local_start * channels..local_end * channels];
        loudness.add_frames_f32(selected).context("响度分析失败")?;
        let momentary = loudness
            .loudness_momentary()
            .unwrap_or(-72.0)
            .clamp(-72.0, 0.0);
        for frame in selected.chunks_exact(channels) {
            frames += 1;
            let mut mono = 0.0_f32;
            for &sample in frame {
                let value = f64::from(sample);
                let absolute = value.abs();
                peak = peak.max(absolute);
                sum += value * value;
                count += 1;
                clips += u64::from(absolute >= 0.999_9);
                mono += sample / channels as f32;
            }
            timeline.push(mono, momentary);
            space.push(mono);
            signal.push(mono)?;
            scan.push(mono);
        }
        if packet_end >= to {
            break 'packets;
        }
    }

    if count == 0 || frames == 0 {
        bail!("音轨没有可分析的采样");
    }

    progress(Progress {
        value: 0.72,
        stage: "Measuring loudness",
    });
    let rms = (sum / count as f64).sqrt();
    let loudness = loudness
        .loudness_global()
        .unwrap_or(-72.0)
        .clamp(-72.0, 0.0);
    progress(Progress {
        value: 0.78,
        stage: "Building spectrum",
    });
    let spectrum = signal.finish()?;
    let profile = timeline.finish();
    let space = space.finish();
    progress(Progress {
        value: 0.84,
        stage: "Inferring signal chain",
    });
    let mut chain = chain::infer(fingerprint(
        &profile,
        db(peak),
        db(peak) - db(rms),
        &spectrum,
        space,
    ));
    progress(Progress {
        value: 0.9,
        stage: "Running blind models",
    });
    let model = scan.finish().context("Inspector Routed analysis failed")?;
    model.apply(&mut chain);
    progress(Progress {
        value: 1.0,
        stage: "Finalizing",
    });
    Ok(Report {
        path: path.to_path_buf(),
        codec,
        rate,
        channels,
        duration: frames as f64 / rate as f64,
        peak: db(peak),
        rms: db(rms),
        loudness,
        crest: db(peak) - db(rms),
        centroid: spectrum.centroid,
        rolloff: spectrum.rolloff,
        low: spectrum.low,
        mid: spectrum.mid,
        high: spectrum.high,
        clips,
        spectrum: spectrum.curve,
        profile,
        chain,
    })
}

pub fn training_from_clean(path: &Path) -> Result<blind::Training> {
    if !path.is_file() {
        bail!("请选择一个 clean 音频文件");
    }
    let source = File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(source), Default::default());
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
        .context("不支持或无法识别这个 clean 音频文件")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("clean 文件中没有可解码的音轨")?;
    let track_id = track.id;
    let params = &track.codec_params;
    let rate = params.sample_rate.context("clean 音频缺少采样率信息")?;
    let channels = params
        .channels
        .map(|channels| channels.count())
        .context("clean 音频缺少声道信息")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(params, &DecoderOptions::default())
        .context("没有适合 clean 音轨的解码器")?;
    let mut sample_buffer = None;
    let maximum_frames = rate as usize * 10;
    let mut mono = Vec::with_capacity(maximum_frames);
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(Error::ResetRequired) => bail!("clean 音轨在文件中途发生变化，暂不支持"),
            Err(error) => return Err(error).context("读取 clean 音频失败"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(Error::DecodeError(_)) => continue,
            Err(error) => return Err(error).context("解码 clean 音频失败"),
        };
        let spec = *decoded.spec();
        let capacity = decoded.capacity() as u64;
        let buffer = sample_buffer.get_or_insert_with(|| SampleBuffer::<f32>::new(capacity, spec));
        buffer.copy_interleaved_ref(decoded);
        let remaining = maximum_frames.saturating_sub(mono.len());
        mono.extend(
            buffer
                .samples()
                .chunks_exact(channels)
                .take(remaining)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32),
        );
        if mono.len() == maximum_frames {
            break;
        }
    }
    let name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Clean")
        .to_owned();
    blind::Training::from_clean(&mono, rate, name)
}

#[derive(Clone, Copy)]
struct Meter {
    min: f32,
    max: f32,
    square: f64,
    loudness: f64,
    count: u64,
}

impl Meter {
    fn empty() -> Self {
        Self {
            min: 1.0,
            max: -1.0,
            square: 0.0,
            loudness: 0.0,
            count: 0,
        }
    }

    fn push(&mut self, sample: f32, loudness: f64) {
        self.min = self.min.min(sample);
        self.max = self.max.max(sample);
        self.square += f64::from(sample) * f64::from(sample);
        self.loudness += 10.0_f64.powf(loudness / 10.0);
        self.count += 1;
    }

    fn merge(&mut self, other: Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other;
            return;
        }
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.square += other.square;
        self.loudness += other.loudness;
        self.count += other.count;
    }

    fn point(self) -> Point {
        if self.count == 0 {
            return Point {
                min: 0.0,
                max: 0.0,
                level: -72.0,
                loudness: -72.0,
            };
        }
        let rms = (self.square / self.count as f64).sqrt();
        Point {
            min: self.min,
            max: self.max,
            level: db(rms).clamp(-72.0, 0.0),
            loudness: 10.0 * (self.loudness / self.count as f64).max(FLOOR).log10(),
        }
    }
}

struct Timeline {
    span: u64,
    meter: Meter,
    bins: Vec<Meter>,
}

#[derive(Clone, Copy)]
struct Spaceprint {
    echo: f64,
    delay: f64,
    tail: f64,
}

struct Space {
    span: u64,
    count: u64,
    square: f64,
    mean: f64,
    ring: VecDeque<f64>,
    corr: [f64; 51],
    left: [f64; 51],
    right: [f64; 51],
}

impl Space {
    fn new(rate: u32) -> Self {
        Self {
            span: (rate as u64 / 50).max(1),
            count: 0,
            square: 0.0,
            mean: 0.0,
            ring: VecDeque::with_capacity(51),
            corr: [0.0; 51],
            left: [0.0; 51],
            right: [0.0; 51],
        }
    }

    fn push(&mut self, sample: f32) {
        self.square += f64::from(sample) * f64::from(sample);
        self.count += 1;
        if self.count >= self.span {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.count == 0 {
            return;
        }
        let envelope = (self.square / self.count as f64).sqrt();
        self.mean = self.mean * 0.98 + envelope * 0.02;
        let value = envelope - self.mean;
        for lag in 3..=50 {
            if self.ring.len() < lag {
                continue;
            }
            let previous = self.ring[self.ring.len() - lag];
            self.corr[lag] += value * previous;
            self.left[lag] += value * value;
            self.right[lag] += previous * previous;
        }
        self.ring.push_back(value);
        if self.ring.len() > 50 {
            self.ring.pop_front();
        }
        self.count = 0;
        self.square = 0.0;
    }

    fn finish(mut self) -> Spaceprint {
        self.flush();
        let score = |lag: usize| {
            let norm = (self.left[lag] * self.right[lag]).sqrt();
            if norm <= FLOOR {
                0.0
            } else {
                (self.corr[lag] / norm).max(0.0)
            }
        };
        let (lag, echo) = (3..=50)
            .map(|lag| (lag, score(lag)))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((3, 0.0));
        let tail = (3..=25).map(score).sum::<f64>() / 23.0;
        Spaceprint {
            echo,
            delay: lag as f64 * 20.0,
            tail,
        }
    }
}

impl Timeline {
    fn new(rate: u32) -> Self {
        Self {
            span: (rate as u64 / 40).max(1),
            meter: Meter::empty(),
            bins: Vec::with_capacity(LIMIT),
        }
    }

    fn push(&mut self, sample: f32, loudness: f64) {
        self.meter.push(sample, loudness);
        if self.meter.count >= self.span {
            self.bins.push(self.meter);
            self.meter = Meter::empty();
            if self.bins.len() >= LIMIT {
                self.shrink();
            }
        }
    }

    fn shrink(&mut self) {
        let mut bins = Vec::with_capacity(self.bins.len().div_ceil(2));
        for pair in self.bins.chunks(2) {
            let mut meter = Meter::empty();
            for item in pair {
                meter.merge(*item);
            }
            bins.push(meter);
        }
        self.bins = bins;
        self.span = self.span.saturating_mul(2);
    }

    fn finish(mut self) -> Profile {
        if self.meter.count > 0 {
            self.bins.push(self.meter);
        }
        if self.bins.is_empty() {
            return Profile { points: Vec::new() };
        }

        let points = self.bins.into_iter().map(Meter::point).collect();
        Profile { points }
    }
}

struct Signal {
    rate: u32,
    fft: Arc<dyn RealToComplex<f32>>,
    input: Vec<f32>,
    window: Vec<f32>,
    bins: Vec<f64>,
    frames: u64,
}

impl Signal {
    fn new(rate: u32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT);
        let window = (0..FFT)
            .map(|index| {
                let phase = std::f32::consts::TAU * index as f32 / (FFT - 1) as f32;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        Self {
            rate,
            fft,
            input: Vec::with_capacity(FFT * 2),
            window,
            bins: vec![0.0; FFT / 2 + 1],
            frames: 0,
        }
    }

    fn push(&mut self, sample: f32) -> Result<()> {
        self.input.push(sample);
        if self.input.len() >= FFT {
            self.frame()?;
            self.input.drain(..HOP);
        }
        Ok(())
    }

    fn frame(&mut self) -> Result<()> {
        let mut input: Vec<f32> = self.input[..FFT]
            .iter()
            .zip(&self.window)
            .map(|(sample, window)| sample * window)
            .collect();
        let mut output = self.fft.make_output_vec();
        self.fft
            .process(&mut input, &mut output)
            .context("频谱计算失败")?;
        for (sum, value) in self.bins.iter_mut().zip(output) {
            *sum += f64::from(value.norm_sqr());
        }
        self.frames += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<Spectrum> {
        if self.frames == 0 {
            self.input.resize(FFT, 0.0);
            self.frame()?;
        }
        Spectrum::from_bins(&self.bins, self.rate)
    }
}

struct Spectrum {
    centroid: f64,
    rolloff: f64,
    flatness: f64,
    low: f64,
    mid: f64,
    high: f64,
    curve: Vec<f64>,
}

impl Spectrum {
    fn from_bins(bins: &[f64], rate: u32) -> Result<Self> {
        let bin_hz = rate as f64 / FFT as f64;
        let total: f64 = bins.iter().sum();
        if total <= FLOOR {
            return Ok(Self {
                centroid: 0.0,
                rolloff: 0.0,
                flatness: 0.0,
                low: -120.0,
                mid: -120.0,
                high: -120.0,
                curve: vec![-72.0; CHART],
            });
        }

        let centroid = bins
            .iter()
            .enumerate()
            .map(|(index, power)| index as f64 * bin_hz * power)
            .sum::<f64>()
            / total;
        let target = total * 0.85;
        let mut running = 0.0;
        let mut rolloff = 0.0;
        for (index, power) in bins.iter().enumerate() {
            running += power;
            if running >= target {
                rolloff = index as f64 * bin_hz;
                break;
            }
        }

        let band = |from: f64, to: f64| {
            let power = bins
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    let hz = *index as f64 * bin_hz;
                    hz >= from && hz < to
                })
                .map(|(_, power)| power)
                .sum::<f64>();
            10.0 * (power / total).max(FLOOR).log10()
        };

        let audible: Vec<_> = bins
            .iter()
            .enumerate()
            .skip(1)
            .take_while(|(index, _)| *index as f64 * bin_hz <= 20_000.0)
            .map(|(_, power)| *power)
            .collect();
        let arithmetic = audible.iter().sum::<f64>() / audible.len().max(1) as f64;
        let geometric = (audible
            .iter()
            .map(|power| power.max(FLOOR).ln())
            .sum::<f64>()
            / audible.len().max(1) as f64)
            .exp();
        let flatness = if arithmetic <= FLOOR {
            0.0
        } else {
            (geometric / arithmetic).clamp(0.0, 1.0)
        };

        Ok(Self {
            centroid,
            rolloff,
            flatness,
            low: band(20.0, 250.0),
            mid: band(250.0, 4_000.0),
            high: band(4_000.0, 20_000.0),
            curve: chart(bins, rate),
        })
    }
}

fn fingerprint(
    profile: &Profile,
    peak: f64,
    crest: f64,
    spectrum: &Spectrum,
    space: Spaceprint,
) -> Fingerprint {
    let mut levels: Vec<_> = profile.points.iter().map(|point| point.level).collect();
    levels.sort_by(f64::total_cmp);
    let floor = quantile(&levels, 0.1);
    let range = quantile(&levels, 0.9) - floor;
    let silence = if levels.is_empty() {
        0.0
    } else {
        levels.iter().filter(|level| **level <= -48.0).count() as f64 / levels.len() as f64
    };
    let transient = if profile.points.len() < 2 {
        0.0
    } else {
        let change = profile
            .points
            .windows(2)
            .map(|pair| (pair[1].level - pair[0].level).abs())
            .sum::<f64>()
            / (profile.points.len() - 1) as f64;
        (change / 8.0).clamp(0.0, 1.0)
    };
    Fingerprint {
        peak,
        crest,
        range,
        floor,
        silence,
        transient,
        flatness: spectrum.flatness,
        low: spectrum.low,
        mid: spectrum.mid,
        high: spectrum.high,
        echo: space.echo,
        echo_ms: space.delay,
        tail: space.tail,
    }
}

fn quantile(values: &[f64], amount: f64) -> f64 {
    if values.is_empty() {
        return -72.0;
    }
    let index = (amount.clamp(0.0, 1.0) * values.len().saturating_sub(1) as f64).round() as usize;
    values[index]
}

fn chart(bins: &[f64], rate: u32) -> Vec<f64> {
    let low = 20.0_f64;
    let high = (rate as f64 / 2.0).clamp(40.0, 20_000.0);
    let span = (high / low).ln();
    let hz = rate as f64 / FFT as f64;
    let mut power = vec![0.0_f64; CHART];

    for (index, value) in bins.iter().enumerate().skip(1) {
        let frequency = index as f64 * hz;
        if frequency < low || frequency > high {
            continue;
        }
        let position = ((frequency / low).ln() / span).clamp(0.0, 1.0);
        let band = ((position * CHART as f64) as usize).min(CHART - 1);
        power[band] += value;
    }

    let smooth: Vec<f64> = (0..CHART)
        .map(|index| {
            let before = power[index.saturating_sub(1)];
            let current = power[index];
            let after = power[(index + 1).min(CHART - 1)];
            (before + current * 2.0 + after) / 4.0
        })
        .collect();
    let peak = smooth.iter().copied().fold(0.0_f64, f64::max);
    if peak <= FLOOR {
        return vec![-72.0; CHART];
    }

    smooth
        .into_iter()
        .map(|value| (10.0 * (value / peak).max(FLOOR).log10()).clamp(-72.0, 0.0))
        .collect()
}

fn db(value: f64) -> f64 {
    20.0 * value.max(FLOOR).log10()
}

fn channel_text(channels: usize) -> &'static str {
    match channels {
        1 => "Mono",
        2 => "Stereo",
        _ => "Multi",
    }
}

fn codec_name(codec: CodecType, path: &Path) -> String {
    match codec {
        CODEC_TYPE_MP3 => "MP3".to_owned(),
        CODEC_TYPE_AAC => "AAC".to_owned(),
        CODEC_TYPE_ALAC => "ALAC".to_owned(),
        CODEC_TYPE_FLAC => "FLAC".to_owned(),
        CODEC_TYPE_VORBIS => "Vorbis".to_owned(),
        CODEC_TYPE_PCM_S32LE | CODEC_TYPE_PCM_S32BE | CODEC_TYPE_PCM_S24LE
        | CODEC_TYPE_PCM_S24BE | CODEC_TYPE_PCM_S16LE | CODEC_TYPE_PCM_S16BE
        | CODEC_TYPE_PCM_S8 | CODEC_TYPE_PCM_U32LE | CODEC_TYPE_PCM_U32BE
        | CODEC_TYPE_PCM_U24LE | CODEC_TYPE_PCM_U24BE | CODEC_TYPE_PCM_U16LE
        | CODEC_TYPE_PCM_U16BE | CODEC_TYPE_PCM_U8 | CODEC_TYPE_PCM_F32LE
        | CODEC_TYPE_PCM_F32BE | CODEC_TYPE_PCM_F64LE | CODEC_TYPE_PCM_F64BE => "PCM".to_owned(),
        _ => path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("Audio")
            .to_ascii_uppercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::Kind;
    use std::io::Write;

    #[test]
    fn silence_has_a_finite_floor() {
        assert_eq!(db(0.0), -240.0);
    }

    #[test]
    fn empty_spectrum_is_safe() {
        let spectrum = Spectrum::from_bins(&vec![0.0; FFT / 2 + 1], 48_000).unwrap();
        assert_eq!(spectrum.centroid, 0.0);
        assert_eq!(spectrum.low, -120.0);
    }

    #[test]
    fn profile_is_bounded() {
        let mut timeline = Timeline::new(48_000);
        for index in 0..(48_000 * 240) {
            let phase = std::f32::consts::TAU * 220.0 * index as f32 / 48_000.0;
            timeline.push(phase.sin() * 0.5, -18.0);
        }
        let profile = timeline.finish();
        assert!(!profile.points.is_empty());
        assert!(profile.points.len() <= LIMIT);
        assert!(profile.points.len() > 192);
        assert!(profile.points.iter().all(|point| point.level.is_finite()));
        assert!(
            profile
                .points
                .iter()
                .all(|point| point.loudness.is_finite())
        );
    }

    #[test]
    fn inspects_pcm_wav() {
        let path = std::env::temp_dir().join(format!("muspector-{}.wav", std::process::id()));
        let rate = 48_000_u32;
        let frames = 4_800_u32;
        let data_len = frames * 2;
        let mut file = File::create(&path).unwrap();
        file.write_all(b"RIFF").unwrap();
        file.write_all(&(36 + data_len).to_le_bytes()).unwrap();
        file.write_all(b"WAVEfmt ").unwrap();
        file.write_all(&16_u32.to_le_bytes()).unwrap();
        file.write_all(&1_u16.to_le_bytes()).unwrap();
        file.write_all(&1_u16.to_le_bytes()).unwrap();
        file.write_all(&rate.to_le_bytes()).unwrap();
        file.write_all(&(rate * 2).to_le_bytes()).unwrap();
        file.write_all(&2_u16.to_le_bytes()).unwrap();
        file.write_all(&16_u16.to_le_bytes()).unwrap();
        file.write_all(b"data").unwrap();
        file.write_all(&data_len.to_le_bytes()).unwrap();
        for index in 0..frames {
            let phase = std::f32::consts::TAU * 440.0 * index as f32 / rate as f32;
            let sample = (phase.sin() * i16::MAX as f32 * 0.5) as i16;
            file.write_all(&sample.to_le_bytes()).unwrap();
        }
        drop(file);

        let mut progress = Vec::new();
        let report = inspect_with_progress(&path, blind::Training::embedded(), |update| {
            progress.push(update);
        })
        .unwrap();
        let range = inspect_range(&path, 0.02, 0.05).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(report.codec, "PCM");
        assert_eq!(report.rate, rate);
        assert_eq!(report.channels, 1);
        assert!((report.duration - 0.1).abs() < 0.001);
        assert!((report.centroid - 440.0).abs() < 30.0);
        assert_eq!(report.spectrum.len(), CHART);
        assert!(report.spectrum.iter().all(|value| value.is_finite()));
        assert!(!report.profile.points.is_empty());
        assert!(report.profile.points.len() <= LIMIT);
        assert!(
            progress
                .windows(2)
                .all(|updates| updates[0].value <= updates[1].value)
        );
        assert_eq!(progress.last().map(|update| update.value), Some(1.0));
        assert!(
            progress
                .iter()
                .any(|update| update.stage == "Running blind models")
        );
        assert!((range.duration - 0.03).abs() < 0.001);
        assert!((range.centroid - 440.0).abs() < 35.0);
    }

    #[test]
    #[ignore = "requires MUSPECTOR_DEVICE_FIXTURE with the private labelled WAV directory"]
    fn inspector_device_fixture_matches_labels() {
        let root = PathBuf::from(
            std::env::var("MUSPECTOR_DEVICE_FIXTURE")
                .expect("set MUSPECTOR_DEVICE_FIXTURE to the labelled WAV directory"),
        );
        let mut paths = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|value| value == "wav"))
            .collect::<Vec<_>>();
        paths.sort();
        assert!(!paths.is_empty());
        let mut true_positive = [0_usize; 3];
        let mut false_positive = [0_usize; 3];
        let mut false_negative = [0_usize; 3];
        let mut exact = 0_usize;
        for path in paths {
            let report = inspect(&path).unwrap();
            let name = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap()
                .to_ascii_lowercase();
            let expected = [
                name.contains("drive")
                    || name.contains("fuzz")
                    || name.contains("rat")
                    || name.contains("muff"),
                name.contains("delay") || name.contains("echo"),
                name.contains("ambience")
                    || name.contains("dream")
                    || name.contains("reverb")
                    || name.contains("room")
                    || name.contains("hall")
                    || name.contains("plate"),
            ];
            let actual = [Kind::Drive, Kind::Delay, Kind::Reverb].map(|kind| {
                report
                    .chain
                    .effects
                    .iter()
                    .find(|effect| effect.kind == kind)
                    .is_some_and(|effect| effect.active)
            });
            let scores = [Kind::Drive, Kind::Delay, Kind::Reverb].map(|kind| {
                report
                    .chain
                    .effects
                    .iter()
                    .find(|effect| effect.kind == kind)
                    .map_or(0.0, |effect| effect.score)
            });
            println!(
                "{} expected {expected:?} actual {actual:?} scores {scores:?}",
                path.display()
            );
            exact += usize::from(actual == expected);
            for index in 0..3 {
                true_positive[index] += usize::from(expected[index] && actual[index]);
                false_positive[index] += usize::from(!expected[index] && actual[index]);
                false_negative[index] += usize::from(expected[index] && !actual[index]);
            }
            if name == "clean" {
                assert_eq!(actual, [false; 3], "{}", path.display());
            }
        }
        for index in 0..3 {
            let recall =
                true_positive[index] as f64 / (true_positive[index] + false_negative[index]) as f64;
            let precision =
                true_positive[index] as f64 / (true_positive[index] + false_positive[index]) as f64;
            assert!(recall >= 0.80, "class {index} recall {recall}");
            assert!(precision >= 0.50, "class {index} precision {precision}");
        }
        assert!(exact >= 9, "external exact matches {exact}/15");
    }

    #[test]
    #[ignore = "requires MUSPECTOR_DEVICE_FIXTURE with clean.wav"]
    fn inspector_clean_import_builds_portable_profile() {
        let root = PathBuf::from(
            std::env::var("MUSPECTOR_DEVICE_FIXTURE")
                .expect("set MUSPECTOR_DEVICE_FIXTURE to the labelled WAV directory"),
        );
        let clean = root.join("clean.wav");
        let training = training_from_clean(&clean).unwrap();
        assert!(!training.calibrated());
        let restored = blind::Training::import(&training.export()).unwrap();
        assert_eq!(restored.name(), "clean");
        assert!(!restored.calibrated());
        let report = inspect_with_training(&clean, restored.clone()).unwrap();
        let clean_actual = [Kind::Drive, Kind::Delay, Kind::Reverb].map(|kind| {
            report
                .chain
                .effects
                .iter()
                .find(|effect| effect.kind == kind)
                .unwrap()
                .active
        });
        assert_eq!(clean_actual, [false; 3]);

        let mut paths = std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|value| value == "wav"))
            .collect::<Vec<_>>();
        paths.sort();
        let mut exact = 0;
        for path in &paths {
            let report = inspect_with_training(path, restored.clone()).unwrap();
            let name = path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap()
                .to_ascii_lowercase();
            let expected = [
                name.contains("drive")
                    || name.contains("fuzz")
                    || name.contains("rat")
                    || name.contains("muff"),
                name.contains("delay") || name.contains("echo"),
                name.contains("ambience")
                    || name.contains("dream")
                    || name.contains("reverb")
                    || name.contains("room")
                    || name.contains("hall")
                    || name.contains("plate"),
            ];
            let actual = [Kind::Drive, Kind::Delay, Kind::Reverb].map(|kind| {
                report
                    .chain
                    .effects
                    .iter()
                    .find(|effect| effect.kind == kind)
                    .is_some_and(|effect| effect.active)
            });
            exact += usize::from(actual == expected);
            println!("{} expected {expected:?} actual {actual:?}", path.display());
        }
        println!("clean-only exact {exact}/{}", paths.len());
    }
}
