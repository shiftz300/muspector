use anyhow::{Context, Result, bail};
use realfft::{RealFftPlanner, RealToComplex};
use std::{
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
const LIMIT: usize = 4096;
const POINTS: usize = 192;

#[derive(Clone, Debug)]
pub struct Point {
    pub min: f32,
    pub max: f32,
    pub level: f64,
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
    pub crest: f64,
    pub centroid: f64,
    pub rolloff: f64,
    pub low: f64,
    pub mid: f64,
    pub high: f64,
    pub clips: u64,
    pub spectrum: Vec<f64>,
    pub profile: Profile,
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

pub fn inspect(path: &Path) -> Result<Report> {
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
    let mut decoder = symphonia::default::get_codecs()
        .make(params, &DecoderOptions::default())
        .context("没有适合这个音轨的解码器")?;

    let mut signal = Signal::new(rate);
    let mut timeline = Timeline::new(rate);
    let mut sample_buffer = None;
    let mut frames = 0_u64;
    let mut count = 0_u64;
    let mut sum = 0.0_f64;
    let mut peak = 0.0_f64;
    let mut clips = 0_u64;

    loop {
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
        frames += (samples.len() / channels) as u64;

        for frame in samples.chunks_exact(channels) {
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
            timeline.push(mono);
            signal.push(mono)?;
        }
    }

    if count == 0 || frames == 0 {
        bail!("音轨没有可分析的采样");
    }

    let rms = (sum / count as f64).sqrt();
    let spectrum = signal.finish()?;
    Ok(Report {
        path: path.to_path_buf(),
        codec,
        rate,
        channels,
        duration: frames as f64 / rate as f64,
        peak: db(peak),
        rms: db(rms),
        crest: db(peak) - db(rms),
        centroid: spectrum.centroid,
        rolloff: spectrum.rolloff,
        low: spectrum.low,
        mid: spectrum.mid,
        high: spectrum.high,
        clips,
        spectrum: spectrum.curve,
        profile: timeline.finish(),
    })
}

#[derive(Clone, Copy)]
struct Meter {
    min: f32,
    max: f32,
    square: f64,
    count: u64,
}

impl Meter {
    fn empty() -> Self {
        Self {
            min: 1.0,
            max: -1.0,
            square: 0.0,
            count: 0,
        }
    }

    fn push(&mut self, sample: f32) {
        self.min = self.min.min(sample);
        self.max = self.max.max(sample);
        self.square += f64::from(sample) * f64::from(sample);
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
        self.count += other.count;
    }

    fn point(self) -> Point {
        if self.count == 0 {
            return Point {
                min: 0.0,
                max: 0.0,
                level: -72.0,
            };
        }
        let rms = (self.square / self.count as f64).sqrt();
        Point {
            min: self.min,
            max: self.max,
            level: db(rms).clamp(-72.0, 0.0),
        }
    }
}

struct Timeline {
    span: u64,
    meter: Meter,
    bins: Vec<Meter>,
}

impl Timeline {
    fn new(rate: u32) -> Self {
        Self {
            span: (rate as u64 / 40).max(1),
            meter: Meter::empty(),
            bins: Vec::with_capacity(LIMIT),
        }
    }

    fn push(&mut self, sample: f32) {
        self.meter.push(sample);
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

        let size = self.bins.len().div_ceil(POINTS).max(1);
        let points = self
            .bins
            .chunks(size)
            .map(|chunk| {
                let mut meter = Meter::empty();
                for item in chunk {
                    meter.merge(*item);
                }
                meter.point()
            })
            .collect();
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

        Ok(Self {
            centroid,
            rolloff,
            low: band(20.0, 250.0),
            mid: band(250.0, 4_000.0),
            high: band(4_000.0, 20_000.0),
            curve: chart(bins, rate),
        })
    }
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
            timeline.push(phase.sin() * 0.5);
        }
        let profile = timeline.finish();
        assert!(!profile.points.is_empty());
        assert!(profile.points.len() <= POINTS);
        assert!(profile.points.iter().all(|point| point.level.is_finite()));
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

        let report = inspect(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(report.codec, "PCM");
        assert_eq!(report.rate, rate);
        assert_eq!(report.channels, 1);
        assert!((report.duration - 0.1).abs() < 0.001);
        assert!((report.centroid - 440.0).abs() < 30.0);
        assert_eq!(report.spectrum.len(), CHART);
        assert!(report.spectrum.iter().all(|value| value.is_finite()));
        assert!(!report.profile.points.is_empty());
        assert!(report.profile.points.len() <= POINTS);
    }
}
