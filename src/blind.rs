use crate::{
    chain::{Chain, Kind},
    identity,
};
use anyhow::{Context, Result, bail};
use realfft::{RealFftPlanner, RealToComplex};
use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::owned::InterleavedOwned};
use std::{
    env, fs,
    io::{Cursor, Read},
    path::PathBuf,
    sync::Arc,
};
use tract_onnx::prelude::*;

type Model = Arc<TypedRunnableModel>;

const RATE: u32 = 44_100;
const FFT: usize = 2_048;
const HOP: usize = 1_024;
const MELS: usize = 128;
const FRAMES: usize = 216;
const TEMPORAL_BANDS: usize = 8;
const WINDOW: usize = 220_500;
const WINDOW_HOP: usize = WINDOW / 2;
const FMIN: f64 = 30.0;
const FMAX: f64 = 16_000.0;
const QUERY: usize = 259;
const PAIR_INPUT: usize = QUERY * 5;
const PROFILE_VALUES: usize = QUERY * 2;
const ROUTED_PROFILE_VALUES: usize = PROFILE_VALUES * 2;
const DRIVE_DELAY_REFERENCE_WINDOWS: usize = 1;
const REVERB_REFERENCE_WINDOWS: usize = 3;

const DRIVE_DELAY_ENCODER: &[u8] = include_bytes!("../models/inspector/drive-delay-encoder.onnx");
const DRIVE_DELAY_HEAD: &[u8] = include_bytes!("../models/inspector/drive-delay-head.onnx");
const REVERB_ENCODER: &[u8] = include_bytes!("../models/inspector/reverb-encoder.onnx");
const REVERB_HEAD: &[u8] = include_bytes!("../models/inspector/reverb-head.onnx");
const REVERB_VERIFIER: &[u8] = include_bytes!("../models/inspector/reverb-verifier.onnx");
const DEVICE_PROFILE: &[u8] = include_bytes!("../models/inspector/routed-device-profile.bin");
const TRAINING_MAGIC: &[u8; 8] = b"MUSPTRN1";
const TRAINING_VERSION: u32 = 4;
const ROUTED_PROFILE_FLAG: u32 = 1;
const TRAINING_HEADER: usize = 8 + 5 * size_of::<u32>() + 3 * size_of::<f32>();
const MAX_NAME_BYTES: usize = 512;

const DRIVE_DELAY_SCALE: [f64; 3] = [1.556_447_149, 0.481_221_408, 1.188_732_624];
const DRIVE_DELAY_BIAS: [f64; 3] = [-3.630_986_691, -0.329_666_764, -4.136_132_240];
const REVERB_SCALE: [f64; 3] = [2.313_131_571, 0.506_870_329, 1.665_598_392];
const REVERB_BIAS: [f64; 3] = [-0.863_560_617, -0.273_496_419, -7.076_041_698];
const REVERB_VERIFIER_SCALE: f64 = 0.353_591_529_675_706_3;
const REVERB_VERIFIER_BIAS: f64 = -0.183_689_819_860_455_07;
const REVERB_VERIFIER_THRESHOLD: f64 = 0.342;
const DEFAULT_THRESHOLD: [f64; 3] = [0.05, 0.55, 0.59];
const USER_THRESHOLD_FLOOR: [f64; 3] = [0.05, 0.55, 0.59];
const ONNX_THRESHOLD_EPSILON: f64 = 1.0e-6;

#[derive(Clone)]
pub struct Training(Arc<TrainingData>);

struct TrainingData {
    name: String,
    drive_delay_profile: Profile,
    reverb_profile: Profile,
    profile_bytes: Vec<u8>,
    thresholds: [f64; 3],
}

pub struct Match {
    scores: [f64; 3],
    detected: [bool; 3],
    windows: usize,
    identity: Option<identity::Match>,
}

pub struct Scan {
    rate: u32,
    samples: Vec<f32>,
    training: Training,
}

struct Profile {
    mean: Vec<f32>,
    deviation: Vec<f32>,
}

impl Training {
    pub fn embedded() -> Self {
        Self::from_parts(
            "Public Clean · Routed".to_owned(),
            DEVICE_PROFILE.to_vec(),
            DEFAULT_THRESHOLD,
        )
        .expect("embedded Inspector training must be valid")
    }

    pub fn load_active() -> Self {
        active_path()
            .and_then(|path| fs::read(path).ok())
            .and_then(|bytes| Self::import(&bytes).ok())
            .unwrap_or_else(Self::embedded)
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    pub fn calibrated(&self) -> bool {
        false
    }

    pub fn summary(&self) -> &'static str {
        "Clean reference"
    }

    pub fn save_active(&self) -> Result<()> {
        let path = active_path().context("could not locate the Inspector profile directory")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .context("could not create the Inspector profile directory")?;
        }
        fs::write(path, self.export()).context("could not save the active Inspector training")
    }

    pub fn export(&self) -> Vec<u8> {
        let name = self.0.name.as_bytes();
        let mut bytes =
            Vec::with_capacity(TRAINING_HEADER + name.len() + self.0.profile_bytes.len());
        bytes.extend_from_slice(TRAINING_MAGIC);
        push_u32(&mut bytes, TRAINING_VERSION);
        push_u32(&mut bytes, ROUTED_PROFILE_FLAG);
        push_u32(&mut bytes, name.len() as u32);
        push_u32(&mut bytes, (PROFILE_VALUES * size_of::<f32>()) as u32);
        push_u32(&mut bytes, (PROFILE_VALUES * size_of::<f32>()) as u32);
        for threshold in self.0.thresholds {
            bytes.extend_from_slice(&(threshold as f32).to_le_bytes());
        }
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&self.0.profile_bytes);
        bytes
    }

    pub fn import(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < TRAINING_HEADER || &bytes[..8] != TRAINING_MAGIC {
            bail!("not a Muspector training file");
        }
        let mut cursor = Cursor::new(&bytes[8..]);
        let version = read_u32(&mut cursor)?;
        if version != TRAINING_VERSION {
            bail!("unsupported Muspector training version {version}");
        }
        let flags = read_u32(&mut cursor)?;
        let name_len = read_u32(&mut cursor)? as usize;
        let profile_len = read_u32(&mut cursor)? as usize;
        let adapter_len = read_u32(&mut cursor)? as usize;
        let mut thresholds = [0.0; 3];
        for threshold in &mut thresholds {
            let mut value = [0; 4];
            cursor.read_exact(&mut value)?;
            *threshold = f64::from(f32::from_le_bytes(value));
        }
        if name_len > MAX_NAME_BYTES
            || profile_len != PROFILE_VALUES * size_of::<f32>()
            || adapter_len != PROFILE_VALUES * size_of::<f32>()
            || flags != ROUTED_PROFILE_FLAG
            || thresholds
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            bail!("invalid Muspector training contract");
        }
        let body = TRAINING_HEADER
            .checked_add(name_len)
            .and_then(|size| size.checked_add(profile_len))
            .and_then(|size| size.checked_add(adapter_len))
            .context("Muspector training file is too large")?;
        if bytes.len() != body {
            bail!("truncated or trailing Muspector training data");
        }
        let name_start = TRAINING_HEADER;
        let profile_start = name_start + name_len;
        let profile_end = profile_start + profile_len + adapter_len;
        let name = std::str::from_utf8(&bytes[name_start..profile_start])
            .context("training name is not UTF-8")?
            .to_owned();
        let profile = bytes[profile_start..profile_end].to_vec();
        Self::from_parts(name, profile, thresholds)
    }

    pub fn from_clean(samples: &[f32], rate: u32, name: String) -> Result<Self> {
        if samples.is_empty() || samples.iter().all(|sample| sample.abs() <= 1.0e-8) {
            bail!("the clean reference contains no usable audio");
        }
        let audio = resample(samples, rate)?;
        let starts = window_starts(audio.len());
        let drive_delay_encoder = drive_delay_encoder_model()?;
        let drive_delay_head = drive_delay_head_model()?;
        let reverb_encoder = reverb_encoder_model()?;
        let reverb_head = reverb_head_model()?;
        let mut drive_delay_queries = Vec::with_capacity(DRIVE_DELAY_REFERENCE_WINDOWS);
        let mut reverb_queries = Vec::with_capacity(REVERB_REFERENCE_WINDOWS);
        for (index, start) in starts
            .into_iter()
            .take(REVERB_REFERENCE_WINDOWS)
            .enumerate()
        {
            let mel = mel(&audio_window(&audio, start))?;
            if index < DRIVE_DELAY_REFERENCE_WINDOWS {
                drive_delay_queries.push(encode(&drive_delay_encoder, &mel)?);
            }
            reverb_queries.push(encode(&reverb_encoder, &mel)?);
        }
        let drive_delay_profile = Profile {
            mean: column_mean(&drive_delay_queries),
            deviation: column_deviation(&drive_delay_queries, 1.0e-4),
        };
        let reverb_profile = Profile {
            mean: column_mean(&reverb_queries),
            deviation: column_deviation(&reverb_queries, 1.0e-4),
        };
        let mut maximum = [0.0_f64; 3];
        for query in &drive_delay_queries {
            let logits = pair_logits(
                &drive_delay_head,
                query,
                &drive_delay_profile.mean,
                &drive_delay_profile.deviation,
            )?;
            for index in 0..2 {
                maximum[index] = maximum[index].max(sigmoid(
                    f64::from(logits[index]) * DRIVE_DELAY_SCALE[index] + DRIVE_DELAY_BIAS[index],
                ));
            }
        }
        for query in &reverb_queries {
            let logits = pair_logits(
                &reverb_head,
                query,
                &reverb_profile.mean,
                &reverb_profile.deviation,
            )?;
            maximum[2] = maximum[2].max(sigmoid(
                f64::from(logits[2]) * REVERB_SCALE[2] + REVERB_BIAS[2],
            ));
        }
        let thresholds = std::array::from_fn(|index| {
            (maximum[index] + 0.02).clamp(USER_THRESHOLD_FLOOR[index], 0.95)
        });
        let profile_bytes = drive_delay_profile
            .to_bytes()
            .into_iter()
            .chain(reverb_profile.to_bytes())
            .collect();
        Self::from_parts(name, profile_bytes, thresholds)
    }

    fn from_parts(name: String, profile_bytes: Vec<u8>, thresholds: [f64; 3]) -> Result<Self> {
        if name.trim().is_empty() || name.len() > MAX_NAME_BYTES {
            bail!("invalid Inspector training name");
        }
        if profile_bytes.len() != ROUTED_PROFILE_VALUES * size_of::<f32>() {
            bail!("Inspector routed device profile size changed");
        }
        let split = PROFILE_VALUES * size_of::<f32>();
        let drive_delay_profile = Profile::from_bytes(&profile_bytes[..split])?;
        let reverb_profile = Profile::from_bytes(&profile_bytes[split..])?;
        Ok(Self(Arc::new(TrainingData {
            name,
            drive_delay_profile,
            reverb_profile,
            profile_bytes,
            thresholds,
        })))
    }
}

impl Profile {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != PROFILE_VALUES * size_of::<f32>() {
            bail!("Inspector device profile size changed");
        }
        let values = bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            bail!("Inspector profile contains non-finite values");
        }
        Ok(Self {
            mean: values[0..QUERY].to_vec(),
            deviation: values[QUERY..QUERY * 2].to_vec(),
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.mean
            .iter()
            .chain(&self.deviation)
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }
}

impl Scan {
    pub fn with_training(rate: u32, training: Training) -> Self {
        Self {
            rate,
            samples: Vec::with_capacity(rate as usize * 5),
            training,
        }
    }

    pub fn push(&mut self, sample: f32) {
        self.samples.push(sample);
    }

    pub fn finish(self) -> Result<Match> {
        infer(&self.samples, self.rate, &self.training)
    }
}

impl Match {
    pub fn apply(&self, chain: &mut Chain) {
        let windows = self.windows;
        chain.classify([
            (
                Kind::Drive,
                self.detected[0],
                self.scores[0],
                format!(
                    "Inspector Routed · {windows} window{}\nClean-reference pair + temporal verifier",
                    plural(windows)
                ),
            ),
            (
                Kind::Delay,
                self.detected[1],
                self.scores[1],
                format!(
                    "Inspector Routed · {windows} window{}\nClean-reference pair",
                    plural(windows)
                ),
            ),
            (
                Kind::Reverb,
                self.detected[2],
                self.scores[2],
                format!(
                    "Inspector Routed · {windows} window{}\nClean-reference pair",
                    plural(windows)
                ),
            ),
        ]);
        if let Some(identity) = &self.identity
            && let Some(drive) = chain
                .effects
                .iter_mut()
                .find(|effect| effect.kind == Kind::Drive && effect.active)
        {
            drive.model.clone_from(&identity.model);
            drive.evidence.push_str(&format!(
                "\nAFx pedal catalog · {} window{} · {:.0}%{}",
                identity.windows,
                plural(identity.windows),
                identity.score * 100.0,
                if identity.model.is_some() {
                    ""
                } else {
                    " · no known model"
                }
            ));
        }
    }
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

#[cfg(test)]
fn inspect(samples: &[f32], rate: u32) -> Result<Match> {
    if samples.is_empty() {
        bail!("Inspector Routed received no audio");
    }
    infer(samples, rate, &Training::embedded())
}

fn infer(samples: &[f32], rate: u32, training: &Training) -> Result<Match> {
    if samples.is_empty() || samples.iter().all(|sample| sample.abs() <= 1.0e-8) {
        bail!("Inspector Routed could not find non-silent audio");
    }
    let audio = resample(samples, rate)?;
    let starts = window_starts(audio.len());
    let drive_delay_encoder = drive_delay_encoder_model()?;
    let drive_delay_head = drive_delay_head_model()?;
    let reverb_encoder = reverb_encoder_model()?;
    let reverb_head = reverb_head_model()?;
    let reverb_verifier = reverb_verifier_model()?;

    let mut scores = std::array::from_fn(|_| Vec::with_capacity(starts.len()));
    let mut reverb_verifier_scores = Vec::with_capacity(starts.len());
    for start in &starts {
        let window = audio_window(&audio, *start);
        let mel = mel(&window)?;
        let drive_delay_query = encode(&drive_delay_encoder, &mel)?;
        let drive_delay_logits = pair_logits(
            &drive_delay_head,
            &drive_delay_query,
            &training.0.drive_delay_profile.mean,
            &training.0.drive_delay_profile.deviation,
        )?;
        for index in 0..2 {
            scores[index].push(sigmoid(
                f64::from(drive_delay_logits[index]) * DRIVE_DELAY_SCALE[index]
                    + DRIVE_DELAY_BIAS[index],
            ));
        }
        let reverb_query = encode(&reverb_encoder, &mel)?;
        let reverb_logits = pair_logits(
            &reverb_head,
            &reverb_query,
            &training.0.reverb_profile.mean,
            &training.0.reverb_profile.deviation,
        )?;
        let reverb_pair_probabilities = std::array::from_fn(|index| {
            sigmoid(f64::from(reverb_logits[index]) * REVERB_SCALE[index] + REVERB_BIAS[index])
        });
        scores[2].push(reverb_pair_probabilities[2]);
        reverb_verifier_scores.push(reverb_verifier_probability(
            &reverb_verifier,
            &temporal_bands(&mel),
            &reverb_pair_probabilities,
        )?);
    }
    let mut scores = scores.map(|values| top_two_mean(&values));
    let reverb_verifier_score = top_two_mean(&reverb_verifier_scores);
    let mut detected = std::array::from_fn(|index| {
        scores[index] + ONNX_THRESHOLD_EPSILON >= training.0.thresholds[index]
    });
    detected[2] =
        detected[2] && reverb_verifier_score + ONNX_THRESHOLD_EPSILON >= REVERB_VERIFIER_THRESHOLD;
    scores[2] = scores[2].min(reverb_verifier_score);
    Ok(Match {
        scores,
        detected,
        windows: starts.len(),
        identity: if detected[0] && !detected[1] && !detected[2] {
            identity::infer(samples, rate).ok().flatten()
        } else {
            None
        },
    })
}

fn model(bytes: &[u8], name: &str) -> Result<Model> {
    let mut cursor = Cursor::new(bytes);
    tract_onnx::onnx()
        .model_for_read(&mut cursor)
        .with_context(|| format!("could not load the Inspector {name}"))?
        .into_optimized()
        .with_context(|| format!("could not optimize the Inspector {name}"))?
        .into_runnable()
        .with_context(|| format!("could not prepare the Inspector {name}"))
}

fn drive_delay_encoder_model() -> Result<Model> {
    model(DRIVE_DELAY_ENCODER, "Drive/Delay encoder")
}

fn drive_delay_head_model() -> Result<Model> {
    model(DRIVE_DELAY_HEAD, "Drive/Delay pair head")
}

fn reverb_encoder_model() -> Result<Model> {
    model(REVERB_ENCODER, "Reverb encoder")
}

fn reverb_head_model() -> Result<Model> {
    model(REVERB_HEAD, "Reverb pair head")
}

fn reverb_verifier_model() -> Result<Model> {
    model(REVERB_VERIFIER, "Reverb temporal verifier")
}

fn encode(model: &Model, mel: &[f32]) -> Result<Vec<f32>> {
    let input = Tensor::from_shape(&[1, 1, MELS, FRAMES], mel)?;
    let output = model.run(tvec!(input.into()))?;
    let embedding = output[0].to_plain_array_view::<f32>()?;
    let blind_logits = output[1].to_plain_array_view::<f32>()?;
    if embedding.len() != QUERY - 3 || blind_logits.len() != 3 {
        bail!("Inspector encoder output contract changed");
    }
    let mut query = Vec::with_capacity(QUERY);
    query.extend(embedding.iter().copied());
    query.extend(blind_logits.iter().copied());
    Ok(query)
}

fn pair_logits(model: &Model, query: &[f32], mean: &[f32], deviation: &[f32]) -> Result<Vec<f32>> {
    let features = relative(query, mean, deviation);
    let input = Tensor::from_shape(&[1, PAIR_INPUT], &features)?;
    let output = model.run(tvec!(input.into()))?;
    let logits = output[0].to_plain_array_view::<f32>()?;
    if logits.len() != 3 {
        bail!("Inspector pair-head output contract changed");
    }
    Ok(logits.iter().copied().collect())
}

fn temporal_bands(mel: &[f32]) -> Vec<f32> {
    debug_assert_eq!(mel.len(), MELS * FRAMES);
    let mels_per_band = MELS / TEMPORAL_BANDS;
    let mut result = vec![0.0; TEMPORAL_BANDS * FRAMES];
    for band in 0..TEMPORAL_BANDS {
        for mel_band in 0..mels_per_band {
            let source = (band * mels_per_band + mel_band) * FRAMES;
            for frame in 0..FRAMES {
                result[band * FRAMES + frame] += mel[source + frame];
            }
        }
    }
    for value in &mut result {
        *value /= mels_per_band as f32;
    }
    result
}

fn reverb_verifier_probability(
    model: &Model,
    bands: &[f32],
    pair_probabilities: &[f64; 3],
) -> Result<f64> {
    let bands = Tensor::from_shape(&[1, TEMPORAL_BANDS, FRAMES], bands)?;
    let probabilities = pair_probabilities.map(|value| value as f32);
    let probabilities = Tensor::from_shape(&[1, 3], &probabilities)?;
    let output = model.run(tvec!(bands.into(), probabilities.into()))?;
    let logits = output[0].to_plain_array_view::<f32>()?;
    if logits.len() != 1 {
        bail!("Inspector Reverb verifier output contract changed");
    }
    Ok(sigmoid(
        f64::from(logits[0]) * REVERB_VERIFIER_SCALE + REVERB_VERIFIER_BIAS,
    ))
}

fn relative(query: &[f32], mean: &[f32], deviation: &[f32]) -> Vec<f32> {
    debug_assert_eq!(query.len(), mean.len());
    debug_assert_eq!(query.len(), deviation.len());
    let mut result = Vec::with_capacity(query.len() * 5);
    result.extend_from_slice(query);
    result.extend_from_slice(mean);
    result.extend(query.iter().zip(mean).map(|(query, mean)| query - mean));
    result.extend(
        query
            .iter()
            .zip(mean)
            .map(|(query, mean)| (query - mean).abs()),
    );
    result.extend_from_slice(deviation);
    result
}

fn window_starts(length: usize) -> Vec<usize> {
    if length <= WINDOW {
        return vec![0];
    }
    let mut starts = (0..=length - WINDOW)
        .step_by(WINDOW_HOP)
        .collect::<Vec<_>>();
    let final_start = length - WINDOW;
    if starts.last().copied() != Some(final_start) {
        starts.push(final_start);
    }
    starts
}

fn audio_window(audio: &[f32], start: usize) -> Vec<f32> {
    let end = (start + WINDOW).min(audio.len());
    let mut window = Vec::with_capacity(WINDOW);
    window.extend_from_slice(&audio[start.min(audio.len())..end]);
    window.resize(WINDOW, 0.0);
    window
}

fn resample(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    if rate == RATE {
        return Ok(samples.to_vec());
    }
    let input = InterleavedOwned::new_from(samples.to_vec(), 1, samples.len())
        .context("could not prepare audio for Inspector resampling")?;
    let mut resampler = Fft::<f32>::new(rate as usize, RATE as usize, 2_048, 1, FixedSync::Both)
        .context("could not create the Inspector resampler")?;
    let output = resampler
        .process_all(&input, samples.len(), None)
        .context("could not resample audio for Inspector")?;
    Ok(output.take_data())
}

fn mel(samples: &[f32]) -> Result<Vec<f32>> {
    debug_assert_eq!(samples.len(), WINDOW);
    let filters = filters();
    let window = (0..FFT)
        .map(|index| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / FFT as f32).cos())
        .collect::<Vec<_>>();
    let mut planner = RealFftPlanner::<f32>::new();
    let fft: Arc<dyn RealToComplex<f32>> = planner.plan_fft_forward(FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let mut result = vec![0.0_f32; MELS * FRAMES];
    for frame in 0..FRAMES {
        let center = frame * HOP;
        for index in 0..FFT {
            let source = center + index;
            input[index] = if source < FFT / 2 || source >= samples.len() + FFT / 2 {
                0.0
            } else {
                samples[source - FFT / 2]
            } * window[index];
        }
        fft.process(&mut input, &mut output)
            .context("could not create the Inspector Mel spectrogram")?;
        for band in 0..MELS {
            let power = output
                .iter()
                .enumerate()
                .map(|(bin, value)| filters[band * output.len() + bin] * value.norm_sqr())
                .sum::<f32>()
                .max(1.0e-10);
            result[band * FRAMES + frame] = 10.0 * power.log10();
        }
    }
    let peak = result.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for value in &mut result {
        *value = value.max(peak - 80.0);
    }
    let average = mean(&result);
    let deviation = sample_deviation(&result, average).max(1.0e-5);
    for value in &mut result {
        *value = (*value - average) / deviation;
    }
    Ok(result)
}

fn filters() -> Vec<f32> {
    let bins = FFT / 2 + 1;
    let lower = hz_to_mel(FMIN);
    let upper = hz_to_mel(FMAX);
    let points = (0..MELS + 2)
        .map(|index| {
            let mel = lower + (upper - lower) * index as f64 / (MELS + 1) as f64;
            mel_to_hz(mel)
        })
        .collect::<Vec<_>>();
    let mut filters = vec![0.0_f32; MELS * bins];
    for band in 0..MELS {
        let left = points[band];
        let center = points[band + 1];
        let right = points[band + 2];
        let normalization = 2.0 / (right - left);
        for bin in 0..bins {
            let frequency = bin as f64 * f64::from(RATE) / FFT as f64;
            let rise = (frequency - left) / (center - left);
            let fall = (right - frequency) / (right - center);
            filters[band * bins + bin] = (rise.min(fall).max(0.0) * normalization) as f32;
        }
    }
    filters
}

fn hz_to_mel(frequency: f64) -> f64 {
    2_595.0 * (1.0 + frequency / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2_595.0) - 1.0)
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn sample_deviation(values: &[f32], average: f32) -> f32 {
    (values
        .iter()
        .map(|value| {
            let delta = value - average;
            delta * delta
        })
        .sum::<f32>()
        / (values.len() - 1) as f32)
        .sqrt()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}

fn top_two_mean(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(|left, right| right.total_cmp(left));
    values.iter().take(2).sum::<f64>() / values.len().min(2) as f64
}

fn column_mean(rows: &[Vec<f32>]) -> Vec<f32> {
    let width = rows.first().map_or(0, Vec::len);
    let mut result = vec![0.0; width];
    for row in rows {
        debug_assert_eq!(row.len(), width);
        for (total, value) in result.iter_mut().zip(row) {
            *total += value;
        }
    }
    for value in &mut result {
        *value /= rows.len() as f32;
    }
    result
}

fn column_deviation(rows: &[Vec<f32>], floor: f32) -> Vec<f32> {
    let average = column_mean(rows);
    let mut result = vec![0.0; average.len()];
    for row in rows {
        for ((square, value), mean) in result.iter_mut().zip(row).zip(&average) {
            *square += (value - mean).powi(2);
        }
    }
    for value in &mut result {
        *value = (*value / rows.len() as f32).sqrt() + floor;
    }
    result
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut bytes = [0; 4];
    cursor.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn active_path() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    let root = env::var_os("HOME").map(PathBuf::from).map(|home| {
        home.join("Library")
            .join("Application Support")
            .join("Muspector")
    });
    #[cfg(target_os = "windows")]
    let root = env::var_os("APPDATA")
        .map(PathBuf::from)
        .map(|path| path.join("Muspector"));
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let root = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        })
        .map(|path| path.join("muspector"));
    root.map(|path| path.join("inspector.musp-training"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_matches_training_contract() {
        let audio = (0..WINDOW)
            .map(|index| (2.0 * std::f32::consts::PI * 440.0 * index as f32 / RATE as f32).sin())
            .collect::<Vec<_>>();
        let mel = mel(&audio).unwrap();
        assert_eq!(mel.len(), MELS * FRAMES);
        assert!(mel.iter().all(|value| value.is_finite()));
        assert!(mean(&mel).abs() < 1.0e-3, "{}", mean(&mel));
        let deviation = sample_deviation(&mel, mean(&mel));
        assert!((deviation - 1.0).abs() < 1.0e-3, "{deviation}");
    }

    #[test]
    fn windows_match_python_audio_windows() {
        assert_eq!(window_starts(WINDOW - 1), vec![0]);
        assert_eq!(window_starts(WINDOW), vec![0]);
        assert_eq!(window_starts(WINDOW + 1), vec![0, 1]);
        assert_eq!(window_starts(WINDOW * 2), vec![0, WINDOW_HOP, WINDOW]);
    }

    #[test]
    fn profile_has_exported_shape() {
        let training = Training::embedded();
        assert_eq!(training.0.drive_delay_profile.mean.len(), QUERY);
        assert_eq!(training.0.drive_delay_profile.deviation.len(), QUERY);
        assert_eq!(training.0.reverb_profile.mean.len(), QUERY);
        assert_eq!(training.0.reverb_profile.deviation.len(), QUERY);
        assert_eq!(training.0.profile_bytes.len(), ROUTED_PROFILE_VALUES * 4);
    }

    #[test]
    fn training_round_trip_preserves_contract() {
        let training = Training::embedded();
        let restored = Training::import(&training.export()).unwrap();
        assert_eq!(restored.name(), training.name());
        assert!(!restored.calibrated());
        for (restored, original) in restored.0.thresholds.iter().zip(training.0.thresholds) {
            assert!((restored - original).abs() < 1.0e-7);
        }
        assert_eq!(restored.0.profile_bytes, training.0.profile_bytes);
    }

    #[test]
    fn training_rejects_legacy_single_profile() {
        let mut bytes = Training::embedded().export();
        bytes[8..12].copy_from_slice(&3_u32.to_le_bytes());
        let error = match Training::import(&bytes) {
            Ok(_) => panic!("legacy training unexpectedly imported"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            "unsupported Muspector training version 3"
        );
    }

    #[test]
    fn models_run() {
        let audio = (0..WINDOW)
            .map(|index| {
                (2.0 * std::f32::consts::PI * 220.0 * index as f32 / RATE as f32).sin() * 0.25
            })
            .collect::<Vec<_>>();
        let result = inspect(&audio, RATE).unwrap();
        assert_eq!(result.windows, 1);
        assert!(
            result
                .scores
                .iter()
                .all(|score| score.is_finite() && (0.0..=1.0).contains(score))
        );
        let python = [2.948_372_284e-9, 0.004_525_280_107, 8.246_233_100e-10];
        for (rust, python) in result.scores.iter().zip(python) {
            assert!(
                (rust - python).abs() < 2.0e-5,
                "Rust {rust} != Python {python}"
            );
        }

        let mel = mel(&audio).unwrap();
        let training = Training::embedded();
        let query = encode(&reverb_encoder_model().unwrap(), &mel).unwrap();
        let logits = pair_logits(
            &reverb_head_model().unwrap(),
            &query,
            &training.0.reverb_profile.mean,
            &training.0.reverb_profile.deviation,
        )
        .unwrap();
        let pair_probabilities = std::array::from_fn(|index| {
            sigmoid(f64::from(logits[index]) * REVERB_SCALE[index] + REVERB_BIAS[index])
        });
        let probability = reverb_verifier_probability(
            &reverb_verifier_model().unwrap(),
            &temporal_bands(&mel),
            &pair_probabilities,
        )
        .unwrap();
        assert!(
            (probability - 0.963_555_932).abs() < 2.0e-5,
            "Rust verifier {probability} != Python 0.963555932"
        );
    }
}
