//! Blind Drive-family subtype and control inference using GFX Classifier.
//!
//! The upstream model is a closed-set classifier for isolated wet guitar. It
//! distinguishes 13 overdrive, distortion, and fuzz units, but it does not
//! prove that Drive is present or infer effect order.

use crate::chain::{Effect, Kind, Param, Subtype};
use anyhow::{Context, Result, bail};
use realfft::{RealFftPlanner, RealToComplex};
use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::owned::InterleavedOwned};
use std::{
    collections::VecDeque,
    io::Cursor,
    sync::{Arc, OnceLock},
};
use tract_onnx::prelude::*;

type Model = Arc<TypedRunnableModel>;

const RATE: u32 = 22_050;
const FFT: usize = 1_024;
const HOP: usize = 512;
const MELS: usize = 128;
const FRAMES: usize = 87;
const WINDOW: usize = 44_100;
const SEGMENTS: usize = 5;
const CANDIDATES: usize = SEGMENTS * 4;
const FX_BYTES: &[u8] = include_bytes!("../models/gfx/fx.onnx");
const SETTINGS_BYTES: &[u8] = include_bytes!("../models/gfx/settings.onnx");

#[derive(Clone, Copy)]
struct Label {
    code: &'static str,
    subtype: Subtype,
}

const LABELS: [Label; 13] = [
    label("808", Subtype::Overdrive),
    label("BD2", Subtype::Overdrive),
    label("BMF", Subtype::Fuzz),
    label("DPL", Subtype::Distortion),
    label("DS1", Subtype::Distortion),
    label("FFC", Subtype::Fuzz),
    label("MGS", Subtype::Overdrive),
    label("OD1", Subtype::Overdrive),
    label("RAT", Subtype::Distortion),
    label("RBM", Subtype::Fuzz),
    label("SD1", Subtype::Overdrive),
    label("TS9", Subtype::Overdrive),
    label("VTB", Subtype::Fuzz),
];

const fn label(code: &'static str, subtype: Subtype) -> Label {
    Label { code, subtype }
}

pub struct Match {
    label: usize,
    score: f64,
    settings: [f64; 3],
    windows: usize,
}

struct Candidate {
    start: usize,
    energy: f64,
    audio: Vec<f32>,
}

pub struct Scan {
    rate: u32,
    window: usize,
    stride: usize,
    start: usize,
    buffer: VecDeque<f32>,
    candidates: Vec<Candidate>,
}

impl Scan {
    pub fn new(rate: u32) -> Self {
        let window = rate as usize * 2;
        Self {
            rate,
            window,
            stride: window / 2,
            start: 0,
            buffer: VecDeque::with_capacity(window),
            candidates: Vec::with_capacity(CANDIDATES + 1),
        }
    }

    pub fn push(&mut self, sample: f32) {
        self.buffer.push_back(sample);
        if self.buffer.len() == self.window {
            self.record();
            for _ in 0..self.stride {
                self.buffer.pop_front();
            }
            self.start += self.stride;
        }
    }

    pub fn finish(mut self) -> Result<Option<Match>> {
        if self.candidates.is_empty() && !self.buffer.is_empty() {
            self.record();
        }
        let mut selected = Vec::with_capacity(SEGMENTS);
        let mut starts = Vec::with_capacity(SEGMENTS);
        for candidate in self.candidates {
            if candidate.energy <= 1.0e-8 {
                continue;
            }
            if starts
                .iter()
                .all(|start: &usize| start.abs_diff(candidate.start) >= self.stride)
            {
                starts.push(candidate.start);
                selected.push(candidate.audio);
            }
            if selected.len() == SEGMENTS {
                break;
            }
        }
        if selected.is_empty() {
            return Ok(None);
        }
        infer(selected, self.rate).map(Some)
    }

    fn record(&mut self) {
        let audio = self.buffer.iter().copied().collect::<Vec<_>>();
        let energy = audio
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum::<f64>();
        self.candidates.push(Candidate {
            start: self.start,
            energy,
            audio,
        });
        self.candidates
            .sort_by(|left, right| right.energy.total_cmp(&left.energy));
        self.candidates.truncate(CANDIDATES);
    }
}

impl Match {
    pub fn effect(&self) -> Effect {
        let label = LABELS[self.label];
        let mut params = vec![knob("Level", self.settings[0])];
        params.push(knob(gain_name(label.code), self.settings[1]));
        if let Some(name) = tone_name(label.code) {
            params.push(knob(name, self.settings[2]));
        }
        Effect {
            kind: Kind::Drive,
            subtype: Some(label.subtype),
            model: Some(display_name(label.code).to_owned()),
            active: true,
            score: self.score,
            evidence: format!(
                "GFX closed-set model · {} high-energy window{} · {} candidate",
                self.windows,
                if self.windows == 1 { "" } else { "s" },
                label.subtype.name()
            ),
            params,
        }
    }
}

fn infer(segments: Vec<Vec<f32>>, rate: u32) -> Result<Match> {
    let fx = model(&FX_MODEL, FX_BYTES, "classifier")?;
    let mut probabilities = [0.0_f64; LABELS.len()];
    let mut features = Vec::with_capacity(segments.len());

    for segment in &segments {
        let mut segment = resample(segment, rate)?;
        segment.resize(WINDOW, 0.0);
        segment.truncate(WINDOW);
        let mel = mel(&segment)?;
        let input = Tensor::from_shape(&[1, 1, MELS, FRAMES], &mel)?;
        let output = fx.run(tvec!(input.into()))?;
        let logits = output[0].to_plain_array_view::<f32>()?;
        if logits.len() != LABELS.len() {
            bail!("GFX classifier output contract changed");
        }
        let values = logits
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        for (total, value) in probabilities.iter_mut().zip(softmax(&values)?) {
            *total += value;
        }
        features.push(mel);
    }

    let count = features.len() as f64;
    for probability in &mut probabilities {
        *probability /= count;
    }
    let label = probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .context("GFX classifier returned no labels")?;

    let settings_model = model(&SETTINGS_MODEL, SETTINGS_BYTES, "settings model")?;
    let mut settings = [0.0_f64; 3];
    for mel in features {
        let audio = Tensor::from_shape(&[1, 1, MELS, FRAMES], &mel)?;
        let label_tensor = Tensor::from_shape(&[1], &[label as i64])?;
        let output = settings_model.run(tvec!(audio.into(), label_tensor.into()))?;
        let values = output[0].to_plain_array_view::<f32>()?;
        if values.len() != settings.len() {
            bail!("GFX settings output contract changed");
        }
        for (total, value) in settings.iter_mut().zip(values.iter()) {
            *total += f64::from(*value);
        }
    }
    for setting in &mut settings {
        *setting = (*setting / count).clamp(0.0, 1.0);
    }

    Ok(Match {
        label,
        score: probabilities[label],
        settings,
        windows: segments.len(),
    })
}

static FX_MODEL: OnceLock<Model> = OnceLock::new();
static SETTINGS_MODEL: OnceLock<Model> = OnceLock::new();

fn model(slot: &'static OnceLock<Model>, bytes: &[u8], name: &str) -> Result<Model> {
    if let Some(model) = slot.get() {
        return Ok(model.clone());
    }
    let mut cursor = Cursor::new(bytes);
    let loaded = tract_onnx::onnx()
        .model_for_read(&mut cursor)
        .with_context(|| format!("could not load the GFX {name}"))?
        .into_optimized()
        .with_context(|| format!("could not optimize the GFX {name}"))?
        .into_runnable()
        .with_context(|| format!("could not prepare the GFX {name}"))?;
    let _ = slot.set(loaded.clone());
    Ok(slot.get().cloned().unwrap_or(loaded))
}

fn resample(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    if rate == RATE {
        return Ok(samples.to_vec());
    }
    let input = InterleavedOwned::new_from(samples.to_vec(), 1, samples.len())
        .context("could not prepare audio for GFX resampling")?;
    let mut resampler = Fft::<f32>::new(rate as usize, RATE as usize, 1_024, 1, FixedSync::Both)
        .context("could not create the GFX resampler")?;
    let output = resampler
        .process_all(&input, samples.len(), None)
        .context("could not resample audio for GFX")?;
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
            .context("could not create the GFX Mel spectrogram")?;
        for band in 0..MELS {
            let mut power = 0.0_f32;
            for (bin, value) in output.iter().enumerate() {
                power += filters[band * output.len() + bin] * value.norm_sqr();
            }
            result[band * FRAMES + frame] = power;
        }
    }
    Ok(result)
}

fn filters() -> &'static [f32] {
    static FILTERS: OnceLock<Vec<f32>> = OnceLock::new();
    FILTERS.get_or_init(|| {
        let bins = FFT / 2 + 1;
        let min = hz_to_mel(0.0);
        let max = hz_to_mel(RATE as f64 / 2.0);
        let points = (0..MELS + 2)
            .map(|index| {
                let mel = min + (max - min) * index as f64 / (MELS + 1) as f64;
                mel_to_hz(mel)
            })
            .collect::<Vec<_>>();
        let mut filters = vec![0.0_f32; MELS * bins];
        for band in 0..MELS {
            let lower = points[band];
            let center = points[band + 1];
            let upper = points[band + 2];
            let norm = 2.0 / (upper - lower);
            for bin in 0..bins {
                let frequency = bin as f64 * RATE as f64 / FFT as f64;
                let lower_slope = (frequency - lower) / (center - lower);
                let upper_slope = (upper - frequency) / (upper - center);
                filters[band * bins + bin] = (lower_slope.min(upper_slope).max(0.0) * norm) as f32;
            }
        }
        filters
    })
}

fn hz_to_mel(frequency: f64) -> f64 {
    let linear = frequency / (200.0 / 3.0);
    if frequency < 1_000.0 {
        linear
    } else {
        15.0 + (frequency / 1_000.0).ln() / (6.4_f64.ln() / 27.0)
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * (200.0 / 3.0)
    } else {
        1_000.0 * ((6.4_f64.ln() / 27.0) * (mel - 15.0)).exp()
    }
}

fn softmax(values: &[f64]) -> Result<Vec<f64>> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        bail!("GFX classifier returned invalid logits");
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

fn display_name(code: &str) -> &str {
    match code {
        "RBM" => "Russian Big Muff",
        _ => code,
    }
}

fn knob(name: &'static str, value: f64) -> Param {
    Param::new(name, value * 100.0, 0.0, 100.0, 1.0, "%")
}

fn gain_name(code: &str) -> &'static str {
    match code {
        "808" => "Overdrive",
        "BMF" | "RBM" => "Sustain",
        "DPL" | "DS1" | "RAT" => "Distortion",
        "FFC" => "Fuzz",
        "MGS" | "OD1" | "SD1" | "TS9" => "Drive",
        "VTB" => "Filter",
        _ => "Gain",
    }
}

fn tone_name(code: &str) -> Option<&'static str> {
    match code {
        "808" | "BD2" | "BMF" | "DS1" | "MGS" | "RBM" | "SD1" | "TS9" => Some("Tone"),
        "RAT" => Some("Filter"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_expose_fuzz_as_a_subtype() {
        for code in ["BMF", "FFC", "RBM", "VTB"] {
            assert_eq!(
                LABELS
                    .iter()
                    .find(|label| label.code == code)
                    .map(|label| label.subtype),
                Some(Subtype::Fuzz)
            );
        }
    }

    #[test]
    fn mel_has_training_shape() {
        let audio = (0..WINDOW)
            .map(|index| (2.0 * std::f32::consts::PI * 440.0 * index as f32 / RATE as f32).sin())
            .collect::<Vec<_>>();
        let mel = mel(&audio).expect("create Mel input");
        assert_eq!(mel.len(), MELS * FRAMES);
        assert!(mel.iter().all(|value| value.is_finite() && *value >= 0.0));
    }

    #[test]
    fn windows_prefer_energy() {
        let mut audio = vec![0.0_f32; WINDOW * 3];
        audio[WINDOW..WINDOW * 2].fill(0.5);
        let mut scan = Scan::new(RATE);
        for sample in audio {
            scan.push(sample);
        }
        let strongest = &scan.candidates[0];
        assert!(strongest.audio.iter().any(|sample| *sample != 0.0));
        assert!(strongest.energy > 1.0);
    }

    #[test]
    fn models_run() {
        let audio = (0..WINDOW)
            .map(|index| (2.0 * std::f32::consts::PI * 220.0 * index as f32 / RATE as f32).sin())
            .collect::<Vec<_>>();
        let mut scan = Scan::new(RATE);
        for sample in audio {
            scan.push(sample);
        }
        let result = scan
            .finish()
            .expect("run GFX models")
            .expect("non-silent match");
        assert!(result.label < LABELS.len());
        assert!((0.0..=1.0).contains(&result.score));
        assert!(
            result
                .settings
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
    }
}
