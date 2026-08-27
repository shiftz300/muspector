use crate::chain::{Effect, Kind, Param};
use anyhow::{Context, Result, bail};
use realfft::{RealFftPlanner, RealToComplex};
use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::owned::InterleavedOwned};
use std::{collections::VecDeque, io::Cursor, sync::Arc};
use tract_onnx::prelude::*;

const RATE: u32 = 22_050;
const FFT: usize = 1_024;
const HOP: usize = 512;
const MELS: usize = 128;
const FRAMES: usize = 87;
const WINDOW: usize = 44_100;
const SEGMENTS: usize = 5;
const CANDIDATES: usize = SEGMENTS * 4;
const LABELS: [&str; 13] = [
    "808", "BD2", "BMF", "DPL", "DS1", "FFC", "MGS", "OD1", "RAT", "RBM", "SD1", "TS9", "VTB",
];
const FX: &[u8] = include_bytes!("../models/gfx/fx.onnx");
const SETTINGS: &[u8] = include_bytes!("../models/gfx/settings.onnx");

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

    pub fn finish(mut self) -> Result<Match> {
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
        infer(selected, self.rate)
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
        params.push(knob(gain_name(label), self.settings[1]));
        if let Some(name) = tone_name(label) {
            params.push(knob(name, self.settings[2]));
        }
        Effect {
            kind: Kind::Drive,
            model: Some(label.to_owned()),
            active: true,
            score: self.score,
            evidence: format!(
                "GFX blind · {} high-energy window{} · guitar drive/fuzz model",
                self.windows,
                if self.windows == 1 { "" } else { "s" }
            ),
            params,
        }
    }
}

#[cfg(test)]
fn inspect(samples: &[f32], rate: u32) -> Result<Match> {
    if samples.is_empty() {
        bail!("GFX blind received no audio");
    }
    let mut scan = Scan::new(rate);
    for sample in samples {
        scan.push(*sample);
    }
    scan.finish()
}

fn infer(segments: Vec<Vec<f32>>, rate: u32) -> Result<Match> {
    if segments.is_empty() {
        bail!("GFX blind could not find a non-silent window");
    }

    let mut fx_cursor = Cursor::new(FX);
    let fx = tract_onnx::onnx()
        .model_for_read(&mut fx_cursor)
        .context("could not load the GFX classifier")?
        .into_optimized()
        .context("could not optimize the GFX classifier")?
        .into_runnable()
        .context("could not prepare the GFX classifier")?;
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
        let values: Vec<f64> = logits.iter().map(|value| f64::from(*value)).collect();
        let softmax = softmax(&values);
        for (total, value) in probabilities.iter_mut().zip(softmax) {
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

    let mut settings_cursor = Cursor::new(SETTINGS);
    let settings_model = tract_onnx::onnx()
        .model_for_read(&mut settings_cursor)
        .context("could not load the GFX settings model")?
        .into_optimized()
        .context("could not optimize the GFX settings model")?
        .into_runnable()
        .context("could not prepare the GFX settings model")?;
    let mut settings = [0.0_f64; 3];
    for mel in features {
        let audio = Tensor::from_shape(&[1, 1, MELS, FRAMES], &mel)?;
        let label_tensor = Tensor::from_shape(&[1], &[label as i64])?;
        let output = settings_model.run(tvec!(audio.into(), label_tensor.into()))?;
        let values = output[0].to_plain_array_view::<f32>()?;
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

fn filters() -> Vec<f32> {
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

fn softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut values = values
        .iter()
        .map(|value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f64>();
    for value in &mut values {
        *value /= total;
    }
    values
}

fn knob(name: &'static str, value: f64) -> Param {
    Param::new(name, value * 100.0, 0.0, 100.0, 1.0, "%")
}

fn gain_name(label: &str) -> &'static str {
    match label {
        "808" => "Overdrive",
        "BMF" | "RBM" => "Sustain",
        "DPL" | "DS1" | "RAT" => "Distortion",
        "FFC" => "Fuzz",
        "MGS" | "OD1" | "SD1" | "TS9" => "Drive",
        "VTB" => "Filter",
        _ => "Gain",
    }
}

fn tone_name(label: &str) -> Option<&'static str> {
    match label {
        "808" | "BD2" | "BMF" | "DS1" | "MGS" | "RBM" | "SD1" | "TS9" => Some("Tone"),
        "RAT" => Some("Filter"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_has_training_shape() {
        let audio = (0..WINDOW)
            .map(|index| (2.0 * std::f32::consts::PI * 440.0 * index as f32 / RATE as f32).sin())
            .collect::<Vec<_>>();
        let mel = mel(&audio).unwrap();
        assert_eq!(mel.len(), MELS * FRAMES);
        assert!(mel.iter().all(|value| value.is_finite() && *value >= 0.0));
        let total = mel.iter().map(|value| f64::from(*value)).sum::<f64>();
        let maximum = mel.iter().copied().fold(0.0_f32, f32::max);
        assert!((total - 328_413.42).abs() < 700.0, "{total}");
        assert!((maximum - 2_164.79).abs() < 5.0, "{maximum}");
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
        let result = inspect(&audio, RATE).unwrap();
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
