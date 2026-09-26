//! Development graybox Wet-to-Clean preview renderer.
//!
//! The UI stores effects in forward signal order. Reconstruction deliberately
//! traverses that graph in reverse; individual processors never inspect their
//! neighbours or their position in the chain.

use crate::chain::{Chain, Effect, Kind, Subtype};

pub struct Graybox {
    channels: usize,
    processors: Vec<Processor>,
}

impl Graybox {
    pub fn new(chain: &Chain, rate: u32, channels: usize) -> Self {
        let processors = chain
            .effects
            .iter()
            .rev()
            .filter(|effect| effect.active)
            .filter_map(|effect| Processor::new(effect, rate, channels))
            .collect();
        Self {
            channels,
            processors,
        }
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        debug_assert!(samples.len().is_multiple_of(self.channels));
        for processor in &mut self.processors {
            processor.process(samples, self.channels);
        }
        for sample in samples {
            *sample = sample.clamp(-8.0, 8.0);
        }
    }
}

enum Processor {
    Drive(Drive),
    Comp(Comp),
    Eq(Eq),
    Delay(Delay),
}

impl Processor {
    fn new(effect: &Effect, rate: u32, channels: usize) -> Option<Self> {
        match effect.kind {
            Kind::Drive => Some(Self::Drive(Drive::new(effect))),
            Kind::Comp => Some(Self::Comp(Comp::new(effect, rate, channels))),
            Kind::Eq => Some(Self::Eq(Eq::new(effect, rate, channels))),
            Kind::Delay => Delay::new(effect, rate, channels).map(Self::Delay),
            // A gate has destroyed samples and cannot be inverted. Reverb is
            // intentionally frozen until its acceptance route is reopened.
            Kind::Gate | Kind::Reverb => None,
        }
    }

    fn process(&mut self, samples: &mut [f32], channels: usize) {
        match self {
            Self::Drive(processor) => processor.process(samples),
            Self::Comp(processor) => processor.process(samples, channels),
            Self::Eq(processor) => processor.process(samples, channels),
            Self::Delay(processor) => processor.process(samples, channels),
        }
    }
}

struct Drive {
    drive: f32,
    level: f32,
    shape: Subtype,
}

impl Drive {
    fn new(effect: &Effect) -> Self {
        let gain = effect
            .params
            .iter()
            .find(|param| !matches!(param.name, "Level" | "Tone" | "Filter"))
            .map_or(0.5, |param| param.normal());
        let level = effect
            .params
            .iter()
            .find(|param| param.name == "Level")
            .map_or(1.0, |param| {
                if param.unit == "dB" {
                    10.0_f32.powf(param.value as f32 / 20.0)
                } else {
                    0.45 + 0.40 * param.normal()
                }
            });
        Self {
            drive: 1.8 + 6.2 * gain,
            level: level.max(0.05),
            shape: effect.subtype.unwrap_or(Subtype::Overdrive),
        }
    }

    fn process(&self, samples: &mut [f32]) {
        for sample in samples {
            let shaped = (*sample / self.level).clamp(-0.995, 0.995);
            let restored = match self.shape {
                Subtype::Overdrive => shaped.atanh(),
                Subtype::Distortion => {
                    ((shaped * std::f32::consts::FRAC_PI_2).clamp(-1.45, 1.45)).tan() / 1.8
                }
                Subtype::Fuzz => inverse_cubic(shaped),
            };
            *sample = restored / self.drive;
        }
    }
}

fn inverse_cubic(target: f32) -> f32 {
    let mut low = -1.5_f32;
    let mut high = 1.5_f32;
    for _ in 0..20 {
        let value = (low + high) * 0.5;
        let shaped = value - value.powi(3) / 6.75;
        if shaped < target {
            low = value;
        } else {
            high = value;
        }
    }
    (low + high) * 0.5
}

struct Comp {
    ratio: f32,
    attack: f32,
    release: f32,
    envelope: Vec<f32>,
}

impl Comp {
    fn new(effect: &Effect, rate: u32, channels: usize) -> Self {
        let ratio = value(effect, "Ratio", 2.0) as f32;
        let attack_ms = value(effect, "Attack", 20.0) as f32;
        let release_ms = value(effect, "Release", 180.0) as f32;
        let coefficient =
            |milliseconds: f32| (-1.0 / (milliseconds.max(1.0) * 0.001 * rate as f32)).exp();
        Self {
            ratio,
            attack: coefficient(attack_ms),
            release: coefficient(release_ms),
            envelope: vec![0.0; channels],
        }
    }

    fn process(&mut self, samples: &mut [f32], channels: usize) {
        const THRESHOLD_DB: f32 = -24.0;
        for frame in samples.chunks_exact_mut(channels) {
            for (channel, sample) in frame.iter_mut().enumerate() {
                let level = sample.abs().max(1.0e-8);
                let coefficient = if level > self.envelope[channel] {
                    self.attack
                } else {
                    self.release
                };
                self.envelope[channel] =
                    coefficient * self.envelope[channel] + (1.0 - coefficient) * level;
                let wet_db = 20.0 * self.envelope[channel].max(1.0e-8).log10();
                if wet_db > THRESHOLD_DB && self.ratio > 1.0 {
                    let clean_db = THRESHOLD_DB + (wet_db - THRESHOLD_DB) * self.ratio;
                    let gain = 10.0_f32.powf((clean_db - wet_db) / 20.0).min(8.0);
                    *sample *= gain;
                }
            }
        }
    }
}

struct Eq {
    filters: Vec<Biquad>,
}

impl Eq {
    fn new(effect: &Effect, rate: u32, channels: usize) -> Self {
        let rate = rate as f32;
        Self {
            filters: vec![
                Biquad::low_shelf(rate, 120.0, -value(effect, "Low", 0.0) as f32, channels),
                Biquad::peak(
                    rate,
                    1_000.0,
                    0.8,
                    -value(effect, "Mid", 0.0) as f32,
                    channels,
                ),
                Biquad::high_shelf(rate, 6_000.0, -value(effect, "High", 0.0) as f32, channels),
            ],
        }
    }

    fn process(&mut self, samples: &mut [f32], channels: usize) {
        for filter in &mut self.filters {
            filter.process(samples, channels);
        }
    }
}

struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: Vec<f32>,
    z2: Vec<f32>,
}

impl Biquad {
    fn normalized(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32, channels: usize) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: vec![0.0; channels],
            z2: vec![0.0; channels],
        }
    }

    fn peak(rate: f32, frequency: f32, q: f32, gain_db: f32, channels: usize) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * std::f32::consts::PI * frequency.min(rate * 0.45) / rate;
        let alpha = omega.sin() / (2.0 * q);
        Self::normalized(
            1.0 + alpha * a,
            -2.0 * omega.cos(),
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * omega.cos(),
            1.0 - alpha / a,
            channels,
        )
    }

    fn low_shelf(rate: f32, frequency: f32, gain_db: f32, channels: usize) -> Self {
        Self::shelf(rate, frequency, gain_db, channels, false)
    }

    fn high_shelf(rate: f32, frequency: f32, gain_db: f32, channels: usize) -> Self {
        Self::shelf(rate, frequency, gain_db, channels, true)
    }

    fn shelf(rate: f32, frequency: f32, gain_db: f32, channels: usize, high: bool) -> Self {
        let a = 10.0_f32.powf(gain_db / 40.0);
        let omega = 2.0 * std::f32::consts::PI * frequency.min(rate * 0.45) / rate;
        let cosine = omega.cos();
        let sine = omega.sin();
        let root = a.sqrt() * sine * std::f32::consts::SQRT_2;
        let (b0, b1, b2, a0, a1, a2) = if high {
            (
                a * ((a + 1.0) + (a - 1.0) * cosine + root),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cosine),
                a * ((a + 1.0) + (a - 1.0) * cosine - root),
                (a + 1.0) - (a - 1.0) * cosine + root,
                2.0 * ((a - 1.0) - (a + 1.0) * cosine),
                (a + 1.0) - (a - 1.0) * cosine - root,
            )
        } else {
            (
                a * ((a + 1.0) - (a - 1.0) * cosine + root),
                2.0 * a * ((a - 1.0) - (a + 1.0) * cosine),
                a * ((a + 1.0) - (a - 1.0) * cosine - root),
                (a + 1.0) + (a - 1.0) * cosine + root,
                -2.0 * ((a - 1.0) + (a + 1.0) * cosine),
                (a + 1.0) + (a - 1.0) * cosine - root,
            )
        };
        Self::normalized(b0, b1, b2, a0, a1, a2, channels)
    }

    fn process(&mut self, samples: &mut [f32], channels: usize) {
        for frame in samples.chunks_exact_mut(channels) {
            for (channel, sample) in frame.iter_mut().enumerate() {
                let output = self.b0 * *sample + self.z1[channel];
                self.z1[channel] = self.b1 * *sample - self.a1 * output + self.z2[channel];
                self.z2[channel] = self.b2 * *sample - self.a2 * output;
                *sample = output;
            }
        }
    }
}

struct Delay {
    mix: f32,
    feedback: f32,
    position: usize,
    clean: Vec<f32>,
    echo: Vec<f32>,
}

impl Delay {
    fn new(effect: &Effect, rate: u32, channels: usize) -> Option<Self> {
        let time_ms = value(effect, "Time", 0.0);
        let frames = (time_ms * f64::from(rate) / 1_000.0).round() as usize;
        if frames == 0 {
            return None;
        }
        Some(Self {
            mix: (value(effect, "Mix", 0.0) as f32 / 100.0).clamp(0.0, 0.7),
            feedback: (value(effect, "Feedback", 0.0) as f32 / 100.0).clamp(0.0, 0.9),
            position: 0,
            clean: vec![0.0; frames * channels],
            echo: vec![0.0; frames * channels],
        })
    }

    fn process(&mut self, samples: &mut [f32], channels: usize) {
        let delay_frames = self.clean.len() / channels;
        for frame in samples.chunks_exact_mut(channels) {
            let base = self.position * channels;
            for (channel, wet) in frame.iter_mut().enumerate() {
                let index = base + channel;
                let delayed_echo = self.clean[index] + self.feedback * self.echo[index];
                let clean = (*wet - self.mix * delayed_echo) / (1.0 - self.mix).max(0.05);
                self.clean[index] = clean;
                self.echo[index] = delayed_echo;
                *wet = clean;
            }
            self.position = (self.position + 1) % delay_frames;
        }
    }
}

fn value(effect: &Effect, name: &str, fallback: f64) -> f64 {
    effect
        .params
        .iter()
        .find(|param| param.name == name)
        .map_or(fallback, |param| param.value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::Param;

    fn effect(kind: Kind, params: Vec<Param>) -> Effect {
        Effect {
            kind,
            subtype: None,
            model: None,
            active: true,
            score: 1.0,
            evidence: String::new(),
            params,
        }
    }

    #[test]
    fn bypassed_effects_do_not_change_audio() {
        let mut drive = effect(
            Kind::Drive,
            vec![Param::new("Gain", 20.0, 0.0, 30.0, 0.5, "dB")],
        );
        drive.active = false;
        let chain = Chain {
            effects: vec![drive],
            score: 1.0,
        };
        let mut renderer = Graybox::new(&chain, 48_000, 1);
        let mut audio = vec![-0.5, 0.0, 0.5];
        let original = audio.clone();
        renderer.process(&mut audio);
        assert_eq!(audio, original);
    }

    #[test]
    fn active_drive_changes_audio_without_non_finite_samples() {
        let chain = Chain {
            effects: vec![effect(
                Kind::Drive,
                vec![
                    Param::new("Gain", 20.0, 0.0, 30.0, 0.5, "dB"),
                    Param::new("Level", 0.0, -18.0, 12.0, 0.5, "dB"),
                ],
            )],
            score: 1.0,
        };
        let mut renderer = Graybox::new(&chain, 48_000, 1);
        let mut audio = vec![-0.5, 0.0, 0.5];
        renderer.process(&mut audio);
        assert_ne!(audio, vec![-0.5, 0.0, 0.5]);
        assert!(audio.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn graph_order_is_reversed_by_the_executor() {
        let drive = effect(
            Kind::Drive,
            vec![Param::new("Gain", 15.0, 0.0, 30.0, 0.5, "dB")],
        );
        let eq = effect(
            Kind::Eq,
            vec![
                Param::new("Low", 8.0, -12.0, 12.0, 0.5, "dB"),
                Param::new("Mid", -5.0, -12.0, 12.0, 0.5, "dB"),
                Param::new("High", 6.0, -12.0, 12.0, 0.5, "dB"),
            ],
        );
        let input = (0..512)
            .map(|index| ((index as f32 * 0.17).sin() * 0.6).clamp(-0.9, 0.9))
            .collect::<Vec<_>>();
        let mut first = Graybox::new(
            &Chain {
                effects: vec![drive.clone(), eq.clone()],
                score: 1.0,
            },
            48_000,
            1,
        );
        let mut second = Graybox::new(
            &Chain {
                effects: vec![eq, drive],
                score: 1.0,
            },
            48_000,
            1,
        );
        let mut left = input.clone();
        let mut right = input;
        first.process(&mut left);
        second.process(&mut right);
        assert!(left.iter().zip(right).any(|(a, b)| (a - b).abs() > 1.0e-5));
    }

    #[test]
    fn delay_inverse_recovers_its_causal_forward_model() {
        let rate = 1_000;
        let delay = 4;
        let feedback = 0.35_f32;
        let mix = 0.3_f32;
        let mut clean = vec![0.0_f32; 64];
        clean[0] = 0.7;
        clean[11] = -0.4;
        let mut echo = vec![0.0_f32; clean.len()];
        let mut wet = vec![0.0_f32; clean.len()];
        for index in 0..clean.len() {
            if index >= delay {
                echo[index] = clean[index - delay] + feedback * echo[index - delay];
            }
            wet[index] = (1.0 - mix) * clean[index] + mix * echo[index];
        }
        let chain = Chain {
            effects: vec![effect(
                Kind::Delay,
                vec![
                    Param::new("Time", 4.0, 1.0, 100.0, 1.0, "ms"),
                    Param::new("Feedback", 35.0, 0.0, 90.0, 1.0, "%"),
                    Param::new("Mix", 30.0, 0.0, 70.0, 1.0, "%"),
                ],
            )],
            score: 1.0,
        };
        let mut renderer = Graybox::new(&chain, rate, 1);
        renderer.process(&mut wet);
        assert!(
            wet.iter()
                .zip(clean)
                .all(|(restored, expected)| (restored - expected).abs() < 1.0e-5)
        );
    }
}
