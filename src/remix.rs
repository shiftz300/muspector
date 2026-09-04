//! App-side model contract shared by inference, rendering, and projects.
//!
//! Values use physical units. Models may normalize them internally, but the
//! application boundary must never expose anonymous logits as knob values.
//! Implementations belong in a model crate; Muspector only adapts that crate to
//! [`ModelRuntime`] and consumes the types in this module.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
pub const AUDIO_QUALITY_SCHEMA_VERSION: u32 = 1;
pub const MAX_MODEL_FRAMES: usize = 480_000;
pub const MAX_RENDER_BLOCK_FRAMES: usize = 2_048;

/// Interleaved audio passed across the model boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioView<'a> {
    pub samples: &'a [f32],
    pub sample_rate: u32,
    pub channels: usize,
}

impl AudioView<'_> {
    pub fn validate(self) -> Result<()> {
        if self.sample_rate == 0
            || self.channels == 0
            || !self.samples.len().is_multiple_of(self.channels)
            || self.samples.len() / self.channels > MAX_MODEL_FRAMES
            || self.samples.iter().any(|sample| !sample.is_finite())
        {
            bail!("invalid or oversized interleaved audio at the model boundary");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChainEstimate {
    pub chain: ChainSpec,
    pub confidence: f32,
}

impl ChainEstimate {
    pub fn validate(&self) -> Result<()> {
        self.chain.validate()?;
        if !in_range(self.confidence, 0.0, 1.0) {
            bail!("model confidence must be finite and between zero and one");
        }
        Ok(())
    }
}

/// Fixed geometry negotiated before a runtime enters the audio callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamSpec {
    pub sample_rate: u32,
    pub channels: usize,
    pub max_block_frames: usize,
}

impl StreamSpec {
    pub fn validate(self) -> Result<()> {
        if self.sample_rate == 0
            || self.channels == 0
            || self.max_block_frames == 0
            || self.max_block_frames > MAX_RENDER_BLOCK_FRAMES
        {
            bail!("invalid real-time stream geometry");
        }
        Ok(())
    }

    pub fn validate_block(self, input: AudioView<'_>, output: &[f32]) -> Result<()> {
        self.validate()?;
        input.validate()?;
        if input.sample_rate != self.sample_rate
            || input.channels != self.channels
            || input.samples.len() != output.len()
            || input.samples.len() / input.channels > self.max_block_frames
        {
            bail!("audio block does not match the configured stream");
        }
        Ok(())
    }
}

/// The only surface Muspector needs from a future `muspector-models` runtime.
///
/// A local adapter should implement this trait for the dependency's runtime so
/// neither repository needs to know about GPUI or the other's internal types.
/// `configure` and `update_chain` run away from the audio callback. Once
/// configured, `process_block` must not allocate, lock, perform I/O, or change
/// the interleaved audio geometry; it writes into the caller-owned output.
pub trait ModelRuntime: Send {
    fn infer_segment(&mut self, audio: AudioView<'_>) -> Result<Option<ChainEstimate>>;

    fn configure(&mut self, stream: StreamSpec) -> Result<()>;

    fn update_chain(&mut self, chain: &ChainSpec, smoothing_frames: usize) -> Result<()>;

    fn process_block(&mut self, input: AudioView<'_>, output: &mut [f32]) -> Result<()>;

    fn latency_frames(&self) -> usize;

    fn reset(&mut self);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioQualityPolicy {
    pub schema: u32,
    pub source_audio_immutable: bool,
    pub analysis_copy_may_downmix_or_resample: bool,
    pub render_preserves_frames: bool,
    pub render_preserves_channels: bool,
    pub render_preserves_sample_rate: bool,
    pub processing_sample_format: String,
    pub automatic_normalization: bool,
    pub automatic_limiting: bool,
    pub automatic_dither: bool,
    pub lossy_reencoding: bool,
    pub bypass_max_absolute_error: f32,
}

impl AudioQualityPolicy {
    pub fn loss_preserving() -> Self {
        Self {
            schema: AUDIO_QUALITY_SCHEMA_VERSION,
            source_audio_immutable: true,
            analysis_copy_may_downmix_or_resample: true,
            render_preserves_frames: true,
            render_preserves_channels: true,
            render_preserves_sample_rate: true,
            processing_sample_format: "float32".to_owned(),
            automatic_normalization: false,
            automatic_limiting: false,
            automatic_dither: false,
            lossy_reencoding: false,
            bypass_max_absolute_error: 0.0,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != AUDIO_QUALITY_SCHEMA_VERSION
            || !self.source_audio_immutable
            || !self.render_preserves_frames
            || !self.render_preserves_channels
            || !self.render_preserves_sample_rate
            || self.processing_sample_format != "float32"
            || self.automatic_normalization
            || self.automatic_limiting
            || self.automatic_dither
            || self.lossy_reencoding
            || self.bypass_max_absolute_error != 0.0
        {
            bail!("project audio-quality policy is not loss-preserving");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChainSpec {
    pub schema: u32,
    pub effects: Vec<EffectSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EffectSpec {
    Drive {
        gain_db: f32,
        tone: f32,
        level_db: f32,
    },
    Delay {
        time_ms: f32,
        feedback: f32,
        mix: f32,
    },
    Reverb {
        decay_s: f32,
        damping: f32,
        mix: f32,
    },
}

impl ChainSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA_VERSION {
            bail!("unsupported remix chain schema {}", self.schema);
        }
        let mut seen = [false; 3];
        for effect in &self.effects {
            let (slot, valid) = match effect {
                EffectSpec::Drive {
                    gain_db,
                    tone,
                    level_db,
                } => (
                    0,
                    in_range(*gain_db, 0.0, 30.0)
                        && in_range(*tone, 0.0, 1.0)
                        && in_range(*level_db, -18.0, 12.0),
                ),
                EffectSpec::Delay {
                    time_ms,
                    feedback,
                    mix,
                } => (
                    1,
                    in_range(*time_ms, 40.0, 1_000.0)
                        && in_range(*feedback, 0.0, 0.9)
                        && in_range(*mix, 0.0, 0.7),
                ),
                EffectSpec::Reverb {
                    decay_s,
                    damping,
                    mix,
                } => (
                    2,
                    in_range(*decay_s, 0.2, 8.0)
                        && in_range(*damping, 0.0, 1.0)
                        && in_range(*mix, 0.0, 0.7),
                ),
            };
            if !valid {
                bail!("remix effect contains a non-finite or out-of-range control");
            }
            if seen[slot] {
                bail!("remix chain contains the same effect family more than once");
            }
            seen[slot] = true;
        }
        Ok(())
    }
}

fn in_range(value: f32, minimum: f32, maximum: f32) -> bool {
    value.is_finite() && (minimum..=maximum).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_physical_controls_round_trip() {
        let spec = ChainSpec {
            schema: SCHEMA_VERSION,
            effects: vec![
                EffectSpec::Drive {
                    gain_db: 12.0,
                    tone: 0.5,
                    level_db: -3.0,
                },
                EffectSpec::Delay {
                    time_ms: 375.0,
                    feedback: 0.4,
                    mix: 0.3,
                },
            ],
        };
        spec.validate().expect("valid UI chain spec");
        let json = serde_json::to_string(&spec).expect("serialize chain spec");
        assert!(json.contains("\"kind\":\"drive\""));
        assert!(json.contains("\"time_ms\":375.0"));
    }

    #[test]
    fn duplicate_and_invalid_effects_are_rejected() {
        let drive = EffectSpec::Drive {
            gain_db: 12.0,
            tone: 0.5,
            level_db: 0.0,
        };
        let duplicate = ChainSpec {
            schema: SCHEMA_VERSION,
            effects: vec![drive.clone(), drive],
        };
        assert!(duplicate.validate().is_err());
        let invalid = ChainSpec {
            schema: SCHEMA_VERSION,
            effects: vec![EffectSpec::Delay {
                time_ms: 0.0,
                feedback: 0.2,
                mix: 0.3,
            }],
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn audio_quality_policy_forbids_hidden_processing() {
        let policy = AudioQualityPolicy::loss_preserving();
        policy.validate().expect("loss-preserving policy");
        assert!(policy.source_audio_immutable);
        assert!(!policy.automatic_normalization);
        assert!(!policy.automatic_limiting);
        assert!(!policy.lossy_reencoding);
        assert_eq!(policy.bypass_max_absolute_error, 0.0);
    }

    #[test]
    fn model_boundary_rejects_invalid_or_oversized_audio() {
        let input = AudioView {
            samples: &[0.25, -0.25, 0.5, -0.5],
            sample_rate: 48_000,
            channels: 2,
        };
        input.validate().expect("valid stereo audio");

        let invalid = AudioView {
            samples: &[f32::NAN],
            sample_rate: 48_000,
            channels: 1,
        };
        assert!(invalid.validate().is_err());

        let oversized = vec![0.0; MAX_MODEL_FRAMES + 1];
        assert!(
            AudioView {
                samples: &oversized,
                sample_rate: 48_000,
                channels: 1,
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn real_time_blocks_use_preallocated_matching_output() {
        let stream = StreamSpec {
            sample_rate: 48_000,
            channels: 2,
            max_block_frames: 256,
        };
        let input = AudioView {
            samples: &[0.25, -0.25, 0.5, -0.5],
            sample_rate: 48_000,
            channels: 2,
        };
        stream
            .validate_block(input, &[0.0; 4])
            .expect("matching caller-owned output");
        assert!(stream.validate_block(input, &[0.0; 2]).is_err());
        assert!(
            StreamSpec {
                max_block_frames: MAX_RENDER_BLOCK_FRAMES + 1,
                ..stream
            }
            .validate()
            .is_err()
        );
    }
}
