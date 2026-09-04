use crate::{
    chain::{Chain, Kind, Param},
    remix::{AudioQualityPolicy, ChainSpec, EffectSpec, SCHEMA_VERSION},
};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct Saved {
    pub project: PathBuf,
    pub audio: PathBuf,
}

#[derive(Serialize)]
struct Document {
    version: u32,
    source: String,
    audio: String,
    confidence: f64,
    audio_quality: AudioQualityPolicy,
    reconstruction: ChainSpec,
    effects: Vec<Effect>,
}

#[derive(Serialize)]
struct Effect {
    position: usize,
    kind: String,
    model: Option<String>,
    enabled: bool,
    confidence: f64,
    evidence: String,
    parameters: Vec<Parameter>,
}

#[derive(Serialize)]
struct Parameter {
    name: String,
    value: f64,
    unit: String,
    minimum: f64,
    maximum: f64,
}

pub fn save(
    project: Option<&Path>,
    logical: &Path,
    source: &Path,
    audio_dirty: bool,
    chain: &Chain,
) -> Result<Saved> {
    let project = project
        .map(Path::to_owned)
        .unwrap_or_else(|| sibling(logical, "json"));
    let audio = if audio_dirty {
        let audio = sibling(logical, "wav");
        fs::copy(source, &audio).with_context(|| {
            format!(
                "could not save edited audio from {} to {}",
                source.display(),
                audio.display()
            )
        })?;
        audio
    } else {
        source.to_owned()
    };
    let reconstruction = reconstruction(chain);
    reconstruction
        .validate()
        .context("could not encode the reconstructed effect chain")?;
    let audio_quality = AudioQualityPolicy::loss_preserving();
    audio_quality
        .validate()
        .context("could not enforce the audio-quality policy")?;
    let document = Document {
        version: 3,
        source: logical.to_string_lossy().into_owned(),
        audio: audio.to_string_lossy().into_owned(),
        confidence: chain.score,
        audio_quality,
        reconstruction,
        effects: chain
            .effects
            .iter()
            .enumerate()
            .map(|(position, effect)| Effect {
                position: position + 1,
                kind: effect.kind.name().to_owned(),
                model: effect.model.clone(),
                enabled: effect.active,
                confidence: effect.score,
                evidence: effect.evidence.clone(),
                parameters: effect
                    .params
                    .iter()
                    .map(|parameter| Parameter {
                        name: parameter.name.to_owned(),
                        value: parameter.value,
                        unit: parameter.unit.to_owned(),
                        minimum: parameter.min,
                        maximum: parameter.max,
                    })
                    .collect(),
            })
            .collect(),
    };
    let bytes = serde_json::to_vec_pretty(&document).context("could not encode the project")?;
    fs::write(&project, bytes)
        .with_context(|| format!("could not write project {}", project.display()))?;
    Ok(Saved { project, audio })
}

fn reconstruction(chain: &Chain) -> ChainSpec {
    let effects = chain
        .active()
        .filter_map(|effect| match effect.kind {
            Kind::Drive => Some(EffectSpec::Drive {
                gain_db: value(&effect.params, "Gain") as f32,
                tone: value(&effect.params, "Tone") as f32 / 100.0,
                level_db: value(&effect.params, "Level") as f32,
            }),
            Kind::Delay => Some(EffectSpec::Delay {
                time_ms: value(&effect.params, "Time") as f32,
                feedback: value(&effect.params, "Feedback") as f32 / 100.0,
                mix: value(&effect.params, "Mix") as f32 / 100.0,
            }),
            Kind::Reverb => Some(EffectSpec::Reverb {
                decay_s: value(&effect.params, "Decay") as f32,
                damping: value(&effect.params, "Damp") as f32 / 100.0,
                mix: value(&effect.params, "Mix") as f32 / 100.0,
            }),
            _ => None,
        })
        .collect();
    ChainSpec {
        schema: SCHEMA_VERSION,
        effects,
    }
}

fn value(params: &[Param], name: &str) -> f64 {
    params
        .iter()
        .find(|param| param.name == name)
        .map_or(0.0, |param| param.value)
}

fn sibling(source: &Path, extension: &str) -> PathBuf {
    let parent = source.parent().unwrap_or_else(|| Path::new("."));
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("audio");
    parent.join(format!("{stem}.muspector.{extension}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_project_and_edited_audio_without_overwriting_source() {
        let id = std::process::id();
        let logical = std::env::temp_dir().join(format!("muspector-project-{id}.mp3"));
        let working = std::env::temp_dir().join(format!("muspector-project-{id}-working.wav"));
        fs::write(&logical, b"original").expect("write logical source");
        fs::write(&working, b"edited").expect("write working audio");
        let chain = Chain {
            effects: Vec::new(),
            score: 0.72,
        };

        let saved = save(None, &logical, &working, true, &chain).expect("save project");
        assert_eq!(fs::read(&logical).expect("read original"), b"original");
        assert_eq!(fs::read(&saved.audio).expect("read edited"), b"edited");
        let json = fs::read_to_string(&saved.project).expect("read project");
        assert!(json.contains("\"version\": 3"));
        assert!(json.contains("\"confidence\": 0.72"));
        assert!(json.contains("\"source_audio_immutable\": true"));
        assert!(json.contains("\"automatic_normalization\": false"));
        assert!(json.contains("\"lossy_reencoding\": false"));
        assert!(json.contains("\"reconstruction\""));

        for path in [logical, working, saved.audio, saved.project] {
            let _ = fs::remove_file(path);
        }
    }
}
