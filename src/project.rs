use crate::{chain::Chain, remix::AudioQualityPolicy};
use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::{Builder, NamedTempFile};

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
    let audio_quality = AudioQualityPolicy::loss_preserving();
    audio_quality
        .validate()
        .context("could not enforce the audio-quality policy")?;
    let document = Document {
        version: 4,
        source: logical.to_string_lossy().into_owned(),
        audio: if audio_dirty {
            sibling(logical, "wav")
        } else {
            source.to_owned()
        }
        .to_string_lossy()
        .into_owned(),
        confidence: chain.score,
        audio_quality,
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
    let mut project_file = temporary_beside(&project, ".muspector-project-")?;
    project_file
        .write_all(&bytes)
        .with_context(|| format!("could not write project {}", project.display()))?;
    project_file
        .as_file()
        .sync_all()
        .with_context(|| format!("could not flush project {}", project.display()))?;

    let audio = if audio_dirty {
        let audio = sibling(logical, "wav");
        let mut input = File::open(source)
            .with_context(|| format!("could not open edited audio {}", source.display()))?;
        let mut audio_file = temporary_beside(&audio, ".muspector-audio-")?;
        copy(&mut input, &mut audio_file).with_context(|| {
            format!(
                "could not stage edited audio from {} beside {}",
                source.display(),
                audio.display()
            )
        })?;
        audio_file
            .as_file()
            .sync_all()
            .with_context(|| format!("could not flush edited audio {}", audio.display()))?;
        persist(audio_file, &audio, "edited audio")?;
        audio
    } else {
        source.to_owned()
    };
    persist(project_file, &project, "project")?;
    Ok(Saved { project, audio })
}

fn temporary_beside(target: &Path, prefix: &str) -> Result<NamedTempFile> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    Builder::new()
        .prefix(prefix)
        .tempfile_in(parent)
        .with_context(|| {
            format!(
                "could not create a temporary file beside {}",
                target.display()
            )
        })
}

fn copy(input: &mut File, output: &mut NamedTempFile) -> Result<u64> {
    let mut buffer = [0_u8; 64 * 1024];
    let mut written = 0_u64;
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            return Ok(written);
        }
        output.write_all(&buffer[..count])?;
        written = written.saturating_add(count as u64);
    }
}

fn persist(file: NamedTempFile, target: &Path, kind: &str) -> Result<()> {
    file.persist(target)
        .map(|_| ())
        .map_err(|error| error.error)
        .with_context(|| format!("could not replace {kind} {}", target.display()))
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
    use std::fs;

    #[test]
    fn saves_project_and_edited_audio_without_overwriting_source() {
        let directory = tempfile::tempdir().expect("create test directory");
        let logical = directory.path().join("source.mp3");
        let working = directory.path().join("working.wav");
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
        assert!(json.contains("\"version\": 4"));
        assert!(json.contains("\"confidence\": 0.72"));
        assert!(json.contains("\"source_audio_immutable\": true"));
        assert!(json.contains("\"automatic_normalization\": false"));
        assert!(json.contains("\"lossy_reencoding\": false"));
        assert!(!json.contains("\"reconstruction\""));
    }

    #[test]
    fn project_staging_failure_does_not_replace_saved_audio() {
        let directory = tempfile::tempdir().expect("create test directory");
        let logical = directory.path().join("source.mp3");
        let working = directory.path().join("working.wav");
        let saved_audio = sibling(&logical, "wav");
        fs::write(&logical, b"original").expect("write logical source");
        fs::write(&working, b"new edit").expect("write working audio");
        fs::write(&saved_audio, b"previous edit").expect("write previous audio");
        let missing_project = directory.path().join("missing/project.json");

        let result = save(
            Some(&missing_project),
            &logical,
            &working,
            true,
            &Chain {
                effects: Vec::new(),
                score: 0.5,
            },
        );

        assert!(result.is_err());
        assert_eq!(
            fs::read(saved_audio).expect("read previous audio"),
            b"previous edit"
        );
    }
}
