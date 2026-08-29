use anyhow::{Context, Result, bail};
use rubato::{Fft, FixedSync, Resampler, audioadapter_buffers::owned::InterleavedOwned};
use serde::Deserialize;
use sha2::{Digest, Sha256};
#[cfg(feature = "embedded-identity")]
use std::io::Cursor;
use std::sync::{Arc, OnceLock};
#[cfg(not(feature = "embedded-identity"))]
use std::{env, fs, io::Read, path::PathBuf};
use tract_onnx::prelude::*;

type Model = Arc<TypedRunnableModel>;

const RATE: u32 = 48_000;
const WINDOW: usize = 240_000;
const WINDOWS: usize = 3;
#[cfg(feature = "embedded-identity")]
const EMBEDDED_MODEL: &[u8] = include_bytes!("../models/inspector/afx-pedal-identity.onnx");
#[cfg(feature = "embedded-identity")]
const EMBEDDED_CONFIG: &[u8] = include_bytes!("../models/inspector/afx-pedal-identity.json");
const CATALOG: [&str; 7] = [
    "BluesDriver",
    "RAT",
    "TubeScreamer",
    "BigMuff",
    "MetalMuff",
    "FuzzyLogic",
    "SillyFuzz",
];

#[derive(Debug)]
pub struct Match {
    pub model: Option<String>,
    pub score: f64,
    pub windows: usize,
}

#[derive(Deserialize)]
struct Config {
    schema: u32,
    labels: Vec<String>,
    display_labels: Vec<String>,
    threshold: f64,
    model_sha256: String,
    license_scope: String,
    user_gradient_updates: u32,
}

struct Runtime {
    model: Model,
    labels: Vec<String>,
    threshold: f64,
}

static RUNTIME: OnceLock<Option<Runtime>> = OnceLock::new();

pub fn infer(samples: &[f32], rate: u32) -> Result<Option<Match>> {
    let Some(runtime) = runtime() else {
        return Ok(None);
    };
    if samples.is_empty() || samples.iter().all(|sample| sample.abs() <= 1.0e-8) {
        return Ok(None);
    }
    let audio = resample(samples, rate)?;
    let starts = energetic_windows(&audio);
    let mut combined = vec![0.0_f64; runtime.labels.len()];
    for start in &starts {
        let mut window = vec![0.0_f32; WINDOW];
        let available = (audio.len() - *start).min(WINDOW);
        window[..available].copy_from_slice(&audio[*start..*start + available]);
        let input = Tensor::from_shape(&[1, 1, WINDOW], &window)?;
        let output = runtime.model.run(tvec!(input.into()))?;
        let logits = output[0].to_plain_array_view::<f32>()?;
        let known = output[1].to_plain_array_view::<f32>()?;
        if logits.len() != runtime.labels.len() || known.len() != 1 {
            bail!("AFx identity output contract changed");
        }
        let probability = softmax(
            logits
                .as_slice()
                .context("identity logits are not contiguous")?,
        );
        let known = sigmoid(f64::from(known[0]));
        for (total, probability) in combined.iter_mut().zip(probability) {
            *total += probability * known;
        }
    }
    for value in &mut combined {
        *value /= starts.len() as f64;
    }
    let (candidate, score) = combined
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .context("AFx identity catalog is empty")?;
    Ok(Some(Match {
        model: (score >= runtime.threshold).then(|| runtime.labels[candidate].clone()),
        score,
        windows: starts.len(),
    }))
}

fn runtime() -> Option<&'static Runtime> {
    RUNTIME
        .get_or_init(|| load_runtime().ok().flatten())
        .as_ref()
}

#[cfg(feature = "embedded-identity")]
fn load_runtime() -> Result<Option<Runtime>> {
    let config = parse_config(EMBEDDED_CONFIG)?;
    verify_hash(EMBEDDED_MODEL, &config.model_sha256)?;
    let model = tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(EMBEDDED_MODEL))
        .context("could not load the embedded AFx identity model")?
        .into_optimized()
        .context("could not optimize the embedded AFx identity model")?
        .into_runnable()
        .context("could not prepare the embedded AFx identity model")?;
    Ok(Some(Runtime {
        model,
        labels: config.display_labels,
        threshold: config.threshold,
    }))
}

fn parse_config(bytes: &[u8]) -> Result<Config> {
    let config: Config =
        serde_json::from_slice(bytes).context("could not parse the AFx identity package")?;
    if config.schema != 1
        || config.user_gradient_updates != 0
        || config.labels.len() != config.display_labels.len()
        || config.labels != CATALOG
        || !config.threshold.is_finite()
        || !(0.0..=1.0).contains(&config.threshold)
        || config.license_scope != "non-commercial research"
        || config.model_sha256.len() != 64
    {
        bail!("invalid AFx identity package contract");
    }
    Ok(config)
}

#[cfg(feature = "embedded-identity")]
fn verify_hash(bytes: &[u8], expected: &str) -> Result<()> {
    let mut digest = Sha256::new();
    digest.update(bytes);
    if format!("{:x}", digest.finalize()) != expected {
        bail!("AFx identity model hash does not match its package metadata");
    }
    Ok(())
}

#[cfg(not(feature = "embedded-identity"))]
fn load_runtime() -> Result<Option<Runtime>> {
    let Some(model_path) = model_path() else {
        return Ok(None);
    };
    if !model_path.is_file() {
        return Ok(None);
    }
    let config_path = model_path.with_extension("json");
    let config = parse_config(
        &fs::read(&config_path)
            .with_context(|| format!("could not read {}", config_path.display()))?,
    )?;
    let mut source = fs::File::open(&model_path)
        .with_context(|| format!("could not read {}", model_path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = source
            .read(&mut buffer)
            .context("could not hash the AFx identity model")?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    if format!("{:x}", digest.finalize()) != config.model_sha256 {
        bail!("AFx identity model hash does not match its package metadata");
    }
    let model = tract_onnx::onnx()
        .model_for_path(&model_path)
        .context("could not load the AFx identity model")?
        .into_optimized()
        .context("could not optimize the AFx identity model")?
        .into_runnable()
        .context("could not prepare the AFx identity model")?;
    Ok(Some(Runtime {
        model,
        labels: config.display_labels,
        threshold: config.threshold,
    }))
}

#[cfg(not(feature = "embedded-identity"))]
fn model_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("MUSPECTOR_AFX_IDENTITY_ONNX") {
        return Some(PathBuf::from(path));
    }
    #[cfg(target_os = "macos")]
    let root = env::var_os("HOME").map(PathBuf::from).map(|home| {
        home.join("Library")
            .join("Application Support")
            .join("Muspector")
            .join("models")
    });
    #[cfg(target_os = "windows")]
    let root = env::var_os("APPDATA")
        .map(PathBuf::from)
        .map(|path| path.join("Muspector").join("models"));
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let root = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        })
        .map(|path| path.join("muspector").join("models"));
    root.map(|path| path.join("afx-pedal-identity.onnx"))
}

fn resample(samples: &[f32], rate: u32) -> Result<Vec<f32>> {
    if rate == RATE {
        return Ok(samples.to_vec());
    }
    let input = InterleavedOwned::new_from(samples.to_vec(), 1, samples.len())
        .context("could not prepare audio for AFx identity resampling")?;
    let mut resampler = Fft::<f32>::new(rate as usize, RATE as usize, 2_048, 1, FixedSync::Both)
        .context("could not create the AFx identity resampler")?;
    let output = resampler
        .process_all(&input, samples.len(), None)
        .context("could not resample audio for AFx identity")?;
    Ok(output.take_data())
}

fn energetic_windows(samples: &[f32]) -> Vec<usize> {
    if samples.len() <= WINDOW {
        return vec![0];
    }
    let mut starts = (0..=samples.len() - WINDOW)
        .step_by(WINDOW / 2)
        .collect::<Vec<_>>();
    let final_start = samples.len() - WINDOW;
    if starts.last().copied() != Some(final_start) {
        starts.push(final_start);
    }
    starts.sort_by(|left, right| {
        energy(samples, *right)
            .total_cmp(&energy(samples, *left))
            .then_with(|| left.cmp(right))
    });
    starts.truncate(WINDOWS);
    starts.sort_unstable();
    starts
}

fn energy(samples: &[f32], start: usize) -> f64 {
    samples[start..start + WINDOW]
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>()
        / WINDOW as f64
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|value| f64::from(*value - maximum).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f64>();
    for value in &mut values {
        *value /= total;
    }
    values
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energetic_selection_is_bounded_and_prefers_signal() {
        let mut samples = vec![0.0; WINDOW * 3];
        samples[WINDOW * 2..].fill(0.5);
        let starts = energetic_windows(&samples);
        assert!(starts.len() <= WINDOWS);
        assert!(starts.contains(&(WINDOW * 2)));
    }

    #[test]
    fn softmax_is_normalized() {
        let values = softmax(&[1.0, 2.0, 3.0]);
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1.0e-9);
        assert_eq!(
            values.iter().copied().max_by(f64::total_cmp),
            Some(values[2])
        );
    }
}
