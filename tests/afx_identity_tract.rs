//! Compatibility check for the optional, non-commercial AFx identity package.
//!
//! The model is intentionally not stored in the repository. Run with:
//! `MUSPECTOR_AFX_IDENTITY_ONNX=/path/to/model.onnx cargo test --test afx_identity_tract -- --ignored`

use std::{env, time::Instant};

use tract_onnx::prelude::*;

#[test]
#[ignore = "requires the local non-commercial AFx identity model"]
fn exported_identity_model_runs_in_tract() -> TractResult<()> {
    let path = env::var("MUSPECTOR_AFX_IDENTITY_ONNX")
        .expect("MUSPECTOR_AFX_IDENTITY_ONNX must name the exported model");
    let started = Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .into_optimized()?
        .into_runnable()?;
    eprintln!(
        "tract_identity_load_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    let input = Tensor::zero::<f32>(&[1, 1, 240_000])?;
    let started = Instant::now();
    let output = model.run(tvec!(input.into()))?;
    eprintln!(
        "tract_identity_inference_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(output[0].len(), 7);
    assert_eq!(output[1].len(), 1);
    Ok(())
}
