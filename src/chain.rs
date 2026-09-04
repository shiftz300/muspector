#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gate,
    Comp,
    Drive,
    Eq,
    Delay,
    Reverb,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gate => "Gate",
            Self::Comp => "Comp",
            Self::Drive => "Drive",
            Self::Eq => "EQ",
            Self::Delay => "Delay",
            Self::Reverb => "Reverb",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub unit: &'static str,
}

impl Param {
    pub(crate) fn new(
        name: &'static str,
        value: f64,
        min: f64,
        max: f64,
        step: f64,
        unit: &'static str,
    ) -> Self {
        Self {
            name,
            value: value.clamp(min, max),
            min,
            max,
            step,
            unit,
        }
    }

    pub fn normal(&self) -> f32 {
        ((self.value - self.min) / (self.max - self.min)).clamp(0.0, 1.0) as f32
    }

    pub fn text(&self) -> String {
        match self.unit {
            "dB" | "s" => format!("{:.1} {}", self.value, self.unit),
            ":1" => format!("{:.1}:1", self.value),
            "" => format!("{:.1}", self.value),
            _ => format!("{:.0} {}", self.value, self.unit),
        }
    }

    pub fn input(&self) -> String {
        if self.step >= 1.0 {
            format!("{:.0}", self.value)
        } else if self.step >= 0.1 {
            format!("{:.1}", self.value)
        } else {
            format!("{:.2}", self.value)
        }
    }

    pub fn shift(&mut self, direction: f64) {
        self.set(self.value + self.step * direction);
    }

    pub fn set(&mut self, value: f64) {
        self.value = value.clamp(self.min, self.max);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Effect {
    pub kind: Kind,
    pub model: Option<String>,
    pub active: bool,
    pub score: f64,
    pub evidence: String,
    pub params: Vec<Param>,
}

impl Effect {
    pub fn name(&self) -> &str {
        self.model.as_deref().unwrap_or_else(|| self.kind.name())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    pub effects: Vec<Effect>,
    pub score: f64,
}

impl Chain {
    pub fn active(&self) -> impl Iterator<Item = &Effect> {
        self.effects.iter().filter(|effect| effect.active)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Fingerprint {
    pub peak: f64,
    pub crest: f64,
    pub range: f64,
    pub floor: f64,
    pub silence: f64,
    pub transient: f64,
    pub flatness: f64,
    pub low: f64,
    pub mid: f64,
    pub high: f64,
    pub echo: f64,
    pub echo_ms: f64,
    pub tail: f64,
}

pub fn infer(f: Fingerprint) -> Chain {
    let gate = scale(f.silence, 0.03, 0.30) * 0.65 + scale(-f.floor, 38.0, 68.0) * 0.35;
    let comp = scale(18.0 - f.crest, 0.0, 11.0) * 0.58 + scale(28.0 - f.range, 0.0, 20.0) * 0.42;
    let drive = scale(f.flatness, 0.015, 0.24) * 0.58
        + scale(f.high, -24.0, -9.0) * 0.27
        + scale(f.peak, -8.0, -0.2) * 0.15;
    let delay = scale(f.echo, 0.12, 0.55);
    let reverb = scale(f.tail, 0.06, 0.30) * 0.72 + scale(f.echo, 0.08, 0.35) * 0.28;

    let effects = vec![
        Effect {
            kind: Kind::Gate,
            model: None,
            active: gate >= 0.55,
            score: gate,
            evidence: format!(
                "{:.0}% low-level frames · floor {:.0} dB",
                f.silence * 100.0,
                f.floor
            ),
            params: vec![Param::new(
                "Threshold",
                f.floor + 7.0,
                -72.0,
                -18.0,
                1.0,
                "dB",
            )],
        },
        Effect {
            kind: Kind::Comp,
            model: None,
            active: comp >= 0.48,
            score: comp,
            evidence: format!("Crest {:.1} dB · range {:.1} dB", f.crest, f.range),
            params: vec![
                Param::new("Ratio", 1.0 + comp * 7.0, 1.0, 10.0, 0.5, ":1"),
                Param::new(
                    "Attack",
                    42.0 - f.transient.clamp(0.0, 1.0) * 34.0,
                    1.0,
                    80.0,
                    1.0,
                    "ms",
                ),
                Param::new("Release", 80.0 + comp * 260.0, 40.0, 600.0, 10.0, "ms"),
            ],
        },
        Effect {
            kind: Kind::Drive,
            model: None,
            active: drive >= 0.55,
            score: drive,
            evidence: format!(
                "Spectral flatness {:.2} · high band {:.1} dB",
                f.flatness, f.high
            ),
            params: vec![
                Param::new("Gain", drive * 24.0, 0.0, 30.0, 0.5, "dB"),
                Param::new(
                    "Tone",
                    50.0 + (f.high - f.low).clamp(-20.0, 20.0) * 1.8,
                    0.0,
                    100.0,
                    2.0,
                    "%",
                ),
                Param::new("Level", 0.0, -18.0, 12.0, 0.5, "dB"),
            ],
        },
        Effect {
            kind: Kind::Eq,
            model: None,
            active: true,
            score: 0.58,
            evidence: format!(
                "Band balance {:.1} / {:.1} / {:.1} dB",
                f.low, f.mid, f.high
            ),
            params: vec![
                Param::new("Low", (f.low + 10.0) * 0.55, -12.0, 12.0, 0.5, "dB"),
                Param::new("Mid", (f.mid + 2.0) * 0.55, -12.0, 12.0, 0.5, "dB"),
                Param::new("High", (f.high + 15.0) * 0.55, -12.0, 12.0, 0.5, "dB"),
            ],
        },
        Effect {
            kind: Kind::Delay,
            model: None,
            active: delay >= 0.52,
            score: delay,
            evidence: format!("Envelope echo {:.2} near {:.0} ms", f.echo, f.echo_ms),
            params: vec![
                Param::new("Time", f.echo_ms, 40.0, 1_000.0, 5.0, "ms"),
                Param::new("Feedback", 12.0 + delay * 58.0, 0.0, 90.0, 2.0, "%"),
                Param::new("Mix", 5.0 + delay * 35.0, 0.0, 70.0, 2.0, "%"),
            ],
        },
        Effect {
            kind: Kind::Reverb,
            model: None,
            active: reverb >= 0.45,
            score: reverb,
            evidence: format!("Diffuse envelope persistence {:.2}", f.tail),
            params: vec![
                Param::new("Decay", 0.4 + reverb * 4.6, 0.2, 8.0, 0.1, "s"),
                Param::new(
                    "Damp",
                    62.0 - f.high.clamp(-30.0, 0.0),
                    0.0,
                    100.0,
                    2.0,
                    "%",
                ),
                Param::new("Mix", 6.0 + reverb * 38.0, 0.0, 70.0, 2.0, "%"),
            ],
        },
    ];

    let active: Vec<_> = effects.iter().filter(|effect| effect.active).collect();
    let score = if active.is_empty() {
        0.0
    } else {
        (active.iter().map(|effect| effect.score).sum::<f64>() / active.len() as f64).min(0.78)
    };
    Chain { effects, score }
}

fn scale(value: f64, low: f64, high: f64) -> f64 {
    ((value - low) / (high - low)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_is_bounded() {
        let chain = infer(Fingerprint {
            peak: -0.2,
            crest: 8.0,
            range: 10.0,
            floor: -64.0,
            silence: 0.25,
            transient: 0.6,
            flatness: 0.18,
            low: -8.0,
            mid: -2.0,
            high: -13.0,
            echo: 0.4,
            echo_ms: 375.0,
            tail: 0.2,
        });
        assert_eq!(chain.effects.len(), 6);
        assert!(chain.score <= 0.78);
        assert!(
            chain
                .effects
                .iter()
                .flat_map(|effect| &effect.params)
                .all(|param| { param.value >= param.min && param.value <= param.max })
        );
    }

    #[test]
    fn parameter_shift_clamps() {
        let mut param = Param::new("Mix", 99.0, 0.0, 100.0, 2.0, "%");
        param.shift(10.0);
        assert_eq!(param.value, 100.0);
        param.shift(-100.0);
        assert_eq!(param.value, 0.0);
        param.set(500.0);
        assert_eq!(param.value, 100.0);
    }
}
