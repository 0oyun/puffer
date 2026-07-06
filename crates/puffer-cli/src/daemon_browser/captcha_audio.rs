//! Autonomous reCAPTCHA v2 AUDIO-challenge solving with a local whisper model
//! (feature `captcha-audio`). No paid solver, no network ASR: the challenge mp3
//! is decoded with symphonia and transcribed with whisper.cpp (via whisper-rs).
//!
//! Two entry points:
//!  - [`ensure_model`] lazily downloads the ggml `small.en` model (~150 MB) to
//!    `~/.puffer/models/whisper` on first use (mirrors the LocalModelInstaller
//!    download pattern), so a puffer install carries no model until it hits a
//!    captcha.
//!  - [`transcribe`] turns the mp3 bytes into the lowercase phrase reCAPTCHA
//!    expects in `#audio-response`.

use anyhow::{anyhow, Context, Result};
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::time::Duration;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// English-only ggml model for the short spoken phrases reCAPTCHA uses. Defaults
/// to `base.en` (~142 MB): good accuracy, fast on CPU, light enough for a lazy
/// first-use download. Override via env for more accuracy on noisy audio:
///   PUFFER_WHISPER_MODEL_FILE=ggml-small.en.bin  (~466 MB)
///   PUFFER_WHISPER_MODEL_URL=<full https url>
// small.en is markedly more accurate than base.en on reCAPTCHA's short spoken
// phrases (still fast on CPU); override with PUFFER_WHISPER_MODEL_FILE/_URL.
const DEFAULT_MODEL_FILE: &str = "ggml-small.en.bin";
const MODEL_MIN_BYTES: u64 = 50_000_000;

fn model_file() -> String {
    std::env::var("PUFFER_WHISPER_MODEL_FILE").unwrap_or_else(|_| DEFAULT_MODEL_FILE.to_string())
}

fn model_url() -> String {
    std::env::var("PUFFER_WHISPER_MODEL_URL").unwrap_or_else(|_| {
        format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}", model_file())
    })
}

/// `~/.puffer/models/whisper` (honours `PUFFER_HOME`).
fn model_dir() -> PathBuf {
    let base = std::env::var_os("PUFFER_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".puffer")))
        .unwrap_or_else(|| PathBuf::from(".puffer"));
    base.join("models").join("whisper")
}

/// Lazily fetch the ggml model on first use; returns its path. Idempotent: a
/// fully-downloaded model short-circuits. Downloads to a `.part` file and renames
/// atomically so an interrupted download never leaves a half model in place.
pub(crate) fn ensure_model() -> Result<PathBuf> {
    let dir = model_dir();
    let file = model_file();
    let url = model_url();
    let path = dir.join(&file);
    if std::fs::metadata(&path)
        .map(|m| m.len() >= MODEL_MIN_BYTES)
        .unwrap_or(false)
    {
        return Ok(path);
    }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create whisper model dir {}", dir.display()))?;
    tracing::info!(target: "puffer::captcha", %url, "downloading whisper model (first use)");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(900))
        .build()
        .context("build http client")?;
    let mut resp = client.get(&url).send().context("request whisper model")?;
    if !resp.status().is_success() {
        return Err(anyhow!("whisper model download HTTP {}", resp.status()));
    }
    let tmp = dir.join(format!("{file}.part"));
    {
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("create {}", tmp.display()))?;
        std::io::copy(&mut resp, &mut f).context("write whisper model")?;
        f.flush().ok();
    }
    std::fs::rename(&tmp, &path).context("finalize whisper model")?;
    Ok(path)
}

/// Transcribe reCAPTCHA challenge mp3 bytes into the lowercase phrase to type.
pub(crate) fn transcribe(mp3: &[u8]) -> Result<String> {
    let model = ensure_model()?;
    let samples = decode_mp3_to_16k_mono(mp3)?;
    if samples.is_empty() {
        return Err(anyhow!("decoded audio was empty"));
    }
    let model_str = model
        .to_str()
        .ok_or_else(|| anyhow!("model path not utf-8"))?;
    let ctx = WhisperContext::new_with_params(model_str, WhisperContextParameters::default())
        .context("load whisper model")?;
    let mut state = ctx.create_state().context("create whisper state")?;
    // Beam search (vs greedy best_of:1) is more accurate on the short, noisy
    // reCAPTCHA audio clips — worth the small extra CPU for a higher pass rate.
    let mut params = FullParams::new(SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 });
    params.set_language(Some("en"));
    params.set_n_threads(4);
    params.set_no_context(true); // each clip is independent — don't carry prior text
    params.set_suppress_blank(true);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    state
        .full(params, &samples)
        .context("whisper transcription failed")?;
    let n = state.full_n_segments().context("count whisper segments")?;
    let mut out = String::new();
    for i in 0..n {
        if let Ok(text) = state.full_get_segment_text(i) {
            out.push_str(&text);
            out.push(' ');
        }
    }
    Ok(normalize_answer(&out))
}

/// reCAPTCHA audio answers are lowercase words; strip punctuation and collapse
/// whitespace so the typed response matches what the grader expects.
fn normalize_answer(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Decode mp3 bytes to 16 kHz mono f32 PCM (whisper's expected input).
fn decode_mp3_to_16k_mono(mp3: &[u8]) -> Result<Vec<f32>> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(mp3.to_vec())), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probe mp3 container")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("mp3 has no audio track"))?;
    let track_id = track.id;
    let src_rate = track.codec_params.sample_rate.unwrap_or(16_000);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("build mp3 decoder")?;
    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break, // end of stream
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(_) => break,
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks(channels) {
            let sum: f32 = frame.iter().copied().sum();
            mono.push(sum / channels as f32);
        }
    }
    if src_rate == 16_000 {
        Ok(mono)
    } else {
        Ok(resample_linear(&mono, src_rate, 16_000))
    }
}

/// Cheap linear resampler — the challenge clip is a few seconds of speech, so
/// linear interpolation is more than enough for whisper to transcribe.
fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if input.is_empty() || from == 0 || to == 0 {
        return Vec::new();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}
