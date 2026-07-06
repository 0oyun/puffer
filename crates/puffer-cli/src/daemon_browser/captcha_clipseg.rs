//! reCAPTCHA v2 **4×4 "area" challenge** solving with CLIPSeg segmentation
//! (feature `captcha-image`). The 4×4 grid is ONE photo sliced into 16 tiles; the
//! target (bridge, bus, crosswalk, motorcycle, …) spans several tiles, so the
//! per-tile CLIP classifier in `captcha_clip.rs` — great for 3×3 where each tile
//! is a whole scene — is weak here (a lone wheel isn't "a motorcycle"). CLIPSeg
//! instead segments the WHOLE grid for the prompt and returns a per-pixel heatmap;
//! we then pick the 88×88 tile blocks the mask covers. Native (ort + image), and
//! it REUSES the already-cached CLIP tokenizer from `captcha_clip`.
//!
//! Two things differ from the CLIP path and are easy to get wrong:
//!   * image normalization is **ImageNet** (ViTImageProcessor), NOT CLIP stats;
//!   * the graph requires all THREE inputs (pixel_values, input_ids, attention_mask).

use anyhow::{anyhow, Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;

const CLIPSEG_URL: &str = "https://huggingface.co/Xenova/clipseg-rd64-refined/resolve/main/onnx/model.onnx";
const MODEL_MIN_BYTES: u64 = 100_000_000;
const SEG: usize = 352;
const CELL: usize = 88; // 352 = 4 × 88
// CLIPSeg's image tower is a ViTImageProcessor → ImageNet mean/std (NOT the CLIP
// image stats used in captcha_clip). Using CLIP stats here silently degrades masks.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn clipseg_dir() -> PathBuf {
    let base = std::env::var_os("PUFFER_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".puffer")))
        .unwrap_or_else(|| PathBuf::from(".puffer"));
    base.join("models").join("clipseg")
}

fn ensure_model() -> Result<PathBuf> {
    let file = std::env::var("PUFFER_CLIPSEG_FILE").unwrap_or_else(|_| "model.onnx".to_string());
    let dir = clipseg_dir();
    let path = dir.join(&file);
    if std::fs::metadata(&path).map(|m| m.len() >= MODEL_MIN_BYTES).unwrap_or(false) {
        return Ok(path);
    }
    let url = std::env::var("PUFFER_CLIPSEG_URL").unwrap_or_else(|_| CLIPSEG_URL.to_string());
    std::fs::create_dir_all(&dir)?;
    tracing::info!(target: "puffer::captcha", %url, "downloading CLIPSeg ONNX (first 4x4-area use)");
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(1800)).build()?;
    let mut resp = client.get(&url).send().context("request CLIPSeg model")?;
    if !resp.status().is_success() {
        return Err(anyhow!("CLIPSeg model download HTTP {}", resp.status()));
    }
    let tmp = dir.join(format!("{file}.part"));
    {
        let mut f = std::fs::File::create(&tmp)?;
        std::io::copy(&mut resp, &mut f)?;
        f.flush().ok();
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

fn session() -> Result<&'static Mutex<Session>> {
    static S: OnceLock<Mutex<Session>> = OnceLock::new();
    if let Some(s) = S.get() {
        return Ok(s);
    }
    let path = ensure_model()?;
    let sess = Session::builder()?
        .commit_from_file(&path)
        .with_context(|| format!("load CLIPSeg onnx {}", path.display()))?;
    Ok(S.get_or_init(|| Mutex::new(sess)))
}

/// CLIPSeg preprocessing: RGB → bilinear resize to 352×352 (squash, no crop) →
/// /255 → ImageNet normalize → NCHW f32 [1,3,352,352].
fn preprocess_352(img: &image::RgbImage) -> Vec<f32> {
    let r = image::imageops::resize(img, SEG as u32, SEG as u32, FilterType::Triangle);
    let plane = SEG * SEG;
    let mut out = vec![0f32; 3 * plane];
    for y in 0..SEG {
        for x in 0..SEG {
            let p = r.get_pixel(x as u32, y as u32);
            let idx = y * SEG + x;
            for c in 0..3 {
                out[c * plane + idx] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    out
}

/// One combined image+text forward → per-pixel foreground probability heatmap
/// (row-major, `probs[y*w + x]`, w=h=352).
fn segment(img: &image::RgbImage, prompt: &str) -> Result<(usize, usize, Vec<f32>)> {
    let (ids, mask) = super::captcha_clip::encode_prompt(prompt)?;
    let l = ids.len();
    let pv = Tensor::from_array(([1usize, 3, SEG, SEG], preprocess_352(img)))?;
    let ii = Tensor::from_array(([1usize, l], ids))?;
    let am = Tensor::from_array(([1usize, l], mask))?;
    let sess = session()?;
    let mut guard = sess.lock().unwrap();
    let outputs = guard.run(ort::inputs![
        "pixel_values" => pv,
        "input_ids" => ii,
        "attention_mask" => am,
    ])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    // logits rank varies by export: [1,352,352], [352,352], or [1,1,352,352].
    // Take the last two dims as (h,w) and slice the leading h*w — identical for all.
    let n = shape.len();
    let w = shape[n - 1] as usize;
    let h = shape[n - 2] as usize;
    if data.len() < h * w || h == 0 || w == 0 {
        return Err(anyhow!("clipseg bad output shape {:?}", shape));
    }
    // sigmoid → [0,1] foreground probability (binary head, NOT softmax).
    let probs: Vec<f32> = data[..h * w].iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect();
    Ok((h, w, probs))
}

/// True if we can turn this task into a segmentation prompt (open-vocab: yes for
/// any resolvable label).
pub(crate) fn can_handle(task: &str) -> bool {
    !super::captcha_clip::resolve_class(task).is_empty()
}

/// Segment the whole 4×4 grid for the task and return the row-major tile indices
/// (i = r*4 + c) the target covers. `grid_png` is a screenshot of just the grid.
pub(crate) fn segment_tiles(grid_png: &[u8], task: &str, rows: usize) -> Result<Vec<usize>> {
    let label = super::captcha_clip::resolve_class(task);
    let prompt = format!("a {label}");
    let img = image::load_from_memory(grid_png).context("decode grid image")?.to_rgb8();
    let (h, w, probs) = segment(&img, &prompt)?;

    // Per-88×88-cell gates (env-tunable for calibration). The mean/frac gates catch
    // tile-filling coverage (bus, crosswalk); the max gate rescues a small object
    // clipping just a corner (traffic light, fire hydrant).
    let envf = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (t, m, f, x) = (
        envf("PUFFER_CLIPSEG_T", 0.40),
        envf("PUFFER_CLIPSEG_M", 0.30),
        envf("PUFFER_CLIPSEG_F", 0.12),
        envf("PUFFER_CLIPSEG_X", 0.65),
    );
    let cw = w / rows;
    let ch = h / rows;
    if cw == 0 || ch == 0 {
        return Ok(Vec::new());
    }
    let debug = std::env::var_os("PUFFER_CLIPSEG_DEBUG").is_some();
    let mut dbg: Vec<String> = Vec::new();
    if debug {
        dbg.push(format!("# task='{task}' prompt='{prompt}' T={t} M={m} F={f} X={x} {w}x{h}"));
        dump_heatmap(&img, w, h, &probs, &label);
    }

    let mut sel = Vec::new();
    for r in 0..rows {
        for c in 0..rows {
            let (mut sum, mut over, mut mx) = (0.0f32, 0usize, 0.0f32);
            let mut cnt = 0usize;
            for yy in 0..ch {
                for xx in 0..cw {
                    let p = probs[(r * ch + yy) * w + (c * cw + xx)];
                    sum += p;
                    if p > t {
                        over += 1;
                    }
                    if p > mx {
                        mx = p;
                    }
                    cnt += 1;
                }
            }
            let mean_p = sum / cnt.max(1) as f32;
            let frac = over as f32 / cnt.max(1) as f32;
            // mean/frac gates catch tile-filling coverage; the max gate rescues a
            // genuinely small object clipping a corner — but require a tiny minimum
            // area (PUFFER_CLIPSEG_XF, default 3%) so a single hot speck of a large
            // object's edge-bleed doesn't grab a neighbouring tile.
            let xf = std::env::var("PUFFER_CLIPSEG_XF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.03f32);
            let hit = mean_p >= m || frac >= f || (mx >= x && frac >= xf);
            if debug {
                dbg.push(format!(
                    "cell {:2} mean={mean_p:.3} frac={frac:.3} max={mx:.3} hit={hit}",
                    r * rows + c
                ));
            }
            if hit {
                sel.push(r * rows + c);
            }
        }
    }
    if debug && !dbg.is_empty() {
        if let Some(dir) = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".puffer").join("clip_debug")) {
            let _ = std::fs::create_dir_all(&dir);
            if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("clipseg_scores.log")) {
                let _ = writeln!(fh, "{}\n", dbg.join("\n"));
            }
        }
    }
    Ok(sel)
}

/// Dump the segmentation heatmap as a grayscale PNG next to the grid capture, so
/// thresholds can be tuned visually from real 4×4 challenges.
fn dump_heatmap(_orig: &image::RgbImage, w: usize, h: usize, probs: &[f32], label: &str) {
    let Some(dir) = std::env::var_os("HOME").map(|hh| PathBuf::from(hh).join(".puffer").join("clip_debug")) else {
        return;
    };
    let _ = std::fs::create_dir_all(&dir);
    let mut gray = image::GrayImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let v = (probs[y * w + x] * 255.0).clamp(0.0, 255.0) as u8;
            gray.put_pixel(x as u32, y as u32, image::Luma([v]));
        }
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let _ = gray.save(dir.join(format!("heat_{label}_{stamp}.png")));
}

#[cfg(test)]
mod tests {
    use super::*;

    // Offline sanity: run CLIPSeg on a saved 4×4 area grid and print per-cell
    // scores. Requires ~/.puffer/models/clipseg/model.onnx + a saved grid.
    // Run: `cargo test -p puffer-cli --features captcha clipseg_smoke -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn clipseg_smoke() {
        let path = std::env::var("CLIPSEG_TEST_IMG")
            .unwrap_or_else(|_| format!("{}/.puffer/clipseg_cal/moto_4x4.png", std::env::var("HOME").unwrap()));
        let task = std::env::var("CLIPSEG_TEST_TASK").unwrap_or_else(|_| "摩托车".to_string());
        std::env::set_var("PUFFER_CLIPSEG_DEBUG", "1");
        let png = std::fs::read(&path).expect("test grid");
        let tiles = segment_tiles(&png, &task, 4).expect("segment");
        println!("selected tiles for '{task}' ({path}): {tiles:?}");
        assert!(!tiles.is_empty(), "expected some tiles for a real area challenge");
    }
}
