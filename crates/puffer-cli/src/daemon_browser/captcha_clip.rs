//! reCAPTCHA v2 IMAGE-challenge solving with a native **CLIP open-vocabulary**
//! tile classifier (feature `captcha-image`). Unlike the fixed-class YOLO path
//! in `captcha_image.rs`, CLIP scores each grid tile against an arbitrary text
//! prompt, so it covers the non-COCO scene classes reCAPTCHA loves — crosswalk,
//! stairs, bridge, mountain, chimney, palm tree, taxi — that a fixed detector
//! can never name. All native (ort + image + tokenizers), no Python.
//!
//! Method (per the validated `recognizer` approach): per tile, CLIP image embed;
//! softmax over {target prompt(s), every other class prompt, scene negatives}
//! scaled by logit_scale≈100; select a tile iff the target's probability mass
//! clears a per-class threshold AND the target is the arg-max object class for
//! that tile (the dual gate kills the "one wrong tile fails the whole challenge"
//! problem). Models + tokenizer are lazily downloaded and cached on first use,
//! exactly like the whisper and YOLO weights.

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

const HF_BASE: &str = "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main";
const CLIP_SIZE: u32 = 224;
// Standard CLIP preprocessing (from preprocessor_config.json).
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
// Learned/clamped logit_scale.exp() for ViT-B/16 — keeps the softmax peaky.
const LOGIT_SCALE: f32 = 100.0;
const BOS: i64 = 49406;
const EOS: i64 = 49407;
const CTX: usize = 77;

/// Known reCAPTCHA classes: (canonical name, positive prompt(s), softmax
/// threshold). Simple object classes use a bare "a photo of a X."; the ambiguous
/// scene/texture classes use a rich descriptive prompt (a one-word label is too
/// weak). Thresholds seed from `recognizer`'s calibrated values; tune via env.
const CLASSES: &[(&str, &[&str], f32)] = &[
    ("bicycle", &["a photo of a bicycle."], 0.50),
    ("boat", &["a photo of a boat."], 0.50),
    ("bus", &["a photo of a bus."], 0.50),
    ("car", &["a photo of a car."], 0.50),
    ("motorcycle", &["a photo of a motorcycle."], 0.50),
    ("fire hydrant", &["a photo of a fire hydrant."], 0.50),
    ("traffic light", &["a photo of a traffic light."], 0.50),
    ("taxi", &["a photo of a taxi.", "a photo of a yellow car."], 0.80),
    (
        "bridge",
        &["the front or bottom or side of a concrete or steel bridge supported by concrete pillars over a street or highway"],
        0.73,
    ),
    (
        "chimney",
        &["a close-up of a chimney on a house, with rooftops and ceiling below"],
        0.79,
    ),
    (
        "crosswalk",
        &["striped pedestrian crossing with white or yellow markings of a crosswalk stretching over a gray street"],
        0.89,
    ),
    (
        "mountain",
        &["a green or grey landscape with trees or a road connecting two mountain slopes or hills"],
        0.56,
    ),
    (
        "palm tree",
        &["a feather-like palm tree growing behind a tiled rooftop, with a road or street"],
        0.81,
    ),
    (
        "stairs",
        &["a stairway for pedestrians in front of a house or building leading to a walkway"],
        0.91,
    ),
    (
        "tractor",
        &["a tractor or agricultural vehicle on a street or field"],
        0.94,
    ),
];

/// Scene-background negatives added to every prompt set — distractor tiles are
/// usually building/road/sky/foliage fragments, so concrete scene negatives beat
/// a lone catch-all.
const NEGATIVES: &[&str] = &[
    "an empty street",
    "a house wall",
    "the sky",
    "trees",
    "a road",
    "a blank gray image",
    "a photo of something else",
];

fn clip_dir() -> PathBuf {
    let base = std::env::var_os("PUFFER_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".puffer")))
        .unwrap_or_else(|| PathBuf::from(".puffer"));
    base.join("models").join("clip")
}

/// Lazily download a CLIP artifact (vision/text onnx, tokenizer) to the cache.
fn ensure_file(file: &str, url_suffix: &str, min_bytes: u64, env_url: &str) -> Result<PathBuf> {
    let dir = clip_dir();
    let path = dir.join(file);
    if std::fs::metadata(&path).map(|m| m.len() >= min_bytes).unwrap_or(false) {
        return Ok(path);
    }
    let url = std::env::var(env_url).unwrap_or_else(|_| format!("{HF_BASE}/{url_suffix}"));
    std::fs::create_dir_all(&dir)?;
    tracing::info!(target: "puffer::captcha", %url, "downloading CLIP artifact (first image-challenge use)");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(1800))
        .build()?;
    let mut resp = client.get(&url).send().with_context(|| format!("request {file}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("CLIP {file} download HTTP {}", resp.status()));
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

fn build_session(path: &PathBuf) -> Result<Session> {
    // Mirror captcha_image's session build exactly — the tuning-builder methods
    // (with_optimization_level/with_intra_threads) surface an ort error type that
    // isn't Send+Sync and won't convert through `?` into anyhow here.
    Session::builder()?
        .commit_from_file(path)
        .with_context(|| format!("load CLIP onnx {}", path.display()))
}

fn vision_session() -> Result<&'static Mutex<Session>> {
    static S: OnceLock<Mutex<Session>> = OnceLock::new();
    if let Some(s) = S.get() {
        return Ok(s);
    }
    let path = ensure_file(
        "vision_model.onnx",
        "onnx/vision_model.onnx",
        100_000_000,
        "PUFFER_CLIP_VISION_URL",
    )?;
    let sess = build_session(&path)?;
    Ok(S.get_or_init(|| Mutex::new(sess)))
}

fn text_session() -> Result<&'static Mutex<Session>> {
    static S: OnceLock<Mutex<Session>> = OnceLock::new();
    if let Some(s) = S.get() {
        return Ok(s);
    }
    let path = ensure_file(
        "text_model.onnx",
        "onnx/text_model.onnx",
        100_000_000,
        "PUFFER_CLIP_TEXT_URL",
    )?;
    let sess = build_session(&path)?;
    Ok(S.get_or_init(|| Mutex::new(sess)))
}

fn tokenizer() -> Result<&'static Tokenizer> {
    static T: OnceLock<Tokenizer> = OnceLock::new();
    if let Some(t) = T.get() {
        return Ok(t);
    }
    let path = ensure_file("tokenizer.json", "tokenizer.json", 1_000_000, "PUFFER_CLIP_TOKENIZER_URL")?;
    let tok = Tokenizer::from_file(&path).map_err(|e| anyhow!("load CLIP tokenizer: {e}"))?;
    Ok(T.get_or_init(|| tok))
}

/// Tokenize a prompt to (input_ids, attention_mask) as i64, length exactly 77.
/// Robust to a tokenizer.json with or without the BOS/EOS post-processor.
pub(crate) fn encode_prompt(prompt: &str) -> Result<(Vec<i64>, Vec<i64>)> {
    let tok = tokenizer()?;
    let enc = tok.encode(prompt, true).map_err(|e| anyhow!("tokenize: {e}"))?;
    let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    // Guarantee BOS/EOS even if the file lacks the TemplateProcessing post-proc.
    if ids.first() != Some(&BOS) {
        ids.insert(0, BOS);
    }
    if ids.last() != Some(&EOS) {
        ids.push(EOS);
    }
    ids.truncate(CTX);
    if ids.last() != Some(&EOS) {
        // Truncation cut the EOS — restore it at the final slot.
        let last = CTX - 1;
        if ids.len() > last {
            ids[last] = EOS;
        }
    }
    let real = ids.len();
    let mut mask: Vec<i64> = vec![1; real];
    if real < CTX {
        ids.resize(CTX, EOS);
        mask.resize(CTX, 0);
    }
    Ok((ids, mask))
}

/// L2-normalized 512-d text embedding for a prompt, cached across tiles/calls.
fn embed_text(prompt: &str) -> Result<Vec<f32>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Vec<f32>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = cache.lock().unwrap().get(prompt) {
        return Ok(v.clone());
    }
    let (ids, _mask) = encode_prompt(prompt)?;
    let sess = text_session()?;
    // This Xenova export declares ONLY `input_ids` (no attention_mask input) —
    // passing an undeclared input errors "Invalid input name". CLIP's text tower
    // uses causal attention + EOS-position pooling, so the mask is irrelevant to
    // the pooled embedding anyway (padding is after EOS).
    let it = Tensor::from_array(([1usize, CTX], ids))?;
    let mut emb = {
        let mut guard = sess.lock().unwrap();
        let outputs = guard.run(ort::inputs!["input_ids" => it])?;
        let (_s, data) = outputs["text_embeds"].try_extract_tensor::<f32>()?;
        data.to_vec()
    };
    l2_normalize(&mut emb);
    cache.lock().unwrap().insert(prompt.to_string(), emb.clone());
    Ok(emb)
}

/// L2-normalized 512-d image embedding for one already-cropped RGB tile.
fn embed_image(tile: &image::RgbImage) -> Result<Vec<f32>> {
    let pixels = preprocess(tile);
    let pt = Tensor::from_array(([1usize, 3, CLIP_SIZE as usize, CLIP_SIZE as usize], pixels))?;
    let sess = vision_session()?;
    let mut emb = {
        let mut guard = sess.lock().unwrap();
        let outputs = guard.run(ort::inputs!["pixel_values" => pt])?;
        let (_s, data) = outputs["image_embeds"].try_extract_tensor::<f32>()?;
        data.to_vec()
    };
    l2_normalize(&mut emb);
    Ok(emb)
}

/// CLIP preprocessing: RGB → resize shortest edge to 224 (bicubic ≈ CatmullRom)
/// → center-crop 224 → /255 → per-channel normalize → NCHW f32.
fn preprocess(img: &image::RgbImage) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let (nw, nh) = if w < h {
        (CLIP_SIZE, ((h as f32) * (CLIP_SIZE as f32 / w as f32)).round() as u32)
    } else {
        (((w as f32) * (CLIP_SIZE as f32 / h as f32)).round() as u32, CLIP_SIZE)
    };
    let resized = image::imageops::resize(img, nw.max(CLIP_SIZE), nh.max(CLIP_SIZE), FilterType::CatmullRom);
    let (ox, oy) = ((resized.width() - CLIP_SIZE) / 2, (resized.height() - CLIP_SIZE) / 2);
    let plane = (CLIP_SIZE * CLIP_SIZE) as usize;
    let mut out = vec![0f32; 3 * plane];
    for y in 0..CLIP_SIZE {
        for x in 0..CLIP_SIZE {
            let px = resized.get_pixel(ox + x, oy + y);
            let idx = (y * CLIP_SIZE + x) as usize;
            for c in 0..3 {
                out[c * plane + idx] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    out
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum::<f32>().max(1e-12);
    e.iter().map(|v| v / s).collect()
}

/// Minimal homoglyph fold — reCAPTCHA injects Cyrillic/Greek lookalikes into the
/// instruction text to defeat literal matching. Map the common ones back to ASCII.
fn normalize_label(task: &str) -> String {
    let mut s = String::with_capacity(task.len());
    for ch in task.chars() {
        let a = match ch {
            'а' => 'a', 'е' => 'e', 'о' => 'o', 'р' => 'p', 'с' => 'c', 'у' => 'y',
            'х' => 'x', 'і' => 'i', 'ѕ' => 's', 'ԁ' => 'd', 'ո' => 'n', 'г' => 'r',
            'Α' => 'a', 'Ο' => 'o', 'Ρ' => 'p', 'Τ' => 't', 'Β' => 'b', 'Ε' => 'e',
            other => other.to_ascii_lowercase(),
        };
        s.push(a);
    }
    s.trim().to_string()
}

/// Resolve the challenge instruction to a canonical class name (open-vocab: an
/// unknown label falls back to itself so CLIP still classifies "a photo of a X").
pub(crate) fn resolve_class(task: &str) -> String {
    let t = normalize_label(task);
    // Order matters (first match wins): the specific "…车" compounds (taxi/bus/
    // truck/motorcycle/bicycle) must be checked BEFORE the generic car, so e.g.
    // 公共汽车(bus) / 出租车(taxi) don't fall into car (which contains 汽车/车).
    // reCAPTCHA localizes the task word to the browser locale (a CN session gets
    // 小轿车, not "car"); CLIP's text encoder is English-only, so we map every
    // localized synonym back to the English canonical → English CLIP prompt.
    let rules: &[(&str, &[&str])] = &[
        ("taxi", &["taxi", "cab", "出租车", "出租車", "的士", "计程车", "計程車"]),
        ("bus", &["bus", "公交车", "公共汽车", "公交車", "巴士", "大巴", "公車", "バス", "버스"]),
        ("truck", &["truck", "lorry", "卡车", "货车", "貨車", "トラック", "트럭"]),
        ("motorcycle", &["motorcycle", "motorbike", "摩托车", "摩托車", "机车", "機車", "オートバイ", "오토바이"]),
        ("bicycle", &["bicycle", "bike", "自行车", "自行車", "单车", "單車", "脚踏车", "腳踏車", "自転車", "자전거"]),
        ("car", &["car", "vehicle", "轿车", "轎车", "轎車", "小汽车", "汽车", "汽車", "小客车", "自動車", "자동차", "승용차"]),
        ("fire hydrant", &["fire hydrant", "hydrant", "消防栓", "消火栓", "消防龙头", "消火栓", "소화전"]),
        ("traffic light", &["traffic light", "traffic signal", "红绿灯", "紅綠燈", "交通灯", "交通燈", "信号灯", "信號燈", "交通信号", "신호등", "信号機"]),
        ("boat", &["boat", "ship", "船", "小船", "船只", "船隻", "ボート", "보트"]),
        ("bridge", &["bridge", "桥", "橋", "大桥", "大橋", "橋梁", "다리"]),
        ("chimney", &["chimney", "烟囱", "煙囪", "烟囪", "煙突", "굴뚝"]),
        ("crosswalk", &["crosswalk", "cross walk", "人行横道", "人行橫道", "斑马线", "斑馬線", "人行道", "횡단보도"]),
        ("mountain", &["mountain", "hill", "山", "山丘", "山脉", "山脈", "丘陵", "산"]),
        ("palm tree", &["palm", "棕榈", "棕櫚", "棕榈树", "棕櫚樹", "椰子树", "椰子樹", "야자"]),
        ("stairs", &["stair", "steps", "楼梯", "樓梯", "台阶", "臺階", "階梯", "계단"]),
        ("tractor", &["tractor", "拖拉机", "拖拉機", "トラクター", "트랙터"]),
        ("parking meter", &["parking meter", "停车计时器", "停車計時器", "停车收费表", "泊车计时器"]),
    ];
    for (canon, needles) in rules {
        if needles.iter().any(|n| t.contains(n)) {
            return canon.to_string();
        }
    }
    // Unknown → strip an article/plural and use as its own open-vocab label.
    let mut w = t
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim_start_matches("the ")
        .trim()
        .to_string();
    if let Some(stripped) = w.strip_suffix("es") {
        w = stripped.to_string();
    } else if let Some(stripped) = w.strip_suffix('s') {
        w = stripped.to_string();
    }
    if w.is_empty() {
        w = t;
    }
    w
}

/// Look up a class's positive prompts + threshold; open-vocab fallback for
/// unknown labels ("a photo of a {label}." at a neutral threshold).
fn class_prompts(canon: &str) -> (Vec<String>, f32) {
    for (name, prompts, thr) in CLASSES {
        if *name == canon {
            return (prompts.iter().map(|s| s.to_string()).collect(), *thr);
        }
    }
    (vec![format!("a photo of a {canon}.")], 0.50)
}

/// Open-vocab: we can attempt any resolvable label.
pub(crate) fn can_handle(task: &str) -> bool {
    !resolve_class(task).is_empty()
}

/// Classify which of the `rows`x`rows` grid tiles satisfy the task, via CLIP
/// open-vocabulary per-tile scoring. `grid_png` is a screenshot of just the grid.
/// Returns row-major 0-based tile indices (i = r*rows + c).
pub(crate) fn classify_tiles(grid_png: &[u8], task: &str, rows: usize) -> Result<Vec<usize>> {
    let target = resolve_class(task);
    let (target_prompts, _base_thr) = class_prompts(&target);
    // CLIP softmax (logit_scale 100) is peaky: a tile that truly contains the
    // target lands ~0.85-1.0, while shadow/partial/background false positives sit
    // ~0.3-0.6. A single high cutoff + the arg-max-object gate gives the precision
    // reCAPTCHA demands (one wrong tile fails the whole grid → endless reloads →
    // bot flag). Empirically ~0.62 cleanly separates the two bands; env-tunable.
    let thr: f32 = std::env::var("PUFFER_CLIP_THRESH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.62);
    let debug = std::env::var_os("PUFFER_CLIP_DEBUG").is_some();

    // Build the object-class prompt universe: the target's positive prompt(s)
    // plus one representative prompt for every OTHER known class (mutual
    // competitors) — this is what makes the softmax discriminative.
    let mut prompts: Vec<String> = Vec::new();
    let mut prompt_class: Vec<Option<String>> = Vec::new(); // class name for object prompts, None for negatives
    for p in &target_prompts {
        prompts.push(p.clone());
        prompt_class.push(Some(target.clone()));
    }
    for (name, cps, _) in CLASSES {
        if *name == target {
            continue;
        }
        prompts.push(cps[0].to_string());
        prompt_class.push(Some((*name).to_string()));
    }
    let n_obj = prompts.len();
    for n in NEGATIVES {
        prompts.push((*n).to_string());
        prompt_class.push(None);
    }

    // Embed all prompts once (cached across tiles).
    let mut text_embeds: Vec<Vec<f32>> = Vec::with_capacity(prompts.len());
    for p in &prompts {
        text_embeds.push(embed_text(p)?);
    }

    // Decode the grid and slice into equal cells.
    let img = image::load_from_memory(grid_png).context("decode grid image")?.to_rgb8();
    let (gw, gh) = (img.width(), img.height());
    let cell_w = gw / rows as u32;
    let cell_h = gh / rows as u32;
    if cell_w < 8 || cell_h < 8 {
        return Ok(Vec::new());
    }

    // Calibration aid: when PUFFER_CLIP_DEBUG is set, dump the grid PNG + a
    // per-tile score table to ~/.puffer/clip_debug/ so thresholds can be tuned
    // from real challenges without wrestling the daemon's log routing.
    let mut dbg_lines: Vec<String> = Vec::new();
    if debug {
        if let Some(dir) = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".puffer").join("clip_debug")) {
            let _ = std::fs::create_dir_all(&dir);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let _ = std::fs::write(dir.join(format!("grid_{target}_{stamp}.png")), grid_png);
            dbg_lines.push(format!("# task='{task}' target='{target}' thr={thr:.3} rows={rows}"));
        }
    }

    let mut selected = Vec::new();
    for r in 0..rows {
        for c in 0..rows {
            let tile = image::imageops::crop_imm(
                &img,
                c as u32 * cell_w,
                r as u32 * cell_h,
                cell_w,
                cell_h,
            )
            .to_image();
            let img_embed = embed_image(&tile)?;

            // logits = scale * cos(img, txt); softmax over the full prompt set.
            let logits: Vec<f32> = text_embeds.iter().map(|t| LOGIT_SCALE * dot(&img_embed, t)).collect();
            let probs = softmax(&logits);

            // Target probability mass (sum over the target's positive prompts).
            let target_score: f32 = prompt_class
                .iter()
                .zip(&probs)
                .filter(|(cl, _)| cl.as_deref() == Some(target.as_str()))
                .map(|(_, p)| *p)
                .sum();

            // Arg-max OBJECT class (ignore negatives): sum prob per class.
            let mut per_class: HashMap<&str, f32> = HashMap::new();
            for (cl, p) in prompt_class[..n_obj].iter().zip(&probs[..n_obj]) {
                if let Some(name) = cl {
                    *per_class.entry(name.as_str()).or_insert(0.0) += *p;
                }
            }
            let best = per_class
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(k, _)| *k)
                .unwrap_or("");

            let i = r * rows + c;
            let hit = target_score >= thr && best == target;
            if debug {
                let best_p = per_class.get(best).copied().unwrap_or(0.0);
                dbg_lines.push(format!(
                    "tile {i:2} target_score={target_score:.3} best={best}({best_p:.3}) hit={hit}"
                ));
            }
            if hit {
                selected.push(i);
            }
        }
    }
    if debug && !dbg_lines.is_empty() {
        if let Some(dir) = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".puffer").join("clip_debug")) {
            let _ = std::fs::create_dir_all(&dir);
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("scores.log")).ok();
            if let Some(f) = f.as_mut() {
                let _ = writeln!(f, "{}\n", dbg_lines.join("\n"));
            }
        }
    }
    let _ = gh;
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Offline sanity for the whole CLIP path (download → preprocess → encode →
    // cosine). Requires the cached models (~/.puffer/models/clip) and /tmp/bus.jpg.
    // Run explicitly: `cargo test -p puffer-cli --features captcha clip_smoke -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn clip_smoke() {
        let png = std::fs::read("/tmp/bus.jpg").expect("bus.jpg");
        let img = image::load_from_memory(&png).unwrap().to_rgb8();
        let ie = embed_image(&img).unwrap();
        let mut best = (f32::MIN, String::new());
        for p in [
            "a photo of a bus.",
            "a photo of a car.",
            "a photo of a cat.",
            "a photo of a fire hydrant.",
            "the sky",
        ] {
            let te = embed_text(p).unwrap();
            let cos = dot(&ie, &te);
            println!("cos({p}) = {cos:.4}");
            if cos > best.0 {
                best = (cos, p.to_string());
            }
        }
        println!("best = {}", best.1);
        // The bus photo must be nearest the bus prompt for the pipeline to be sane.
        assert_eq!(best.1, "a photo of a bus.");
    }
}
