//! reCAPTCHA v2 IMAGE-challenge solving with a native ONNX object detector
//! (feature `captcha-image`). Fallback for when the audio challenge is blocked.
//! Runs YOLOv8 (COCO) via `ort` (bundled onnxruntime) to find which grid tiles
//! contain the requested object — no Python. Covers the common object classes
//! reCAPTCHA asks for (bicycle, bus, car, motorcycle, traffic light, fire
//! hydrant, boat/truck). Area classes (crosswalk, stairs, …) need a CLIP pass
//! that is a follow-up; unknown task words return no tiles so the caller retries
//! or reloads rather than mis-clicking.

use anyhow::{anyhow, Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::Mutex;
use std::time::Duration;

use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;

const DEFAULT_MODEL_FILE: &str = "yolov8m.onnx";
// Public COCO YOLOv8m ONNX (input images[1,3,640,640] → output0[1,84,8400]).
// Puffer downloads this on first image-challenge use and caches it — no local
// export/conversion needed. Override with PUFFER_YOLO_MODEL_URL / _FILE.
const DEFAULT_MODEL_URL: &str = "https://huggingface.co/cabelo/yolov8/resolve/main/yolov8m.onnx";
const MODEL_MIN_BYTES: u64 = 20_000_000;
const INPUT: u32 = 640;
const CONF: f32 = 0.25;
const IOU: f32 = 0.45;

/// COCO-80 class names (YOLOv8 default order).
const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// Maps a reCAPTCHA task word (e.g. "bicycles", "a fire hydrant") to the COCO
/// class indices that satisfy it. Returns empty for classes YOLO can't do.
fn task_to_classes(task: &str) -> Vec<usize> {
    let t = task.to_lowercase();
    let mut out = Vec::new();
    let mut push = |name: &str| {
        if let Some(i) = COCO.iter().position(|c| *c == name) {
            out.push(i);
        }
    };
    let has = |needles: &[&str]| needles.iter().any(|n| t.contains(n));
    // Multilingual: reCAPTCHA localizes the challenge word to the browser language
    // (a CN-locale session gets "小轿车", not "car"). Match English plus the common
    // localizations for the 9 COCO-detectable classes. Order matters for the
    // Chinese "…车" family — the more specific compounds (bus/truck/motorcycle)
    // are distinct tokens, and "car" excludes "公共汽车" (bus) which contains 汽车.
    // bicycle
    if has(&["bicycle", "自行车", "脚踏车", "单车", "fahrrad", "vélo", "bicicleta",
             "bicicletta", "自転車", "자전거", "велосипед", "دراجة"]) { push("bicycle"); }
    // motorcycle
    if has(&["motorcycle", "motorbike", "摩托车", "機車", "オートバイ", "オートバイク",
             "오토바이", "motorrad", "moto", "motocicleta", "мотоцикл"]) { push("motorcycle"); }
    // bus
    if has(&["bus", "公交车", "公共汽车", "巴士", "大巴", "バス", "버스", "autobús",
             "ônibus", "autobus", "автобус", "حافلة"]) { push("bus"); }
    // car (exclude carriage; exclude the CN bus compound 公共汽车 that contains 汽车)
    if (has(&["轿车", "小汽车", "轎車", "自動車", "自动车", "自家用車"])
        || (t.contains("car") && !t.contains("carriage"))
        || (t.contains("汽车") && !t.contains("公共汽车"))
        || (t.contains("自動車") && !t.contains("バス"))
        || has(&["자동차", "승용차", "coche", "auto", "automóvil", "voiture",
                 "wagen", "macchina", "автомобиль", "легков", "سيارة"]))
    { push("car"); }
    // truck
    if has(&["truck", "lorry", "卡车", "货车", "貨車", "トラック", "트럭", "camión",
             "camion", "caminhão", "lastwagen", "lkw", "грузовик", "شاحنة"]) { push("truck"); }
    // traffic light
    if has(&["traffic light", "红绿灯", "交通灯", "紅綠燈", "信号灯", "信號燈", "交通信号",
             "信号機", "신호등", "semáforo", "ampel", "feu de", "feux", "светофор",
             "إشارة المرور"]) { push("traffic light"); }
    // fire hydrant
    if has(&["fire hydrant", "hydrant", "消防栓", "消火栓", "消防龙头", "消火栓",
             "消防ホース", "소화전", "boca de incendio", "hidrante", "bouche d",
             "hydranten", "гидрант", "صنبور"]) { push("fire hydrant"); }
    // boat
    if has(&["boat", "船", "ボート", "보트", "barco", "bateau", "boot", "barca",
             "лодка", "قارب"]) { push("boat"); }
    // parking meter
    if has(&["parking meter", "停车计时", "停车收费", "泊车计时", "停車收費", "パーキングメーター",
             "주차 미터", "주차요금", "parquímetro", "parcomètre", "parkuhr",
             "parchimetro", "паркомат"]) { push("parking meter"); }
    out.sort_unstable();
    out.dedup();
    out
}

/// True if this detector can handle the task (YOLO has the class). The caller
/// uses this to decide between the image detector and a reload.
pub(crate) fn can_handle(task: &str) -> bool {
    !task_to_classes(task).is_empty()
}

fn model_dir() -> PathBuf {
    let base = std::env::var_os("PUFFER_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".puffer")))
        .unwrap_or_else(|| PathBuf::from(".puffer"));
    base.join("models").join("captcha")
}

fn ensure_model() -> Result<PathBuf> {
    let file = std::env::var("PUFFER_YOLO_MODEL_FILE").unwrap_or_else(|_| DEFAULT_MODEL_FILE.to_string());
    let dir = model_dir();
    let path = dir.join(&file);
    if std::fs::metadata(&path).map(|m| m.len() >= MODEL_MIN_BYTES).unwrap_or(false) {
        return Ok(path);
    }
    let url = std::env::var("PUFFER_YOLO_MODEL_URL").unwrap_or_else(|_| DEFAULT_MODEL_URL.to_string());
    std::fs::create_dir_all(&dir)?;
    tracing::info!(target: "puffer::captcha", %url, "downloading YOLO ONNX (first image-challenge use)");
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(600)).build()?;
    let mut resp = client.get(&url).send().context("request YOLO model")?;
    if !resp.status().is_success() {
        return Err(anyhow!("YOLO model download HTTP {}", resp.status()));
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
    static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();
    if let Some(s) = SESSION.get() {
        return Ok(s);
    }
    let path = ensure_model()?;
    let sess = Session::builder()?
        .commit_from_file(&path)
        .with_context(|| format!("load YOLO onnx {}", path.display()))?;
    Ok(SESSION.get_or_init(|| Mutex::new(sess)))
}

struct Det {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    conf: f32,
}

/// Detect which of the `rows`x`rows` grid tiles contain the task's object.
/// `grid_png` is a screenshot of just the challenge grid. Returns tile indices
/// (row-major, 0-based).
pub(crate) fn detect_tiles(grid_png: &[u8], task: &str, rows: usize) -> Result<Vec<usize>> {
    let classes = task_to_classes(task);
    if classes.is_empty() {
        return Ok(Vec::new());
    }
    let img = image::load_from_memory(grid_png).context("decode grid image")?.to_rgb8();
    let (ow, oh) = (img.width() as f32, img.height() as f32);
    // Letterbox to INPUT x INPUT preserving aspect.
    let scale = (INPUT as f32 / ow).min(INPUT as f32 / oh);
    let (nw, nh) = ((ow * scale).round() as u32, (oh * scale).round() as u32);
    let resized = image::imageops::resize(&img, nw, nh, FilterType::Triangle);
    let (pad_x, pad_y) = ((INPUT - nw) / 2, (INPUT - nh) / 2);
    // CHW f32 [0,1], gray padding (114/255).
    let mut input = vec![114.0f32 / 255.0; (3 * INPUT * INPUT) as usize];
    let plane = (INPUT * INPUT) as usize;
    for y in 0..nh {
        for x in 0..nw {
            let p = resized.get_pixel(x, y);
            let (px, py) = ((x + pad_x) as usize, (y + pad_y) as usize);
            let idx = py * INPUT as usize + px;
            input[idx] = p[0] as f32 / 255.0;
            input[plane + idx] = p[1] as f32 / 255.0;
            input[2 * plane + idx] = p[2] as f32 / 255.0;
        }
    }
    let tensor = Tensor::from_array(([1usize, 3, INPUT as usize, INPUT as usize], input))?;
    let sess = session()?;
    let mut guard = sess.lock().unwrap();
    let outputs = guard.run(ort::inputs!["images" => tensor])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    // YOLOv8: [1, 84, 8400] — 84 = 4 bbox + 80 classes, transposed layout.
    let nc = 80usize;
    let feat = shape[shape.len() - 2] as usize; // 84
    let anchors = shape[shape.len() - 1] as usize; // 8400
    let at = |row: usize, a: usize| data[row * anchors + a];
    let mut dets: Vec<Det> = Vec::new();
    for a in 0..anchors {
        let mut best = 0.0f32;
        let mut best_c = usize::MAX;
        for c in 0..nc {
            let s = at(4 + c, a);
            if s > best {
                best = s;
                best_c = c;
            }
        }
        if best >= CONF && classes.contains(&best_c) {
            // box in letterboxed 640 space -> back to original image coords
            let cx = (at(0, a) - pad_x as f32) / scale;
            let cy = (at(1, a) - pad_y as f32) / scale;
            let w = at(2, a) / scale;
            let h = at(3, a) / scale;
            dets.push(Det { cx, cy, w, h, conf: best });
        }
    }
    let _ = feat;
    nms(&mut dets);
    // Map each detection to the grid tiles its box overlaps.
    let mut tiles = std::collections::BTreeSet::new();
    let cell_w = ow / rows as f32;
    let cell_h = oh / rows as f32;
    for d in &dets {
        let (x0, y0) = (d.cx - d.w / 2.0, d.cy - d.h / 2.0);
        let (x1, y1) = (d.cx + d.w / 2.0, d.cy + d.h / 2.0);
        for r in 0..rows {
            for col in 0..rows {
                let (tx0, ty0) = (col as f32 * cell_w, r as f32 * cell_h);
                let (tx1, ty1) = (tx0 + cell_w, ty0 + cell_h);
                // overlap area fraction of the tile
                let ix = (x1.min(tx1) - x0.max(tx0)).max(0.0);
                let iy = (y1.min(ty1) - y0.max(ty0)).max(0.0);
                let overlap = ix * iy;
                if overlap > 0.10 * cell_w * cell_h {
                    tiles.insert(r * rows + col);
                }
            }
        }
    }
    Ok(tiles.into_iter().collect())
}

fn nms(dets: &mut Vec<Det>) {
    dets.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<Det> = Vec::new();
    'outer: for d in dets.drain(..) {
        for k in &keep {
            if iou(&d, k) > IOU {
                continue 'outer;
            }
        }
        keep.push(d);
    }
    *dets = keep;
}

fn iou(a: &Det, b: &Det) -> f32 {
    let (ax0, ay0, ax1, ay1) = (a.cx - a.w / 2.0, a.cy - a.h / 2.0, a.cx + a.w / 2.0, a.cy + a.h / 2.0);
    let (bx0, by0, bx1, by1) = (b.cx - b.w / 2.0, b.cy - b.h / 2.0, b.cx + b.w / 2.0, b.cy + b.h / 2.0);
    let ix = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let iy = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = ix * iy;
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    #[test]
    fn detect_sample_grids() {
        let home = std::env::var("HOME").unwrap();
        let _ = home;
        for (path, task, rows) in [("/tmp/bus.jpg", "bus", 3), ("/tmp/bus.jpg", "person", 3), ("/tmp/bus.jpg", "car", 4)] {
            let file = path;
            let Ok(png) = std::fs::read(&path) else { eprintln!("skip missing {path}"); continue; };
            match super::detect_tiles(&png, task, rows) {
                Ok(tiles) => eprintln!("[{file}] {task} {rows}x{rows} -> tiles {tiles:?}"),
                Err(e) => eprintln!("[{file}] ERR {e:#}"),
            }
        }
    }
}
