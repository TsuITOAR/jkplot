use std::path::PathBuf;

use jkplot::ColorMapVisualizer;

#[test]
fn color_map() {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("color_map.png");
    let mut c = ColorMapVisualizer::new(&path, (1000, 1000));
    for i in 0..128 {
        c.push(vec![i as f64; 100]);
    }
    let _s = c.draw().unwrap();
}
