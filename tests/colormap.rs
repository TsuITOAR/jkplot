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
#[test]
#[should_panic]
fn empty_data() {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("empty_data.png");
    let c = ColorMapVisualizer::new(&path, (1000, 1000));
    let _s = c.draw().unwrap();
}
