use std::path::PathBuf;

use jkplot::ColorMapVisualizer;

#[test]
fn color_map() {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("color_map.png");
    let mut c = ColorMapVisualizer::new(&path, (1000, 1000));
    for i in 1..10 {
        c.push(vec![i as f64; 10]);
    }
    c.draw().unwrap();
}
