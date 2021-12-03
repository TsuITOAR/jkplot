use std::path::PathBuf;

use jkplot::ColorMapVisualizer;

#[test]
fn colo_map() {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.set_file_name("color_map.png");
    let mut c = ColorMapVisualizer::new(path, (1000, 1000));
    for i in 1..10 {
        c.push(vec![i as f64; 10]);
    }
    c.draw();
}