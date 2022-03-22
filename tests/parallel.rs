use std::path::PathBuf;

use plotters::prelude::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "rayon")]
#[test]
fn split_area() {
    let mut c = Vec::new();

    for i in 0..10 {
        c.push(vec![i as f64; 10]);
    }

    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push("split_area.png");
    let draw_area = BitMapBackend::new(&path, (1000, 1000)).into_drawing_area();
    draw_area.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&draw_area);
    chart.margin(5.percent());
    let mut ctx = chart.build_cartesian_2d(0usize..10, 0usize..10).unwrap();
    let mut mesh = ctx.configure_mesh();
    mesh.x_label_style(("sans-serif", 5.percent_height()))
        .y_label_style(("sans-serif", 5.percent_width()));
    mesh.draw().unwrap();
    let region = ctx.plotting_area();
    region.fill(&BLACK).unwrap();
    let areas = region.strip_coord_spec().split_evenly((2, 1));

    let plot_sub =
        |data: &[Vec<_>], backend: &DrawingArea<BitMapBackend, plotters::coord::Shift>| {
            ChartBuilder::on(backend)
                .build_cartesian_2d(0usize..10, 0usize..5)
                .unwrap()
                .draw_series(
                    data.iter()
                        .enumerate()
                        .map(|(y, v)| {
                            v.iter().enumerate().map(move |(x, v)| {
                                Rectangle::new(
                                    [(x, y), (x + 1, y + 1)],
                                    HSLColor(
                                        240.0 / 360.0 - 240.0 / 360.0 * v / 10.,
                                        0.7,
                                        0.1 + 0.4 * v / 10.,
                                    )
                                    .filled(),
                                )
                            })
                        })
                        .flatten(),
                )
                .unwrap();
        };
    let mut sub_draws = areas
        .iter()
        .map(|a| BitMapElement::new((0, 0), a.dim_in_pixel()))
        .collect::<Vec<_>>();
    sub_draws
        .par_iter_mut()
        .rev()
        .zip(c.par_chunks(5))
        .for_each(|(s, d)| plot_sub(d, &s.as_bitmap_backend().into_drawing_area()));
    areas
        .into_iter()
        .zip(sub_draws.into_iter())
        .for_each(|(a, s)| a.draw(&s).unwrap());
    draw_area.present().unwrap();
}
