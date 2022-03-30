use plotters::{
    coord::{ranged1d::AsRangedCoord, Shift},
    prelude::*,
    style::{RelativeSize, SizeDesc},
};
use std::ops::Range;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
const MIN_BASE: f64 = 10.;
const MAX_MAINTAIN: u32 = 100;

type GifVisualizer<X = Range<f64>, Y = Range<f64>> = Animator<BitMapBackend<'static>, X, Y>;
pub struct Animator<DB: DrawingBackend, X = Range<f64>, Y = Range<f64>>
where
    X: AsRangedCoord,
    Y: AsRangedCoord,
{
    draw_area: DrawingArea<DB, Shift>,
    x_range: DrawRange<X>,
    y_range: DrawRange<Y>,
    last_y_range: Option<Y>,
    last_refresh: u32,
    caption: Option<String>,
    x_desc: Option<String>,
    y_desc: Option<String>,
    x_label_formatter: Option<Box<dyn Fn(&<X as AsRangedCoord>::Value) -> String>>,
    y_label_formatter: Option<Box<dyn Fn(&<Y as AsRangedCoord>::Value) -> String>>,
    frame: usize,
    min_y_range: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
enum DrawRange<A> {
    Auto,
    Static(A),
}

impl<X, Y> GifVisualizer<X, Y>
where
    X: AsRangedCoord,
    Y: AsRangedCoord,
{
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new<T: AsRef<Path>>(path: T, size: (u32, u32), frame_rate: u32) -> Self {
        Self {
            draw_area: BitMapBackend::gif(path, size, 1000 / frame_rate)
                .expect("creating gif backend")
                .into_drawing_area(),
            x_range: DrawRange::Auto,
            y_range: DrawRange::Auto,
            last_y_range: None,
            last_refresh: 0,
            caption: None,
            x_desc: None,
            y_desc: None,
            x_label_formatter: None,
            y_label_formatter: None,
            frame: 0,
            min_y_range: None,
        }
    }
}
impl<DB: DrawingBackend> Animator<DB, Range<f64>, Range<f64>> {
    pub fn on_backend(back_end: DB) -> Self {
        Self {
            draw_area: back_end.into_drawing_area(),
            x_range: DrawRange::Auto,
            y_range: DrawRange::Auto,
            last_y_range: None,
            last_refresh: 0,
            caption: None,
            x_desc: None,
            y_desc: None,
            x_label_formatter: None,
            y_label_formatter: None,
            frame: 0,
            min_y_range: None,
        }
    }
    pub fn set_x_range(&mut self, x: Range<f64>) -> &mut Self {
        self.x_range = DrawRange::Static(x);
        self
    }
    pub fn set_y_range(&mut self, y: Range<f64>) -> &mut Self {
        self.y_range = DrawRange::Static(y);
        self
    }
    pub fn set_caption<S: ToString>(&mut self, s: S) -> &mut Self {
        self.caption = Some(s.to_string());
        self
    }
    pub fn set_x_desc<S: ToString>(&mut self, s: S) -> &mut Self {
        self.x_desc = Some(s.to_string());
        self
    }
    pub fn set_y_desc<S: ToString>(&mut self, s: S) -> &mut Self {
        self.y_desc = Some(s.to_string());
        self
    }
    pub fn set_x_label_formatter(
        &mut self,
        x_formatter: impl Fn(&<Range<f64> as AsRangedCoord>::Value) -> String + 'static,
    ) -> &mut Self {
        self.x_label_formatter = Some(Box::new(x_formatter));
        self
    }
    pub fn set_y_label_formatter(
        &mut self,
        y_formatter: impl Fn(&<Range<f64> as AsRangedCoord>::Value) -> String + 'static,
    ) -> &mut Self {
        self.y_label_formatter = Some(Box::new(y_formatter));
        self
    }
    pub fn set_min_y_range(&mut self, min_y: f64) -> &mut Self {
        self.min_y_range = min_y.into();
        self
    }
    pub fn get_frame_num(&self) -> usize {
        self.frame
    }
    pub fn new_frame<S>(
        &mut self,
        series: S,
    ) -> Result<impl Fn((i32, i32)) -> Option<(f64, f64)>, DrawingAreaErrorKind<DB::ErrorType>>
    where
        S: IntoIterator<Item = (f64, f64)>,
    {
        let (v, x_range, y_range): (Box<dyn Iterator<Item = (f64, f64)>>, _, _) =
            if let (DrawRange::Static(a), DrawRange::Static(b)) = (&self.x_range, &self.y_range) {
                (Box::new(series.into_iter()), a.clone(), b.clone())
            } else {
                let (mut temp_x_range, mut temp_y_range): (Option<Range<f64>>, Option<Range<f64>>) =
                    (None, None);
                let temp: Vec<_> = series
                    .into_iter()
                    .inspect(|(x, y)| {
                        if let Some(ref mut a) = temp_x_range {
                            if x < &a.start {
                                a.start = *x;
                            } else if x > &a.end {
                                a.end = *x
                            }
                        } else {
                            temp_x_range = Some(*x..*x)
                        }
                        if let Some(ref mut b) = temp_y_range {
                            if y < &b.start {
                                b.start = *y;
                            } else if y > &b.end {
                                b.end = *y
                            }
                        } else {
                            temp_y_range = Some(*y..*y)
                        }
                    })
                    .collect();
                let x_range = if let DrawRange::Static(ref a) = self.x_range {
                    a.clone()
                } else {
                    temp_x_range.unwrap_or(0. ..1.)
                };
                let y_range = if let DrawRange::Static(ref b) = self.y_range {
                    b.clone()
                } else {
                    match self.last_y_range {
                        Some(ref last)
                            if temp_y_range
                                .as_ref()
                                .map(|x| {
                                    last.start < x.start
                                        && last.end > x.end
                                        && (last.end - last.start).abs()
                                            < (x.end - x.start).abs() * MIN_BASE
                                })
                                .unwrap_or(true)
                                && self.last_refresh <= MAX_MAINTAIN =>
                        {
                            //as long as last range can hold current range and not too big, we keep use last range
                            self.last_refresh += 1;
                            last.clone()
                        }
                        _ => {
                            self.last_refresh = 0;
                            temp_y_range
                                .map(|mut y| {
                                    let base: f64 = MIN_BASE;
                                    let mut range = (y.end - y.start).abs();
                                    if let Some(min) = self.min_y_range {
                                        if min > range {
                                            range = min
                                        }
                                    };
                                    let axis_mod = base.powi(range.log(base).floor() as i32);
                                    let scale_dis = base * 0.2;
                                    range = ((range / axis_mod / scale_dis).ceil()) * scale_dis;
                                    if range > base {
                                        //make sure ceil() won't overflow base
                                        range = base;
                                    }
                                    range *= axis_mod;
                                    let center = (y.start + y.end) / 2.;
                                    if y.start < y.end {
                                        y.start =
                                            ((center - range / 2.) / axis_mod).floor() * axis_mod;
                                        y.end =
                                            ((center + range / 2.) / axis_mod).ceil() * axis_mod;
                                    } else {
                                        y.start =
                                            ((center + range / 2.) / axis_mod).ceil() * axis_mod;
                                        y.end =
                                            ((center - range / 2.) / axis_mod).floor() * axis_mod;
                                    }
                                    self.last_y_range = y.clone().into();
                                    y
                                })
                                .unwrap_or(0. ..1.)
                        }
                    }
                };
                (Box::new(temp.into_iter()), x_range, y_range)
            };
        self.draw_area.fill(&WHITE)?;
        let mut ctx = ChartBuilder::on(&self.draw_area)
            .margin_right(2.percent_width())
            .x_label_area_size(10.percent_height())
            .y_label_area_size(10.percent_width())
            .caption(
                self.caption
                    .as_ref()
                    .map(|x| format!("{} {}", x, self.frame))
                    .unwrap_or_else(|| self.frame.to_string()),
                ("sans-serif", 5.percent_height()),
            )
            .build_cartesian_2d(
                x_range,
                y_range.clone().step(
                    MIN_BASE
                        .powi(
                            ((y_range.end - y_range.start).abs().log(MIN_BASE) - 0.1).floor() //if not -0.1 and just over 1, only 1 index will be draw 
                                as i32,
                        )
                        .copysign(y_range.end - y_range.start),
                ),
            )?;
        let mut mesh = ctx.configure_mesh();
        mesh.x_label_style(("sans-serif", 5.percent()))
            .y_label_style(("sans-serif", 5.percent()))
            .y_labels(12);
        if let Some(ref s) = self.x_desc {
            mesh.x_desc(s);
        }
        if let Some(ref s) = self.y_desc {
            mesh.y_desc(s);
        }
        if let Some(ref f) = self.x_label_formatter {
            mesh.x_label_formatter(f);
        }
        if let Some(ref f) = self.y_label_formatter {
            mesh.y_label_formatter(f);
        }
        mesh.axis_style(BLACK.stroke_width(0.1.percent().in_pixels(&self.draw_area) as u32 + 1))
            .draw()?;
        ctx.draw_series(LineSeries::new(v, RED))?;
        self.draw_area.present()?;
        self.frame += 1;
        return Ok(ctx.into_coord_trans());
    }
}

pub struct ColorMapVisualizer<
    DB: DrawingBackend,
    B = f64,
    XF: Fn(&usize) -> String = fn(&usize) -> String,
    YF: Fn(&usize) -> String = fn(&usize) -> String,
> {
    draw_area: DrawingArea<DB, Shift>,
    matrix: Vec<Vec<B>>,
    color_range: DrawRange<Range<B>>,
    caption: Option<String>,
    x_desc: Option<String>,
    y_desc: Option<String>,
    x_label_formatter: Option<XF>,
    y_label_formatter: Option<YF>,
    auto_range: Option<Range<B>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> ColorMapVisualizer<BitMapBackend<'a>, f64, fn(&usize) -> String, fn(&usize) -> String> {
    pub fn new<P: AsRef<Path>>(path: &'a P, size: (u32, u32)) -> Self {
        Self {
            draw_area: BitMapBackend::new(path, size).into_drawing_area(),
            matrix: Vec::new(),
            color_range: DrawRange::Auto,
            caption: None,
            x_desc: None,
            y_desc: None,
            x_label_formatter: None,
            y_label_formatter: None,
            auto_range: None,
        }
    }
}
impl<DB: DrawingBackend> ColorMapVisualizer<DB, f64, fn(&usize) -> String, fn(&usize) -> String> {
    pub fn on_backend(back_end: DB) -> Self {
        Self {
            draw_area: back_end.into_drawing_area(),
            matrix: Vec::new(),
            color_range: DrawRange::Auto,
            caption: None,
            x_desc: None,
            y_desc: None,
            x_label_formatter: None,
            y_label_formatter: None,
            auto_range: None,
        }
    }
    pub fn set_color_range(&mut self, x: Range<f64>) -> &mut Self {
        self.color_range = DrawRange::Static(x);
        self
    }
    pub fn set_caption<S: ToString>(&mut self, s: S) -> &mut Self {
        self.caption = Some(s.to_string());
        self
    }
    pub fn set_x_desc<S: ToString>(&mut self, s: S) -> &mut Self {
        self.x_desc = Some(s.to_string());
        self
    }
    pub fn set_y_desc<S: ToString>(&mut self, s: S) -> &mut Self {
        self.y_desc = Some(s.to_string());
        self
    }
    pub fn set_x_label_formatter(&mut self, x_formatter: fn(&usize) -> String) -> &mut Self {
        self.x_label_formatter = Some(x_formatter);
        self
    }
    pub fn set_y_label_formatter(&mut self, y_formatter: fn(&usize) -> String) -> &mut Self {
        self.y_label_formatter = Some(y_formatter);
        self
    }
    pub fn push(&mut self, row: Vec<f64>) -> &mut Self {
        debug_assert!(self
            .matrix
            .last()
            .map(|x| x.len() == row.len())
            .unwrap_or(true));
        row.iter().for_each(|x| {
            if let Some(ref mut o) = self.auto_range {
                if o.start > *x {
                    o.start = *x;
                } else if o.end < *x {
                    o.end = *x;
                }
            } else {
                self.auto_range = Some((*x)..(*x));
            }
        });
        self.matrix.push(row);
        self
    }
    pub fn draw(
        &self,
    ) -> Result<impl Fn((i32, i32)) -> Option<(usize, usize)>, DrawingAreaErrorKind<DB::ErrorType>>
    {
        let (range_max, range_min) = match self.color_range.clone() {
            DrawRange::Auto => self
                .auto_range
                .clone()
                .map(|x| (x.end, x.start))
                .unwrap_or((1., 0.)),
            DrawRange::Static(s) => (s.end, s.start),
        };
        let range = if range_max != range_min {
            range_max - range_min
        } else {
            1.
        };
        let color_map = |v: f64| ((v - range_min) / range);
        self.draw_area.fill(&WHITE)?;
        let (area, bar) = self.draw_area.split_horizontally(RelativeSize::Width(0.85));
        let mut builder_map = ChartBuilder::on(&area);
        builder_map
            .margin_right(2.percent_width().in_pixels(&self.draw_area))
            .margin_top(2.percent_height().in_pixels(&self.draw_area))
            .y_label_area_size(10.percent_width().in_pixels(&self.draw_area))
            .x_label_area_size(10.percent_height().in_pixels(&self.draw_area));
        if let Some(ref s) = self.caption {
            builder_map.caption(
                s,
                (
                    "sans-serif",
                    2.5.percent_height().in_pixels(&self.draw_area),
                ),
            );
        }
        let row_len = self.matrix.last().map_or(0, |r| r.len());
        let column_len = self.matrix.len();
        let mut chart_map = builder_map.build_cartesian_2d(0..row_len, 0..column_len)?;
        let mut mesh_map = chart_map.configure_mesh();
        mesh_map
            .x_label_style(("sans-serif", 5.percent_height().in_pixels(&self.draw_area)))
            .y_label_style(("sans-serif", 5.percent_width().in_pixels(&self.draw_area)))
            .disable_x_mesh()
            .disable_y_mesh();
        if let Some(ref s) = self.x_desc {
            mesh_map.x_desc(s);
        }
        if let Some(ref s) = self.y_desc {
            mesh_map.y_desc(s);
        }
        if let Some(ref f) = self.x_label_formatter {
            mesh_map.x_label_formatter(f);
        }
        if let Some(ref f) = self.y_label_formatter {
            mesh_map.y_label_formatter(f);
        }
        mesh_map.draw()?;
        draw_map(&mut chart_map, &self.matrix, color_map);

        let mut builder_bar = ChartBuilder::on(&bar);
        builder_bar
            .margin_right(2.percent_width().in_pixels(&self.draw_area))
            .margin_top(2.percent_height().in_pixels(&self.draw_area))
            .margin_bottom(10.percent_height().in_pixels(&self.draw_area)) //take the space for hidden x axis
            .y_label_area_size(10.percent_width().in_pixels(&self.draw_area));
        let mut chart_bar = builder_bar.build_cartesian_2d((0f64)..1., range_min..range_max)?;
        let mut mesh_bar = chart_bar.configure_mesh();
        let step = range / (column_len - 1).max(1) as f64;
        mesh_bar
            .disable_x_mesh()
            .disable_y_mesh()
            .disable_x_axis()
            .y_label_style(("sans-serif", 5.percent_width().in_pixels(&self.draw_area)));
        mesh_bar.draw()?;
        chart_bar.draw_series(
            std::iter::successors(Some(range_min), |x| Some(step + x))
                .take_while(|x| *x <= range_max)
                .map(|v| {
                    Rectangle::new(
                        [(0., v - step / 2.), (1., v + step / 2.)],
                        HSLColor(
                            240.0 / 360.0 - 240.0 / 360.0 * color_map(v),
                            0.7,
                            0.1 + 0.4 * color_map(v),
                        )
                        .filled(),
                    )
                }),
        )?;
        self.draw_area.present()?;
        return Ok(chart_map.into_coord_trans());
    }
}

#[cfg(not(feature = "rayon"))]
fn draw_map<DB: DrawingBackend>(
    ctx: &mut ChartContext<
        DB,
        Cartesian2d<
            plotters::coord::types::RangedCoordusize,
            plotters::coord::types::RangedCoordusize,
        >,
    >,
    data: &Vec<Vec<f64>>,
    color_map: impl Fn(f64) -> f64,
) {
    ctx.draw_series(
        data.iter()
            .enumerate()
            .map(|(y, l)| l.into_iter().enumerate().map(move |(x, v)| (x, y, v)))
            .flatten()
            .map(|(x, y, v)| {
                Rectangle::new(
                    [(x, y), (x + 1, y + 1)],
                    HSLColor(
                        240.0 / 360.0 - 240.0 / 360.0 * color_map(*v),
                        0.7,
                        0.1 + 0.4 * color_map(*v),
                    )
                    .filled(),
                )
            }),
    )
    .expect("plotting reactangles");
}
#[cfg(all(feature = "wasm-rayon", not(target_arch = "wasm32")))]
std::compile_error!("wasm-rayon feature can be only used with wasm32 target");

#[cfg(all(feature = "wasm-rayon", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg(feature = "rayon")]
fn draw_map<DB: DrawingBackend>(
    ctx: &mut ChartContext<
        DB,
        Cartesian2d<
            plotters::coord::types::RangedCoordusize,
            plotters::coord::types::RangedCoordusize,
        >,
    >,
    data: &Vec<Vec<f64>>,
    color_map: impl Fn(f64) -> f64 + Sync,
) {
    const MAX_PARTS_NUM: usize = 8;
    const MIN_PARTS_LEN: usize = 1024;
    let column_num = data.first().map_or(0, |x| x.len());
    let row_num = data.len();
    let parts_num = ((column_num * row_num + MIN_PARTS_LEN - 1) / MIN_PARTS_LEN).min(MAX_PARTS_NUM);
    let parts_row_len = (row_num + parts_num - 1) / parts_num;
    let mut areas = Vec::with_capacity(parts_num);
    let mut area_left = ctx.plotting_area().strip_coord_spec();
    let split_pixels = (100. * parts_row_len as f64 / row_num as f64)
        .percent_height()
        .in_pixels(&area_left);
    let total_pixels = area_left.dim_in_pixel().1;
    (0..(parts_num - 1)).for_each(|i| {
        let (upper, lower) =
            area_left.split_vertically(total_pixels as i32 - (i as i32 + 1) * split_pixels);
        area_left = upper;
        areas.push(lower);
    });
    areas.push(area_left);
    use rayon::prelude::*;
    let mut sub_plots: Vec<_> = areas
        .iter()
        .map(|x| BitMapElement::new((0, 0), x.dim_in_pixel()))
        .collect::<Vec<_>>();
    let color_map = &color_map;
    sub_plots
        .par_iter_mut()
        .zip(data.par_chunks(parts_row_len))
        .for_each(|(s, d)| {
            ChartBuilder::on(&s.as_bitmap_backend().into_drawing_area())
                .build_cartesian_2d(0..d.first().map_or(0, |x| x.len()), 0..d.len())
                .expect("chart builder on subplot")
                .draw_series(
                    d.iter()
                        .enumerate()
                        .map(move |(y, v)| {
                            v.iter().enumerate().map(move |(x, v)| {
                                Rectangle::new(
                                    [(x, y), (x + 1, y + 1)],
                                    HSLColor(
                                        240.0 / 360.0 - 240.0 / 360.0 * color_map(*v),
                                        0.7,
                                        0.1 + 0.4 * color_map(*v),
                                    )
                                    .filled(),
                                )
                            })
                        })
                        .flatten(),
                )
                .expect("drawing rectangles");
        });
    areas
        .into_iter()
        .zip(sub_plots.into_iter())
        .for_each(|(a, s)| a.draw(&s).expect("placing subplots on area"))
}
