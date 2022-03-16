use plotters::{
    coord::{ranged1d::AsRangedCoord, Shift},
    prelude::*,
    style::{RelativeSize, SizeDesc},
};
use std::{ops::Range, path::Path};

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
}

enum DrawRange<A> {
    Auto,
    Static(A),
}

impl<X, Y> GifVisualizer<X, Y>
where
    X: AsRangedCoord,
    Y: AsRangedCoord,
{
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
    pub fn get_frame_num(&self) -> usize {
        self.frame
    }
    pub fn new_frame<S>(&mut self, series: S)
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
                                    let axis_mod = base.powi(range.log(base).floor() as i32);
                                    let scale_dis = base.sqrt().ceil();
                                    range = ((range / axis_mod / scale_dis).ceil()) * scale_dis;
                                    if range > base {
                                        //make sure ceil() won't overflow base
                                        range = base;
                                    }
                                    range *= axis_mod;
                                    let center = (y.start + y.end) / 2.;
                                    y.start = center - range / 2.;
                                    y.end = center + range / 2.;
                                    self.last_y_range = y.clone().into();
                                    y
                                })
                                .unwrap_or(0. ..1.)
                        }
                    }
                };
                (Box::new(temp.into_iter()), x_range, y_range)
            };
        self.draw_area.fill(&WHITE).expect("filling background");
        let mut ctx = ChartBuilder::on(&self.draw_area)
            .margin_right(0.02.percent_width())
            .x_label_area_size(0.1.percent_height())
            .y_label_area_size(0.1.percent_width())
            .caption(
                self.caption
                    .as_ref()
                    .map(|x| format!("{} {}", x, self.frame))
                    .unwrap_or_else(|| self.frame.to_string()),
                ("sans-serif", 0.05.percent_height()),
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
            )
            .expect("building chart context");
        let mut mesh = ctx.configure_mesh();
        mesh.x_label_style(("sans-serif", 0.025.percent_height()))
            .y_label_style(("sans-serif", 0.025.percent_width()))
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
        mesh.axis_style(BLACK.stroke_width(1e-3.percent().in_pixels(&self.draw_area) as u32 + 1))
            .draw()
            .expect("drawing mesh");
        ctx.draw_series(LineSeries::new(v, RED))
            .expect("plotting points");
        self.draw_area.present().expect("flushing current frame");
        self.frame += 1;
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
    pub fn draw(self) {
        let (range_max, range_min) = match self.color_range {
            DrawRange::Auto => self
                .auto_range
                .map(|x| (x.end, x.start))
                .unwrap_or((1., 0.)),
            DrawRange::Static(s) => (s.end, s.start),
        };
        let range = if range_max != range_min {
            range_max - range_min
        } else {
            1.
        };
        let color_map = |v| ((v - range_min) / range);
        self.draw_area
            .fill(&WHITE)
            .expect("filling area background");
        let (area, bar) = self.draw_area.split_horizontally(RelativeSize::Width(0.85));
        let mut builder = ChartBuilder::on(&area);
        builder
            .margin_right(0.02.percent_width())
            .margin_top(0.02.percent_height())
            .y_label_area_size(0.1.percent_width())
            .x_label_area_size(0.1.percent_height());
        if let Some(s) = self.caption {
            builder.caption(s, ("sans-serif", 0.025.percent_height()));
        }
        let row_len = self
            .matrix
            .last()
            .expect("getting last row of matrix")
            .len();
        let column_len = self.matrix.len();
        let mut chart = builder
            .build_cartesian_2d(
                (0..row_len).step(row_len / 5),
                (0..column_len).step(column_len / 5),
            )
            .expect("building chart context");
        let mut mesh = chart.configure_mesh();
        mesh.x_label_style(("sans-serif", 0.025.percent_height()))
            .y_label_style(("sans-serif", 0.025.percent_width()))
            .disable_x_mesh()
            .disable_y_mesh();
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
        mesh.draw().expect("drawing mesh");
        chart
            .draw_series(
                self.matrix
                    .into_iter()
                    .enumerate()
                    .map(|(y, l)| l.into_iter().enumerate().map(move |(x, v)| (x, y, v)))
                    .flatten()
                    .map(|(x, y, v)| {
                        Rectangle::new(
                            [(x, y), (x + 1, y + 1)],
                            HSLColor(
                                240.0 / 360.0 - 240.0 / 360.0 * color_map(v),
                                0.7,
                                0.1 + 0.4 * color_map(v),
                            )
                            .filled(),
                        )
                    }),
            )
            .expect("plotting pixels");
        area.present().expect("writing picture to file");

        let mut builder = ChartBuilder::on(&bar);
        builder
            .margin_right(0.02.percent_width())
            .margin_top(0.02.percent_height())
            .y_label_area_size(0.01.percent_width())
            .x_label_area_size(0.01.percent_height());
        let mut chart = builder
            .build_cartesian_2d((0.)..1., range_min..range_max)
            .expect("building colorbar");
        let mut mesh = chart.configure_mesh();
        let step = range / (column_len - 1).max(1) as f64;
        mesh.disable_x_mesh()
            .disable_y_mesh()
            .disable_x_axis()
            .y_label_style(("sans-serif", 0.025.percent_width()));
        mesh.draw().expect("drawing colorbar mesh");
        chart
            .draw_series(
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
            )
            .expect("plotting colorbar");
        bar.present().expect("writing colorbar to file");
    }
}
