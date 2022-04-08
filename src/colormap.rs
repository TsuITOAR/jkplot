use super::*;

pub struct RawMapVisualizer<
    B = f64,
    XF: Fn(&usize) -> String = fn(&usize) -> String,
    YF: Fn(&usize) -> String = fn(&usize) -> String,
> {
    color_range: DrawRange<Range<B>>,
    caption: Option<String>,
    x_desc: Option<String>,
    y_desc: Option<String>,
    x_label_formatter: Option<XF>,
    y_label_formatter: Option<YF>,
    auto_range: Option<Range<B>>,
}

impl<B, XF: Fn(&usize) -> String, YF: Fn(&usize) -> String> Default
    for RawMapVisualizer<B, XF, YF>
{
    fn default() -> Self {
        Self {
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

impl<B> RawMapVisualizer<B, fn(&usize) -> String, fn(&usize) -> String> {
    pub fn binding<DB: DrawingBackend>(
        self,
        matrix: Vec<Vec<B>>,
        draw_area: DrawingArea<DB, Shift>,
    ) -> ColorMapVisualizer<DB, B, fn(&usize) -> String, fn(&usize) -> String> {
        ColorMapVisualizer {
            draw_area,
            matrix,
            raw: self,
        }
    }
    pub fn set_color_range(&mut self, x: Range<B>) -> &mut Self {
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
}
impl RawMapVisualizer<f64, fn(&usize) -> String, fn(&usize) -> String> {
    pub fn update_range(&mut self, row: &[f64]) -> &mut Self {
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
        self
    }
    pub fn draw_on<DB: DrawingBackend>(
        &self,
        matrix: &Vec<Vec<f64>>,
        draw_area: &DrawingArea<DB, Shift>,
    ) -> Result<impl Fn((i32, i32)) -> Option<(usize, usize)>, DrawingAreaErrorKind<DB::ErrorType>>
    {
        let row_len = matrix.last().map_or(0, |r| r.len());
        let column_len = matrix.len();
        assert_ne!(row_len * column_len, 0);
        let (range_max, range_min) = match self.color_range.clone() {
            DrawRange::Auto => self
                .auto_range
                .as_ref()
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
        draw_area.fill(&WHITE)?;
        let (area, bar) = draw_area.split_horizontally(RelativeSize::Width(0.85));
        let mut builder_map = ChartBuilder::on(&area);
        builder_map
            .margin_right(2.percent_width().in_pixels(draw_area))
            .margin_top(2.percent_height().in_pixels(draw_area))
            .y_label_area_size(10.percent_width().in_pixels(draw_area))
            .x_label_area_size(10.percent_height().in_pixels(draw_area));
        if let Some(ref s) = self.caption {
            builder_map.caption(s, ("sans-serif", 2.5.percent_height().in_pixels(draw_area)));
        }

        let mut chart_map = builder_map.build_cartesian_2d(0..row_len, 0..column_len)?;
        let mut mesh_map = chart_map.configure_mesh();
        mesh_map
            .x_label_style(("sans-serif", 5.percent_height().in_pixels(draw_area)))
            .y_label_style(("sans-serif", 5.percent_width().in_pixels(draw_area)))
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
        draw_map(&mut chart_map, matrix, color_map);

        let mut builder_bar = ChartBuilder::on(&bar);
        builder_bar
            .margin_right(2.percent_width().in_pixels(draw_area))
            .margin_top(2.percent_height().in_pixels(draw_area))
            .margin_bottom(10.percent_height().in_pixels(draw_area)) //take the space for hidden x axis
            .y_label_area_size(10.percent_width().in_pixels(draw_area));
        let mut chart_bar =
            builder_bar.build_cartesian_2d((0f64)..1., range_min..(range + range_min))?;
        let mut mesh_bar = chart_bar.configure_mesh();
        let step = range / (column_len - 1).max(1) as f64;
        mesh_bar
            .disable_x_mesh()
            .disable_y_mesh()
            .disable_x_axis()
            .y_label_style(("sans-serif", 5.percent_width().in_pixels(draw_area)));
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
        draw_area.present()?;
        return Ok(chart_map.into_coord_trans());
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
    raw: RawMapVisualizer<B, XF, YF>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> ColorMapVisualizer<BitMapBackend<'a>, f64, fn(&usize) -> String, fn(&usize) -> String> {
    pub fn new<P: AsRef<Path>>(path: &'a P, size: (u32, u32)) -> Self {
        Self {
            draw_area: BitMapBackend::new(path, size).into_drawing_area(),
            matrix: Vec::new(),
            raw: Default::default(),
        }
    }
}
impl<DB: DrawingBackend> ColorMapVisualizer<DB, f64, fn(&usize) -> String, fn(&usize) -> String> {
    pub fn on_backend(back_end: DB) -> Self {
        Self {
            draw_area: back_end.into_drawing_area(),
            matrix: Vec::new(),
            raw: Default::default(),
        }
    }
    pub fn on_draw_area(draw_area: DrawingArea<DB, Shift>) -> Self {
        Self {
            draw_area,
            matrix: Vec::new(),
            raw: Default::default(),
        }
    }
    pub fn set_color_range(&mut self, x: Range<f64>) -> &mut Self {
        self.raw.set_color_range(x);
        self
    }
    pub fn set_caption<S: ToString>(&mut self, s: S) -> &mut Self {
        self.raw.set_caption(s);
        self
    }
    pub fn set_x_desc<S: ToString>(&mut self, s: S) -> &mut Self {
        self.raw.set_x_desc(s);
        self
    }
    pub fn set_y_desc<S: ToString>(&mut self, s: S) -> &mut Self {
        self.raw.set_y_desc(s);
        self
    }
    pub fn set_x_label_formatter(&mut self, x_formatter: fn(&usize) -> String) -> &mut Self {
        self.raw.set_x_label_formatter(x_formatter);
        self
    }
    pub fn set_y_label_formatter(&mut self, y_formatter: fn(&usize) -> String) -> &mut Self {
        self.raw.set_y_label_formatter(y_formatter);
        self
    }
    pub fn push(&mut self, row: Vec<f64>) -> &mut Self {
        debug_assert!(self
            .matrix
            .last()
            .map(|x| x.len() == row.len())
            .unwrap_or(true));
        self.raw.update_range(&row);
        self.matrix.push(row);
        self
    }
    pub fn draw(
        &self,
    ) -> Result<impl Fn((i32, i32)) -> Option<(usize, usize)>, DrawingAreaErrorKind<DB::ErrorType>>
    {
        self.raw.draw_on(&self.matrix, &self.draw_area)
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
