use super::*;
const MIN_BASE: f64 = 10.;
const MAX_MAINTAIN: u32 = 100;

pub struct RawAnimator<X = Range<f64>, Y = Range<f64>>
where
    X: AsRangedCoord,
    Y: AsRangedCoord,
{
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

impl<X, Y> Default for RawAnimator<X, Y>
where
    X: AsRangedCoord,
    Y: AsRangedCoord,
{
    fn default() -> Self {
        Self {
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

impl RawAnimator<Range<f64>, Range<f64>> {
    pub fn binding<DB: DrawingBackend>(
        self,
        draw_area: DrawingArea<DB, Shift>,
    ) -> Animator<DB, Range<f64>, Range<f64>> {
        Animator {
            draw_area,
            raw: self,
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
    pub fn new_frame_on<S, DB: DrawingBackend>(
        &mut self,
        series: S,
        draw_area: &DrawingArea<DB, Shift>,
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
        draw_area.fill(&WHITE)?;
        let mut ctx = ChartBuilder::on(&draw_area)
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
        mesh.axis_style(BLACK.stroke_width(0.1.percent().in_pixels(draw_area) as u32 + 1))
            .draw()?;
        ctx.draw_series(LineSeries::new(v, &RED))?;
        draw_area.present()?;
        self.frame += 1;
        return Ok(ctx.into_coord_trans());
    }
}

pub struct Animator<DB: DrawingBackend, X = Range<f64>, Y = Range<f64>>
where
    X: AsRangedCoord,
    Y: AsRangedCoord,
{
    draw_area: DrawingArea<DB, Shift>,
    raw: RawAnimator<X, Y>,
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
            raw: Default::default(),
        }
    }
}
impl<DB: DrawingBackend> Animator<DB, Range<f64>, Range<f64>> {
    pub fn on_backend(back_end: DB) -> Self {
        Self {
            draw_area: back_end.into_drawing_area(),
            raw: Default::default(),
        }
    }
    pub fn on_draw_area(draw_area: DrawingArea<DB, Shift>) -> Self {
        Self {
            draw_area,
            raw: Default::default(),
        }
    }
    pub fn set_x_range(&mut self, x: Range<f64>) -> &mut Self {
        self.raw.set_x_range(x);
        self
    }
    pub fn set_y_range(&mut self, y: Range<f64>) -> &mut Self {
        self.raw.set_y_range(y);
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
    pub fn set_x_label_formatter(
        &mut self,
        x_formatter: impl Fn(&<Range<f64> as AsRangedCoord>::Value) -> String + 'static,
    ) -> &mut Self {
        self.raw.set_x_label_formatter(x_formatter);
        self
    }
    pub fn set_y_label_formatter(
        &mut self,
        y_formatter: impl Fn(&<Range<f64> as AsRangedCoord>::Value) -> String + 'static,
    ) -> &mut Self {
        self.raw.set_y_label_formatter(y_formatter);
        self
    }
    pub fn set_min_y_range(&mut self, min_y: f64) -> &mut Self {
        self.raw.set_min_y_range(min_y);
        self
    }
    pub fn get_frame_num(&self) -> usize {
        self.raw.get_frame_num()
    }
    pub fn new_frame<S>(
        &mut self,
        series: S,
    ) -> Result<impl Fn((i32, i32)) -> Option<(f64, f64)>, DrawingAreaErrorKind<DB::ErrorType>>
    where
        S: IntoIterator<Item = (f64, f64)>,
    {
        self.raw.new_frame_on(series, &self.draw_area)
    }
    pub fn into_raw(self) -> RawAnimator<Range<f64>, Range<f64>> {
        self.raw
    }
}
