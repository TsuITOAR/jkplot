use plotters::{
    coord::{ranged1d::AsRangedCoord, Shift},
    prelude::*,
    style::{RelativeSize, SizeDesc},
};
use std::ops::Range;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[derive(Debug, Clone, Copy)]
enum DrawRange<A> {
    Auto,
    Static(A),
}

impl<A> Default for DrawRange<A> {
    fn default() -> Self {
        Self::Auto
    }
}

pub mod animator;
pub use animator::*;
pub mod colormap;
pub use colormap::*;

type GifVisualizer<X = Range<f64>, Y = Range<f64>> = Animator<BitMapBackend<'static>, X, Y>;
