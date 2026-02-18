// src/amr/rect.rs

/// Integer rectangle in (i,j) index space, using half-open intervals:
/// [i0, i0+nx) × [j0, j0+ny)
///
/// This is used to specify AMR patch coverage on the *coarse* (base) grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect2i {
    pub i0: usize,
    pub j0: usize,
    pub nx: usize,
    pub ny: usize,
}

impl Rect2i {
    #[inline]
    pub fn new(i0: usize, j0: usize, nx: usize, ny: usize) -> Self {
        Self { i0, j0, nx, ny }
    }

    #[inline]
    pub fn i1(self) -> usize {
        self.i0 + self.nx
    }

    #[inline]
    pub fn j1(self) -> usize {
        self.j0 + self.ny
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.nx == 0 || self.ny == 0
    }

    /// Returns true if this rect is fully contained within a domain of size (nx, ny).
    #[inline]
    pub fn fits_in(self, nx: usize, ny: usize) -> bool {
        self.i0 <= nx && self.j0 <= ny && self.i1() <= nx && self.j1() <= ny
    }

    /// Intersection of two rectangles.
    pub fn intersect(self, other: Rect2i) -> Option<Rect2i> {
        let i0 = self.i0.max(other.i0);
        let j0 = self.j0.max(other.j0);
        let i1 = self.i1().min(other.i1());
        let j1 = self.j1().min(other.j1());
        if i1 <= i0 || j1 <= j0 {
            None
        } else {
            Some(Rect2i::new(i0, j0, i1 - i0, j1 - j0))
        }
    }

    /// Expand (dilate) by `pad` cells in all directions, clamped to domain (nx, ny).
    pub fn dilate_clamped(self, pad: usize, nx: usize, ny: usize) -> Rect2i {
        let i0 = self.i0.saturating_sub(pad);
        let j0 = self.j0.saturating_sub(pad);
        let i1 = (self.i1() + pad).min(nx);
        let j1 = (self.j1() + pad).min(ny);
        Rect2i::new(i0, j0, i1 - i0, j1 - j0)
    }
}
