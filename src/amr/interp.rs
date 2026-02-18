// src/amr/interp.rs

use crate::vec3::normalize;
use crate::vector_field::VectorField2D;

#[inline]
fn clamp_usize(x: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if x <= 0 {
        0
    } else {
        let max = (n - 1) as isize;
        if x >= max { n - 1 } else { x as usize }
    }
}

/// Bilinear sample (with clamping) of a cell-centred `VectorField2D` at physical coordinates (x,y).
///
/// Assumptions (consistent with VectorField2D::resample_to_grid):
/// - grid origin at (0,0) in the lower-left corner
/// - cell centres at (i+0.5)*dx, (j+0.5)*dy
///
/// Returns a *renormalised* unit vector.
pub fn sample_bilinear_unit(field: &VectorField2D, x: f64, y: f64) -> [f64; 3] {
    let g = &field.grid;
    let nx = g.nx;
    let ny = g.ny;
    if nx == 0 || ny == 0 {
        return [0.0, 0.0, 1.0];
    }

    // Convert physical coordinate to continuous cell-centre index.
    // If x = (i+0.5)dx, then i = x/dx - 0.5.
    let fx = x / g.dx - 0.5;
    let fy = y / g.dy - 0.5;

    let i0f = fx.floor();
    let j0f = fy.floor();
    let tx = fx - i0f;
    let ty = fy - j0f;

    let i0 = clamp_usize(i0f as isize, nx);
    let j0 = clamp_usize(j0f as isize, ny);
    let i1 = clamp_usize(i0f as isize + 1, nx);
    let j1 = clamp_usize(j0f as isize + 1, ny);

    let idx00 = field.idx(i0, j0);
    let idx10 = field.idx(i1, j0);
    let idx01 = field.idx(i0, j1);
    let idx11 = field.idx(i1, j1);

    let v00 = field.data[idx00];
    let v10 = field.data[idx10];
    let v01 = field.data[idx01];
    let v11 = field.data[idx11];

    let lerp = |a: [f64; 3], b: [f64; 3], t: f64| -> [f64; 3] {
        [
            a[0] * (1.0 - t) + b[0] * t,
            a[1] * (1.0 - t) + b[1] * t,
            a[2] * (1.0 - t) + b[2] * t,
        ]
    };

    let v0 = lerp(v00, v10, tx);
    let v1 = lerp(v01, v11, tx);
    let v = lerp(v0, v1, ty);

    normalize(v)
}
