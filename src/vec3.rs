// src/vec3.rs

/// 3D vector dot product.
#[inline]
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3D vector cross product: a × b.
#[inline]
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalise a 3D vector to unit length. If zero, return (0, 0, 1).
#[inline]
pub fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n2 = dot(v, v);
    if n2 == 0.0 {
        return [0.0, 0.0, 1.0];
    }
    let inv = 1.0 / n2.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}