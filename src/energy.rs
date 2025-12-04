use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;
use crate::params::Material;

/// Very simple, *dimensionless* diagnostic energy:
/// E ~ a_ex * sum(|grad m|^2) + k_u * sum(1 - (m·u)^2)
/// (no dx*dy factor yet – this is just for debugging).
pub fn compute_total_energy(grid: &Grid2D, m: &VectorField2D, material: &Material) -> f64 {
    let nx = grid.nx;
    let ny = grid.ny;

    let a_ex = material.a_ex;
    let k_u = material.k_u;
    let u = material.easy_axis;

    if a_ex == 0.0 && k_u == 0.0 {
        return 0.0;
    }

    let mut e_ex = 0.0;
    let mut e_an = 0.0;

    for j in 0..ny {
        for i in 0..nx {
            let idx = grid.idx(i, j);
            let mij = m.data[idx];

            // Exchange with right neighbour
            if i + 1 < nx {
                let idx_r = grid.idx(i + 1, j);
                let mr = m.data[idx_r];
                let dm = [
                    mr[0] - mij[0],
                    mr[1] - mij[1],
                    mr[2] - mij[2],
                ];
                e_ex += dm[0] * dm[0] + dm[1] * dm[1] + dm[2] * dm[2];
            }

            // Exchange with neighbour above
            if j + 1 < ny {
                let idx_u = grid.idx(i, j + 1);
                let mu = m.data[idx_u];
                let dm = [
                    mu[0] - mij[0],
                    mu[1] - mij[1],
                    mu[2] - mij[2],
                ];
                e_ex += dm[0] * dm[0] + dm[1] * dm[1] + dm[2] * dm[2];
            }

            // Uniaxial anisotropy term: min at m || u
            let mdotu = mij[0] * u[0] + mij[1] * u[1] + mij[2] * u[2];
            e_an += 1.0 - mdotu * mdotu;
        }
    }

    a_ex * e_ex + k_u * e_an
}