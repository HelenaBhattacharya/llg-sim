// src/bin/macrospin_convergence.rs
//
// Integrator convergence-order benchmark (cf. MuMax3 paper Fig. 10).
//
// Pick N steps, set dt = T_period / N exactly, compare to m(0).
//
// Methods: Euler (1st), RK23/Bogacki-Shampine (3rd), RK4 (4th), RK45/DP5 (5th)
//
// Run:  cargo run --release --bin macrospin_convergence

use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

fn llg(m:[f64;3], b:[f64;3], g:f64) -> [f64;3] {
    let c = [m[1]*b[2]-m[2]*b[1], m[2]*b[0]-m[0]*b[2], m[0]*b[1]-m[1]*b[0]];
    [-g*c[0], -g*c[1], -g*c[2]]
}
fn norm(v:[f64;3]) -> [f64;3] {
    let n=(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt(); [v[0]/n,v[1]/n,v[2]/n]
}
fn add(a:[f64;3],s:f64,b:[f64;3])->[f64;3]{[a[0]+s*b[0],a[1]+s*b[1],a[2]+s*b[2]]}
fn verr(a:[f64;3],b:[f64;3])->f64{((a[0]-b[0]).powi(2)+(a[1]-b[1]).powi(2)+(a[2]-b[2]).powi(2)).sqrt()}

// Euler (order 1)
fn euler(m:[f64;3],b:[f64;3],g:f64,dt:f64)->[f64;3]{norm(add(m,dt,llg(m,b,g)))}

// Bogacki-Shampine RK23 (order 3) — 3rd-order solution
fn rk23(m:[f64;3],b:[f64;3],g:f64,dt:f64)->[f64;3]{
    let k1=llg(m,b,g);
    let k2=llg(norm(add(m,0.5*dt,k1)),b,g);
    let k3=llg(norm(add(m,0.75*dt,k2)),b,g);
    // 3rd order solution: y_{n+1} = y_n + dt*(2/9 k1 + 1/3 k2 + 4/9 k3)
    norm([m[0]+dt*(2.0/9.0*k1[0]+1.0/3.0*k2[0]+4.0/9.0*k3[0]),
          m[1]+dt*(2.0/9.0*k1[1]+1.0/3.0*k2[1]+4.0/9.0*k3[1]),
          m[2]+dt*(2.0/9.0*k1[2]+1.0/3.0*k2[2]+4.0/9.0*k3[2])])
}

// RK4 (order 4)
fn rk4(m:[f64;3],b:[f64;3],g:f64,dt:f64)->[f64;3]{
    let k1=llg(m,b,g);
    let k2=llg(norm(add(m,0.5*dt,k1)),b,g);
    let k3=llg(norm(add(m,0.5*dt,k2)),b,g);
    let k4=llg(norm(add(m,dt,k3)),b,g);
    norm([m[0]+dt/6.0*(k1[0]+2.0*k2[0]+2.0*k3[0]+k4[0]),
          m[1]+dt/6.0*(k1[1]+2.0*k2[1]+2.0*k3[1]+k4[1]),
          m[2]+dt/6.0*(k1[2]+2.0*k2[2]+2.0*k3[2]+k4[2])])
}

// Dormand-Prince RK45 (order 5)
fn dp5(m:[f64;3],b:[f64;3],g:f64,dt:f64)->[f64;3]{
    let k1=llg(m,b,g);
    let k2=llg(norm(add(m,dt/5.0,k1)),b,g);
    let s3=norm([m[0]+dt*(3.0/40.0*k1[0]+9.0/40.0*k2[0]),m[1]+dt*(3.0/40.0*k1[1]+9.0/40.0*k2[1]),m[2]+dt*(3.0/40.0*k1[2]+9.0/40.0*k2[2])]);
    let k3=llg(s3,b,g);
    let s4=norm([m[0]+dt*(44.0/45.0*k1[0]-56.0/15.0*k2[0]+32.0/9.0*k3[0]),m[1]+dt*(44.0/45.0*k1[1]-56.0/15.0*k2[1]+32.0/9.0*k3[1]),m[2]+dt*(44.0/45.0*k1[2]-56.0/15.0*k2[2]+32.0/9.0*k3[2])]);
    let k4=llg(s4,b,g);
    let s5=norm([m[0]+dt*(19372.0/6561.0*k1[0]-25360.0/2187.0*k2[0]+64448.0/6561.0*k3[0]-212.0/729.0*k4[0]),m[1]+dt*(19372.0/6561.0*k1[1]-25360.0/2187.0*k2[1]+64448.0/6561.0*k3[1]-212.0/729.0*k4[1]),m[2]+dt*(19372.0/6561.0*k1[2]-25360.0/2187.0*k2[2]+64448.0/6561.0*k3[2]-212.0/729.0*k4[2])]);
    let k5=llg(s5,b,g);
    let s6=norm([m[0]+dt*(9017.0/3168.0*k1[0]-355.0/33.0*k2[0]+46732.0/5247.0*k3[0]+49.0/176.0*k4[0]-5103.0/18656.0*k5[0]),m[1]+dt*(9017.0/3168.0*k1[1]-355.0/33.0*k2[1]+46732.0/5247.0*k3[1]+49.0/176.0*k4[1]-5103.0/18656.0*k5[1]),m[2]+dt*(9017.0/3168.0*k1[2]-355.0/33.0*k2[2]+46732.0/5247.0*k3[2]+49.0/176.0*k4[2]-5103.0/18656.0*k5[2])]);
    let k6=llg(s6,b,g);
    norm([m[0]+dt*(35.0/384.0*k1[0]+500.0/1113.0*k3[0]+125.0/192.0*k4[0]-2187.0/6784.0*k5[0]+11.0/84.0*k6[0]),
          m[1]+dt*(35.0/384.0*k1[1]+500.0/1113.0*k3[1]+125.0/192.0*k4[1]-2187.0/6784.0*k5[1]+11.0/84.0*k6[1]),
          m[2]+dt*(35.0/384.0*k1[2]+500.0/1113.0*k3[2]+125.0/192.0*k4[2]-2187.0/6784.0*k5[2]+11.0/84.0*k6[2])])
}

fn main() -> std::io::Result<()> {
    let gamma: f64 = 1.760_859_630_23e11;
    let b0: f64 = 0.1;
    let b: [f64;3] = [0.0, 0.0, b0];
    let theta: f64 = 5.0_f64.to_radians();
    let m0: [f64;3] = [theta.sin(), 0.0, theta.cos()];
    let t_period = 2.0 * PI / (gamma * b0);

    println!("=== Integrator convergence (cf. MuMax3 Fig. 10) ===");
    println!("B = {} T,  T = {:.6e} s", b0, t_period);

    let out_dir = Path::new("out").join("macrospin_convergence");
    create_dir_all(&out_dir)?;
    let file = File::create(out_dir.join("convergence.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "dt,error_euler,error_rk23,error_rk4,error_rk45")?;

    let n_pts: usize = 50;
    for i in 0..n_pts {
        let log_n = 1.0 + 5.0 * (i as f64) / ((n_pts - 1) as f64);
        let n = 10.0_f64.powf(log_n).round() as usize;
        if n < 4 { continue; }
        let dt = t_period / (n as f64);

        let mut me=m0; for _ in 0..n { me=euler(me,b,gamma,dt); }
        let mut m23=m0; for _ in 0..n { m23=rk23(m23,b,gamma,dt); }
        let mut m4=m0; for _ in 0..n { m4=rk4(m4,b,gamma,dt); }
        let mut m5=m0; for _ in 0..n { m5=dp5(m5,b,gamma,dt); }

        let (ee,e23,e4,e5) = (verr(me,m0),verr(m23,m0),verr(m4,m0),verr(m5,m0));
        writeln!(w, "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}", dt,ee,e23,e4,e5)?;
        println!("dt={:.3e} N={:>8} | E={:.2e} RK23={:.2e} RK4={:.2e} DP5={:.2e}",
                 dt,n,ee,e23,e4,e5);
    }
    println!("\nWrote to {:?}", out_dir.join("convergence.csv"));
    Ok(())
}