// src/visualisation.rs

use crate::vector_field::VectorField2D;
use plotters::prelude::*;
use std::io;
use std::process::Command;

/// Map m_z to a blue–white–red colour using a *local* min/max,
/// so small variations are still visible.
///
/// min_mz maps to blue, max_mz maps to red, midpoint to white.
fn mz_to_color(mz: f64, min_mz: f64, max_mz: f64) -> RGBColor {
    // Protect against min ≈ max (e.g. perfectly uniform state)
    let mut lo = min_mz;
    let mut hi = max_mz;
    if !lo.is_finite() || !hi.is_finite() || (hi - lo).abs() < 1e-9 {
        lo = -1.0;
        hi = 1.0;
    }

    let x = ((mz - lo) / (hi - lo)).clamp(0.0, 1.0);

    // blue–white–red: x=0 -> blue, x=0.5 -> white, x=1 -> red
    let r = (255.0 * x) as u8;
    let b = (255.0 * (1.0 - x)) as u8;
    let g = (255.0 * (1.0 - (2.0 * (x - 0.5).abs()))).clamp(0.0, 255.0) as u8;

    RGBColor(r, g, b)
}

/// Save the z-component of magnetisation as a PNG plot with axes and labels.
/// - x/y axes are cell indices
/// - colour encodes m_z (blue ≈ min, white ≈ mid, red ≈ max)
pub fn save_mz_plot(
    field: &VectorField2D,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = field.grid.nx as i32;
    let ny = field.grid.ny as i32;

    // First pass: find min/max m_z over this frame
    let mut min_mz = f64::INFINITY;
    let mut max_mz = f64::NEG_INFINITY;
    for j in 0..ny {
        for i in 0..nx {
            let idx = field.idx(i as usize, j as usize);
            let mz = field.data[idx][2];
            if mz.is_finite() {
                if mz < min_mz {
                    min_mz = mz;
                }
                if mz > max_mz {
                    max_mz = mz;
                }
            }
        }
    }
    if !min_mz.is_finite() || !max_mz.is_finite() {
        min_mz = -1.0;
        max_mz = 1.0;
    }

    // Size of the output image in pixels
    let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .caption(
            "m_z field (blue = min, white = mid, red = max)",
            ("sans-serif", 20),
        )
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..nx, 0..ny)?;

    chart
        .configure_mesh()
        .x_desc("x (cell index)")
        .y_desc("y (cell index)")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    // Draw one coloured rectangle per cell
    chart.draw_series(
        (0..nx).flat_map(|i| {
            (0..ny).map(move |j| {
                let idx = field.idx(i as usize, j as usize);
                let mz = field.data[idx][2];
                let color = mz_to_color(mz, min_mz, max_mz);
                Rectangle::new([(i, j), (i + 1, j + 1)], color.filled())
            })
        }),
    )?;

    // Small annotation inside the plot
    chart.draw_series(std::iter::once(
        Text::new(
            "m_z \u{2208} [-1, 1]",      // "∈"
            (5, ny - 5),
            ("sans-serif", 15),
        )
    ))?;

    Ok(())
}

/// Use `ffmpeg` to stitch all frames/mz_*.png into an MP4 movie.
/// Assumes filenames like frames/mz_0000.png, mz_0010.png, ...
pub fn make_movie_with_ffmpeg(
    pattern: &str,
    output: &str,
    fps: u32,
) -> io::Result<()> {
    // Use the full path to ffmpeg so we don't depend on PATH
    let ffmpeg_path = "/opt/homebrew/bin/ffmpeg";  // <- update if `which ffmpeg` gives a different path

    let status = Command::new(ffmpeg_path)
        .args(&[
            "-y",                          // overwrite output if it exists
            "-framerate", &fps.to_string(),
            "-pattern_type", "glob",
            "-i", pattern,                 // e.g. "frames/mz_*.png"
            "-pix_fmt", "yuv420p",
            output,                        // e.g. "mz_evolution.mp4"
        ])
        .status()?;

    if !status.success() {
        eprintln!("ffmpeg exited with status {:?}", status);
    }

    Ok(())
}

/// Plot exchange, anisotropy, Zeeman and total energy versus time.
pub fn save_energy_components_plot(
    times: &[f64],
    e_ex: &[f64],
    e_an: &[f64],
    e_zee: &[f64],
    e_tot: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if times.is_empty() {
        return Ok(()); // nothing to plot
    }

    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let t_min = *times.first().unwrap();
    let t_max = *times.last().unwrap();

    // --- find global y-range over all components (unscaled) ---
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for &e in e_ex.iter().chain(e_an).chain(e_zee).chain(e_tot) {
        if e.is_finite() {
            if e < y_min {
                y_min = e;
            }
            if e > y_max {
                y_max = e;
            }
        }
    }

    // Handle pathological case (all zero or NaN)
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    } else if (y_max - y_min).abs() < 1e-30 {
        // all values essentially identical; broaden the window
        let delta = if y_max.abs() < 1e-30 {
            1.0
        } else {
            0.1 * y_max.abs()
        };
        y_min -= delta;
        y_max += delta;
    } else {
        // add a 10% margin around the data range
        let margin = 0.1 * (y_max - y_min);
        y_min -= margin;
        y_max += margin;
    }

    // ---------- choose a 10^n scaling for nicer axes ----------
    let magnitude = y_max.abs().max(y_min.abs());
    let (scale, y_label): (f64, String) = if magnitude > 0.0 {
        let exp = magnitude.log10().floor() as i32;
        let scale = 10f64.powi(exp);
        if exp == 0 {
            (1.0, "Energy (arb. units)".to_string())
        } else {
            (
                scale,
                format!("Energy (arb. units × 10^{})", exp),
            )
        }
    } else {
        (1.0, "Energy (arb. units)".to_string())
    };

    let y_min_scaled = y_min / scale;
    let y_max_scaled = y_max / scale;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Energy components vs time", ("sans-serif", 30))
        .set_left_and_bottom_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, y_min_scaled..y_max_scaled)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")           // or "time (arb. units)" if you prefer
        .y_desc(y_label)
        .x_labels(10)
        .y_labels(10)
        .label_style(("sans-serif", 16))
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

    // ---- draw each component + legend entry, with scaling ----
    chart
        .draw_series(LineSeries::new(
            times.iter().zip(e_ex.iter()).map(|(&t, &e)| (t, e / scale)),
            &RED,
        ))?
        .label("Exchange")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(e_an.iter()).map(|(&t, &e)| (t, e / scale)),
            &BLUE,
        ))?
        .label("Anisotropy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(e_zee.iter()).map(|(&t, &e)| (t, e / scale)),
            &GREEN,
        ))?
        .label("Zeeman")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(e_tot.iter()).map(|(&t, &e)| (t, e / scale)),
            &BLACK,
        ))?
        .label("Total")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}




pub fn save_m_avg_plot(
    times: &[f64],
    mx: &[f64],
    my: &[f64],
    mz: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if times.is_empty() {
        return Ok(());
    }

    // Assume m is a unit vector, so components lie in [-1, 1].
    // Give a small margin so the lines don’t touch the frame.
    let y_min = -1.1;
    let y_max = 1.1;

    let t_min = *times.first().unwrap();
    let t_max = *times.last().unwrap();

    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Average magnetisation vs time", ("sans-serif", 30))
        .set_left_and_bottom_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("average magnetisation component")
        .draw()?;

    // mx(t) in red
    chart
        .draw_series(LineSeries::new(
            times.iter().zip(mx.iter()).map(|(&t, &v)| (t, v)),
            &RED,
        ))?
        .label("m_x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // my(t) in green
    chart
        .draw_series(LineSeries::new(
            times.iter().zip(my.iter()).map(|(&t, &v)| (t, v)),
            &GREEN,
        ))?
        .label("m_y")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // mz(t) in blue
    chart
        .draw_series(LineSeries::new(
            times.iter().zip(mz.iter()).map(|(&t, &v)| (t, v)),
            &BLUE,
        ))?
        .label("m_z")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn save_energy_residual_plot(
    times: &[f64],
    e_tot: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if times.is_empty() || e_tot.is_empty() {
        return Ok(());
    }

    let e0 = e_tot[0];
    let residuals: Vec<f64> = e_tot.iter().map(|&e| e - e0).collect();

    let t_min = *times.first().unwrap();
    let t_max = *times.last().unwrap();

    // Auto-scale y-range using the residuals
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for &de in &residuals {
        if de.is_finite() {
            if de < y_min {
                y_min = de;
            }
            if de > y_max {
                y_max = de;
            }
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    } else if (y_max - y_min).abs() < 1e-30 {
        let delta = if y_max.abs() < 1e-30 {
            1.0
        } else {
            0.1 * y_max.abs()
        };
        y_min -= delta;
        y_max += delta;
    } else {
        let margin = 0.1 * (y_max - y_min);
        y_min -= margin;
        y_max += margin;
    }

    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Total energy residual vs time", ("sans-serif", 30))
        .set_left_and_bottom_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("ΔE(t) = E(t) − E(0) (arb. units)")
        .draw()?;

    chart.draw_series(LineSeries::new(
        times
            .iter()
            .zip(residuals.iter())
            .map(|(&t, &de)| (t, de)),
        &BLACK,
    ))?;

    root.present()?;
    Ok(())
}

pub fn save_m_avg_zoom_plot(
    times: &[f64],
    mx: &[f64],
    my: &[f64],
    mz: &[f64],
    t_max: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Collect only points with t <= t_max
    let mut t_zoom:  Vec<f64> = Vec::new();
    let mut mx_zoom: Vec<f64> = Vec::new();
    let mut my_zoom: Vec<f64> = Vec::new();
    let mut mz_zoom: Vec<f64> = Vec::new();

    for (i, &t) in times.iter().enumerate() {
        if t <= t_max {
            t_zoom.push(t);
            mx_zoom.push(mx[i]);
            my_zoom.push(my[i]);
            mz_zoom.push(mz[i]);
        }
    }

    // Nothing (or only one point) to plot
    if t_zoom.len() < 2 {
        return Ok(());
    }

    // X-range for the zoom
    let t_min = *t_zoom.first().unwrap();
    let t_max = *t_zoom.last().unwrap();

    // Y-range: find min/max across all three components
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for &v in mx_zoom.iter().chain(my_zoom.iter()).chain(mz_zoom.iter()) {
        if v.is_finite() {
            if v < y_min { y_min = v; }
            if v > y_max { y_max = v; }
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max =  1.0;
    } else if (y_max - y_min).abs() < 1e-9 {
        // Avoid zero-height range
        let delta = if y_max.abs() < 1e-9 { 1.0 } else { 0.1 * y_max.abs() };
        y_min -= delta;
        y_max += delta;
    } else {
        // 10% margin
        let margin = 0.1 * (y_max - y_min);
        y_min -= margin;
        y_max += margin;
    }

    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Average magnetisation vs time (zoomed)", ("sans-serif", 30))
        .set_left_and_bottom_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("average magnetisation component")
        .draw()?;

    // m_x
    chart
        .draw_series(LineSeries::new(
            t_zoom.iter().zip(mx_zoom.iter()).map(|(&t, &v)| (t, v)),
            &RED,
        ))?
        .label("m_x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // m_y
    chart
        .draw_series(LineSeries::new(
            t_zoom.iter().zip(my_zoom.iter()).map(|(&t, &v)| (t, v)),
            &GREEN,
        ))?
        .label("m_y")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // m_z
    chart
        .draw_series(LineSeries::new(
            t_zoom.iter().zip(mz_zoom.iter()).map(|(&t, &v)| (t, v)),
            &BLUE,
        ))?
        .label("m_z")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}






// // src/visualisation.rs

// use crate::vector_field::VectorField2D;
// use plotters::prelude::*;
// use std::io;
// use std::process::Command;

// /// Map m_z in [-1, 1] to a blue–white–red colour.
// /// -1 -> blue, 0 -> white, +1 -> red.
// fn mz_to_color(mz: f64) -> RGBColor {
//     // Clamp mz to [-1, 1] and map to x in [0, 1]
//     let x = ((mz + 1.0) * 0.5).clamp(0.0, 1.0);

//     // Simple blue–white–red:
//     // x = 0   -> blue  (0, 0, 255)
//     // x = 0.5 -> white (255, 255, 255)
//     // x = 1   -> red   (255, 0, 0)
//     let r = (255.0 * x) as u8;
//     let b = (255.0 * (1.0 - x)) as u8;
//     let g = (255.0 * (1.0 - (2.0 * (x - 0.5).abs()))).clamp(0.0, 255.0) as u8;

//     RGBColor(r, g, b)
// }

// /// Save the z-component of magnetisation as a PNG plot with axes and labels.
// /// - x/y axes are cell indices
// /// - colour encodes m_z (blue=-1, white=0, red=+1)
// pub fn save_mz_plot(
//     field: &VectorField2D,
//     filename: &str,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     let nx = field.grid.nx as i32;
//     let ny = field.grid.ny as i32;

//     // Size of the output image in pixels
//     let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
//     root.fill(&WHITE)?;

//     let mut chart = ChartBuilder::on(&root)
//         .margin(40)
//         .caption(
//             "m_z field (blue = -1, white = 0, red = +1)",
//             ("sans-serif", 20),
//         )
//         .x_label_area_size(40)
//         .y_label_area_size(40)
//         .build_cartesian_2d(0..nx, 0..ny)?;

//     chart
//         .configure_mesh()
//         .x_desc("x (cell index)")
//         .y_desc("y (cell index)")
//         .axis_desc_style(("sans-serif", 15))
//         .draw()?;

//     // Draw one coloured rectangle per cell
//     chart.draw_series(
//         (0..nx).flat_map(|i| {
//             (0..ny).map(move |j| {
//                 let idx = field.idx(i as usize, j as usize);
//                 let mz = field.data[idx][2];
//                 let color = mz_to_color(mz);
//                 Rectangle::new([(i, j), (i + 1, j + 1)], color.filled())
//             })
//         }),
//     )?;

//     Ok(())
// }

// /// Use `ffmpeg` to stitch all frames/mz_*.png into an MP4 movie.
// /// Assumes filenames like frames/mz_0000.png, mz_0010.png, ...
// pub fn make_movie_with_ffmpeg(
//     pattern: &str,
//     output: &str,
//     fps: u32,
// ) -> io::Result<()> {
//     // Use the full path to ffmpeg so we don't depend on PATH
//     let ffmpeg_path = "/opt/homebrew/bin/ffmpeg";  // <- update if `which ffmpeg` gives a different path

//     let status = Command::new(ffmpeg_path)
//         .args(&[
//             "-y",                          // overwrite output if it exists
//             "-framerate", &fps.to_string(),
//             "-pattern_type", "glob",
//             "-i", pattern,                 // e.g. "frames/mz_*.png"
//             "-pix_fmt", "yuv420p",
//             output,                        // e.g. "mz_evolution.mp4"
//         ])
//         .status()?;

//     if !status.success() {
//         eprintln!("ffmpeg exited with status {:?}", status);
//     }

//     Ok(())
// }



// pub fn save_energy_components_plot(
//     times: &[f64],
//     e_ex: &[f64],
//     e_an: &[f64],
//     e_zee: &[f64],
//     e_tot: &[f64],
//     filename: &str,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     if times.is_empty() {
//         return Ok(()); // nothing to plot
//     }

//     let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
//     root.fill(&WHITE)?;

//     let t_min = *times.first().unwrap();
//     let t_max = *times.last().unwrap();

//     // --- find global y-range over all components ---
//     let mut y_min = f64::INFINITY;
//     let mut y_max = f64::NEG_INFINITY;

//     for &e in e_ex.iter().chain(e_an).chain(e_zee).chain(e_tot) {
//         if e.is_finite() {
//             if e < y_min {
//                 y_min = e;
//             }
//             if e > y_max {
//                 y_max = e;
//             }
//         }
//     }

//     // Handle pathological case (all zero or NaN)
//     if !y_min.is_finite() || !y_max.is_finite() {
//         y_min = -1.0;
//         y_max = 1.0;
//     } else if (y_max - y_min).abs() < 1e-30 {
//         // all values essentially identical; broaden the window
//         let delta = if y_max.abs() < 1e-30 {
//             1.0
//         } else {
//             0.1 * y_max.abs()
//         };
//         y_min -= delta;
//         y_max += delta;
//     } else {
//         // add a 10% margin around the data range
//         let margin = 0.1 * (y_max - y_min);
//         y_min -= margin;
//         y_max += margin;
//     }

//     let mut chart = ChartBuilder::on(&root)
//         .margin(20)
//         .caption("Energy components vs time", ("sans-serif", 30))
//         .set_left_and_bottom_label_area_size(60)
//         .build_cartesian_2d(t_min..t_max, y_min..y_max)?;

//     chart
//         .configure_mesh()
//         .x_desc("time (arb. units)")
//         .y_desc("Energy (toy units)")
//         .draw()?;

//     // ---- draw each component + legend entry explicitly ----
//     chart
//         .draw_series(LineSeries::new(
//             times.iter().zip(e_ex.iter()).map(|(&t, &e)| (t, e)),
//             &RED,
//         ))?
//         .label("Exchange")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

//     chart
//         .draw_series(LineSeries::new(
//             times.iter().zip(e_an.iter()).map(|(&t, &e)| (t, e)),
//             &BLUE,
//         ))?
//         .label("Anisotropy")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

//     chart
//         .draw_series(LineSeries::new(
//             times.iter().zip(e_zee.iter()).map(|(&t, &e)| (t, e)),
//             &GREEN,
//         ))?
//         .label("Zeeman")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

//     chart
//         .draw_series(LineSeries::new(
//             times.iter().zip(e_tot.iter()).map(|(&t, &e)| (t, e)),
//             &BLACK,
//         ))?
//         .label("Total")
//         .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

//     chart
//         .configure_series_labels()
//         .border_style(&BLACK)
//         .background_style(&WHITE.mix(0.8))
//         .draw()?;

//     root.present()?;
//     Ok(())
// }









// pub fn save_energy_plot(
//     times: &[f64],
//     energies: &[f64],
//     filename: &str,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     use plotters::prelude::*;

//     if times.is_empty() || energies.is_empty() {
//         return Ok(()); // nothing to plot
//     }

//     let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
//     root.fill(&WHITE)?;

//     let x_min = times[0];
//     let x_max = *times.last().unwrap();

//     let mut y_min = energies[0];
//     let mut y_max = energies[0];
//     for &e in energies.iter() {
//         if e < y_min {
//             y_min = e;
//         }
//         if e > y_max {
//             y_max = e;
//         }
//     }

//     // Add a small margin so the curve is not glued to the frame
//     let margin = 0.05 * (y_max - y_min).abs().max(1.0);
//     y_min -= margin;
//     y_max += margin;

//     let mut chart = ChartBuilder::on(&root)
//         .margin(40)
//         .caption("Total energy vs time", ("sans-serif", 20))
//         .x_label_area_size(40)
//         .y_label_area_size(60)
//         .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

//     chart
//         .configure_mesh()
//         .x_desc("time (arb. units)")
//         .y_desc("E_tot (toy units)")
//         .axis_desc_style(("sans-serif", 15))
//         .draw()?;

//     chart.draw_series(LineSeries::new(
//         times.iter().cloned().zip(energies.iter().cloned()),
//         &RED,
//     ))?;

//     Ok(())
// }


// // src/visualisation.rs

// use crate::vector_field::VectorField2D;
// use plotters::prelude::*;

// /// Map m_z in [-1, 1] to a blue–white–red colour.
// /// -1 -> blue, 0 -> white, +1 -> red.
// fn mz_to_color(mz: f64) -> RGBColor {
//     // Clamp mz to [-1, 1] and map to x in [0, 1]
//     let x = ((mz + 1.0) * 0.5).clamp(0.0, 1.0);

//     // Simple blue–white–red:
//     // x = 0   -> blue  (0, 0, 255)
//     // x = 0.5 -> white (255, 255, 255)
//     // x = 1   -> red   (255, 0, 0)
//     let r = (255.0 * x) as u8;
//     let b = (255.0 * (1.0 - x)) as u8;
//     let g = (255.0 * (1.0 - (2.0 * (x - 0.5).abs()))).clamp(0.0, 255.0) as u8;

//     RGBColor(r, g, b)
// }

// /// Save the z-component of magnetisation as a PNG plot with axes and labels.
// /// - x/y axes are cell indices
// /// - colour encodes m_z (blue=-1, white=0, red=+1)
// pub fn save_mz_plot(
//     field: &VectorField2D,
//     filename: &str,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     let nx = field.grid.nx as i32;
//     let ny = field.grid.ny as i32;

//     // Size of the output image in pixels
//     let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
//     root.fill(&WHITE)?;

//     let mut chart = ChartBuilder::on(&root)
//         .margin(40)
//         .caption(
//             "m_z field (blue = -1, white = 0, red = +1)",
//             ("sans-serif", 20),
//         )
//         .x_label_area_size(40)
//         .y_label_area_size(40)
//         .build_cartesian_2d(0..nx, 0..ny)?;

//     chart
//         .configure_mesh()
//         .x_desc("x (cell index)")
//         .y_desc("y (cell index)")
//         .axis_desc_style(("sans-serif", 15))
//         .draw()?;

//     // Draw one coloured rectangle per cell
//     chart.draw_series(
//         (0..nx).flat_map(|i| {
//             (0..ny).map(move |j| {
//                 let idx = field.idx(i as usize, j as usize);
//                 let mz = field.data[idx][2];
//                 let color = mz_to_color(mz);
//                 Rectangle::new([(i, j), (i + 1, j + 1)], color.filled())
//             })
//         }),
//     )?;

//     Ok(())
// }