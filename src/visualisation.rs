// src/visualisation.rs

use crate::vector_field::VectorField2D;
use plotters::prelude::*;
use std::io;
use std::process::Command;

/// Map m_z in [-1, 1] to a blue–white–red colour.
/// -1 -> blue, 0 -> white, +1 -> red.
fn mz_to_color(mz: f64) -> RGBColor {
    let x = ((mz + 1.0) * 0.5).clamp(0.0, 1.0);

    // blue–white–red: x=0 -> blue, x=0.5 -> white, x=1 -> red
    let r = (255.0 * x) as u8;
    let b = (255.0 * (1.0 - x)) as u8;
    let g = (255.0 * (1.0 - (2.0 * (x - 0.5).abs()))).clamp(0.0, 255.0) as u8;

    RGBColor(r, g, b)
}

/// Save the z-component of magnetisation as a PNG plot with axes and labels.
/// - x/y axes are cell indices
/// - colour encodes m_z (blue=-1, white=0, red=+1)
pub fn save_mz_plot(
    field: &VectorField2D,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = field.grid.nx as i32;
    let ny = field.grid.ny as i32;

    // First pass: find min/max m_z (for annotation only)
    let mut min_mz = f64::INFINITY;
    let mut max_mz = f64::NEG_INFINITY;
    for j in 0..ny {
        for i in 0..nx {
            let idx = field.idx(i as usize, j as usize);
            let mz = field.data[idx][2];
            if mz.is_finite() {
                min_mz = min_mz.min(mz);
                max_mz = max_mz.max(mz);
            }
        }
    }
    if !min_mz.is_finite() || !max_mz.is_finite() {
        min_mz = -1.0;
        max_mz = 1.0;
    }

    let root = BitMapBackend::new(filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .caption(
            "m_z field (blue = -1, white = 0, red = +1)",
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

    chart.draw_series((0..nx).flat_map(|i| {
        (0..ny).map(move |j| {
            let idx = field.idx(i as usize, j as usize);
            let mz = field.data[idx][2];
            let color = mz_to_color(mz);
            Rectangle::new([(i, j), (i + 1, j + 1)], color.filled())
        })
    }))?;

    // Annotation with min/max of this frame
    let annot = format!("m_z ∈ [-1, 1] | frame min={:.3}, max={:.3}", min_mz, max_mz);
    chart.draw_series(std::iter::once(Text::new(
        annot,
        (5, ny - 5),
        ("sans-serif", 15),
    )))?;

    Ok(())
}

/// Use `ffmpeg` to stitch frames into an MP4 movie.
///
/// IMPORTANT:
/// This expects a *numbered sequence* pattern like:
///   "frames/mz_%05d.png"
/// and assumes numbering starts at 0 with no gaps.
/// (main.rs will now generate frames that way.)
pub fn make_movie_with_ffmpeg(pattern_glob: &str, output: &str, fps: u32) -> io::Result<()> {
    let ffmpeg_path = "/opt/homebrew/bin/ffmpeg"; // UPDATE TO REMOVE HARDCODED PATH

    let mut cmd = Command::new(ffmpeg_path);
    cmd.arg("-y")
        .arg("-framerate")
        .arg(fps.to_string())
        // THIS is the key: must be before -i
        .arg("-pattern_type")
        .arg("glob")
        .arg("-i")
        .arg(pattern_glob)
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(output);

    let status = cmd.status()?;

    if !status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("ffmpeg failed with status {:?}", status),
        ));
    }

    Ok(())
}

// --- the rest of your plotting helpers unchanged ---

pub fn save_energy_components_plot(
    times: &[f64],
    e_ex: &[f64],
    e_an: &[f64],
    e_zee: &[f64],
    e_dmi: &[f64],
    e_demag: &[f64],
    e_tot: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if times.is_empty() {
        return Ok(());
    }

    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let t_min = *times.first().unwrap();
    let t_max = *times.last().unwrap();

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for &e in e_ex
        .iter()
        .chain(e_an)
        .chain(e_zee)
        .chain(e_dmi)
        .chain(e_demag)
        .chain(e_tot)
    {
        if e.is_finite() {
            y_min = y_min.min(e);
            y_max = y_max.max(e);
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    } else if (y_max - y_min).abs() < 1e-30 {
        let delta = if y_max.abs() < 1e-30 { 1.0 } else { 0.1 * y_max.abs() };
        y_min -= delta;
        y_max += delta;
    } else {
        let margin = 0.1 * (y_max - y_min);
        y_min -= margin;
        y_max += margin;
    }

    // Scale label like before
    let magnitude = y_max.abs().max(y_min.abs());
    let (scale, y_label): (f64, String) = if magnitude > 0.0 {
        let exp = magnitude.log10().floor() as i32;
        let scale = 10f64.powi(exp);
        if exp == 0 {
            (1.0, "Energy (J)".to_string())
        } else {
            (scale, format!("Energy (J × 10^{})", exp))
        }
    } else {
        (1.0, "Energy (J)".to_string())
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
        .x_desc("time (s)")
        .y_desc(y_label)
        .x_labels(10)
        .y_labels(10)
        .x_label_formatter(&|x| format!("{:.2e}", x))
        .label_style(("sans-serif", 16))
        .axis_desc_style(("sans-serif", 18))
        .draw()?;

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
            times.iter().zip(e_dmi.iter()).map(|(&t, &e)| (t, e / scale)),
            &MAGENTA,
        ))?
        .label("DMI")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(e_demag.iter()).map(|(&t, &e)| (t, e / scale)),
            &CYAN,
        ))?
        .label("Demag")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &CYAN));

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

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &de in &residuals {
        if de.is_finite() {
            y_min = y_min.min(de);
            y_max = y_max.max(de);
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    } else if (y_max - y_min).abs() < 1e-30 {
        let delta = if y_max.abs() < 1e-30 { 1.0 } else { 0.1 * y_max.abs() };
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
        .y_desc("ΔE(t) = E(t) − E(0) (J)")
        .x_label_formatter(&|x| format!("{:.2e}", x))
        .draw()?;

    chart.draw_series(LineSeries::new(
        times.iter().zip(residuals.iter()).map(|(&t, &de)| (t, de)),
        &BLACK,
    ))?;

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
        .x_label_formatter(&|x| format!("{:.2e}", x))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(mx.iter()).map(|(&t, &v)| (t, v)),
            &RED,
        ))?
        .label("m_x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(my.iter()).map(|(&t, &v)| (t, v)),
            &GREEN,
        ))?
        .label("m_y")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

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

pub fn save_m_avg_zoom_plot(
    times: &[f64],
    mx: &[f64],
    my: &[f64],
    mz: &[f64],
    t_max: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut t_zoom: Vec<f64> = Vec::new();
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

    if t_zoom.len() < 2 {
        return Ok(());
    }

    let t_min = *t_zoom.first().unwrap();
    let t_max = *t_zoom.last().unwrap();

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &v in mx_zoom.iter().chain(my_zoom.iter()).chain(mz_zoom.iter()) {
        if v.is_finite() {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }

    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    } else if (y_max - y_min).abs() < 1e-9 {
        let delta = if y_max.abs() < 1e-9 {
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
        .caption("Average magnetisation vs time (zoomed)", ("sans-serif", 30))
        .set_left_and_bottom_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("average magnetisation component")
        .x_label_formatter(&|x| format!("{:.2e}", x))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            t_zoom.iter().zip(mx_zoom.iter()).map(|(&t, &v)| (t, v)),
            &RED,
        ))?
        .label("m_x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            t_zoom.iter().zip(my_zoom.iter()).map(|(&t, &v)| (t, v)),
            &GREEN,
        ))?
        .label("m_y")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

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

pub fn save_dt_vs_time_plot(
    times: &[f64],
    dts: &[f64],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if times.len() < 2 || times.len() != dts.len() {
        return Ok(());
    }

    let t_min = *times.first().unwrap();
    let t_max = *times.last().unwrap();

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &v in dts {
        if v.is_finite() {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    } else {
        let margin = 0.1 * (y_max - y_min).abs().max(1e-30);
        y_min -= margin;
        y_max += margin;
        if y_min < 0.0 {
            y_min = 0.0;
        }
    }

    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Adaptive timestep dt vs time", ("sans-serif", 30))
        .set_left_and_bottom_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("dt (s)")
        .x_label_formatter(&|x| format!("{:.2e}", x))
        .draw()?;

    chart.draw_series(LineSeries::new(
        times.iter().zip(dts.iter()).map(|(&t, &dt)| (t, dt)),
        &BLACK,
    ))?;

    root.present()?;
    Ok(())
}

pub fn save_eps_vs_time_plot(
    times: &[f64],
    eps: &[f64],
    max_err: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if times.len() < 2 || times.len() != eps.len() {
        return Ok(());
    }

    let t_min = *times.first().unwrap();
    let t_max = *times.last().unwrap();

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &v in eps {
        if v.is_finite() {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    } else {
        let margin = 0.1 * (y_max - y_min).abs().max(1e-30);
        y_min = (y_min - margin).max(0.0);
        y_max += margin;
    }
    // ensure max_err visible
    y_max = y_max.max(1.2 * max_err);

    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("RK45 error estimate eps vs time", ("sans-serif", 30))
        .set_left_and_bottom_label_area_size(60)
        .build_cartesian_2d(t_min..t_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("eps")
        .x_label_formatter(&|x| format!("{:.2e}", x))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            times.iter().zip(eps.iter()).map(|(&t, &e)| (t, e)),
            &RED,
        ))?
        .label("eps")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Horizontal line: max_err
    chart
        .draw_series(LineSeries::new(vec![(t_min, max_err), (t_max, max_err)], &BLACK))?
        .label("max_err")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    Ok(())
}
