// src/visualisation.rs

use crate::vector_field::VectorField2D;
use plotters::prelude::*;
use std::io;
use std::process::Command;

/// Map m_z in [-1, 1] to a blue–white–red colour.
/// -1 -> blue, 0 -> white, +1 -> red.
fn mz_to_color(mz: f64) -> RGBColor {
    // Clamp mz to [-1, 1] and map to x in [0, 1]
    let x = ((mz + 1.0) * 0.5).clamp(0.0, 1.0);

    // Simple blue–white–red:
    // x = 0   -> blue  (0, 0, 255)
    // x = 0.5 -> white (255, 255, 255)
    // x = 1   -> red   (255, 0, 0)
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

    // Size of the output image in pixels
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

    // Draw one coloured rectangle per cell
    chart.draw_series(
        (0..nx).flat_map(|i| {
            (0..ny).map(move |j| {
                let idx = field.idx(i as usize, j as usize);
                let mz = field.data[idx][2];
                let color = mz_to_color(mz);
                Rectangle::new([(i, j), (i + 1, j + 1)], color.filled())
            })
        }),
    )?;

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