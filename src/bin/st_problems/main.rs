// src/bin/st_problems/main.rs
mod sp4;

fn usage() -> ! {
    eprintln!(
        "Usage:
  cargo run --release --bin st_problems -- sp4 <a|b>

Outputs:
  runs/st_problems/sp4/sp4a_rust/table.csv
  runs/st_problems/sp4/sp4b_rust/table.csv
"
    );
    std::process::exit(2);
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);

    let Some(cmd) = args.next() else { usage() };
    match cmd.as_str() {
        "sp4" => {
            let Some(case) = args.next() else { usage() };
            let c = case.chars().next().unwrap_or('a');
            sp4::run_sp4(c)
        }
        _ => usage(),
    }
}