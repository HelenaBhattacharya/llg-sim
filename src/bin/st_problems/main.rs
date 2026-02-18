// src/bin/st_problems/main.rs

mod fmr;
mod sp2;
mod sp4;
// mod sk1;

fn usage() -> ! {
    eprintln!(
        "Usage:
  cargo run --release --bin st_problems -- sp2
  cargo run --release --bin st_problems -- sp4 <a|b>
  cargo run --release --bin st_problems -- fmr
//   cargo run --release --bin st_problems -- sk1
"
    );
    std::process::exit(2);
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else { usage() };

    match cmd.as_str() {
        "sp2" => {
            // No extra args needed
            sp2::run_sp2()
        }
        "sp4" => {
            let Some(case) = args.next() else { usage() };
            let c = case.chars().next().unwrap_or('a');
            sp4::run_sp4(c)
        }
        "fmr" => {
            // No extra args needed
            fmr::run_fmr()
        }
        // "sk1" => {
        //     // No extra args needed
        //     sk1::run_sk1()
        // }
        _ => usage(),
    }
}
