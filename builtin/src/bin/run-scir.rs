use solver::utils;
use std::io::BufWriter;
use std::{fmt, hash::Hash};

include!("../scir.rs");

fn run_custom(circ_path: &str, graph_path: &str, solve_mode: &str, output_path: &str, hadamard_type: &str) {
    let circ = utils::extract_gates(circ_path, &["Pauli", "CX", "S", hadamard_type, "TComposite"]);
    let arch = CustomArch::from_file(graph_path);
    let mut res = match solve_mode {
        "--onepass" => my_solve(&circ, &arch),
        // "--sabre" => my_sabre_solve(&circ, &arch),
        // "--joint_optimize-par" => my_joint_solve_parallel(&circ, &arch),
        _ => panic!("Unrecognized solve mode"),
    };

    res.steps = res
        .steps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut new_s = s.clone();

            if i != 0 {
                new_s.counter_map = HashMap::new();
                new_s.patch_map = HashMap::new();
            }

            new_s
        })
        .collect_vec();
    res.transitions = Vec::new();

    let file = File::create(output_path).expect("msg");
    let mut writer = BufWriter::new(file);
    match serde_json::to_writer_pretty(writer, &res) {
        Ok(_) => (),
        Err(e) => panic!("Error writing compilation to stdout: {}", e),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() !=  6 {
        println!("Usage: run-scir <circuit> <graph> <hadamard_type> <output_path> --<solve-mode>");
        return;
    }
    run_custom(&args[1], &args[2], &args[5], &args[4], &args[3]);
}
