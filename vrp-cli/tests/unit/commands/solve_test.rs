use super::*;
use crate::cli::{get_app, run_subcommand};

const PRAGMATIC_PROBLEM_PATH: &str = "../examples/data/pragmatic/simple.basic.problem.json";
const PRAGMATIC_MATRIX_PATH: &str = "../examples/data/pragmatic/simple.basic.matrix.json";
const SOLOMON_PROBLEM_PATH: &str = "../examples/data/scientific/solomon/C101.25.txt";
const LILIM_PROBLEM_PATH: &str = "../examples/data/scientific/lilim/LC101.txt";

struct DummyWrite {}

impl Write for DummyWrite {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn run_solve_without_writer(matches: &ArgMatches) {
    run_solve(matches, |_| BufWriter::new(Box::new(DummyWrite {}))).unwrap();
}

fn get_solomon_matches(params: &[&str]) -> ArgMatches {
    let args = [&["solve", "solomon", SOLOMON_PROBLEM_PATH], params].concat();

    get_solve_app().try_get_matches_from(args).unwrap()
}

#[test]
fn can_solve_pragmatic_problem_with_generation_limit() {
    let args = vec!["vrp-cli", "solve", "pragmatic", PRAGMATIC_PROBLEM_PATH, "--max-generations", "1"];

    run_subcommand(get_app().try_get_matches_from(args).unwrap());
}

#[test]
fn can_solve_pragmatic_problem_with_matrix() {
    let args = vec!["vrp-cli", "solve", "pragmatic", PRAGMATIC_PROBLEM_PATH, "--matrix", PRAGMATIC_MATRIX_PATH];

    run_subcommand(get_app().try_get_matches_from(args).unwrap());
}

#[test]
fn can_solve_pragmatic_problem_with_multiple_matrices() {
    const PRAGMATIC_BASICS_PATH: &str = "../examples/data/pragmatic/basics/";
    let problem_path = format!("{PRAGMATIC_BASICS_PATH}profiles.basic.problem.json");
    let car_matrix_path = format!("{PRAGMATIC_BASICS_PATH}profiles.basic.matrix.car.json");
    let truck_matrix_path = format!("{PRAGMATIC_BASICS_PATH}profiles.basic.matrix.truck.json");

    let args = vec![
        "vrp-cli",
        "solve",
        "pragmatic",
        &problem_path,
        "--matrix",
        &car_matrix_path,
        "--matrix",
        &truck_matrix_path,
    ];

    run_subcommand(get_app().try_get_matches_from(args).unwrap());
}

#[test]
fn can_solve_lilim_problem_with_multiple_limits() {
    let args = vec!["vrp-cli", "solve", "lilim", LILIM_PROBLEM_PATH, "--max-time", "300", "--max-generations", "1"];

    run_subcommand(get_app().try_get_matches_from(args).unwrap());
}

#[test]
fn can_solve_solomon_problem_with_generation_limit() {
    run_solve_without_writer(&get_solomon_matches(&["--max-generations", "1"]));
}

#[test]
fn can_require_problem_path() {
    for format in &["pragmatic", "solomon", "lilim", "tsplib"] {
        get_solve_app().try_get_matches_from(vec!["solve", format]).unwrap_err();
    }
}

#[test]
fn can_specify_search_mode_setting() {
    for mode in &["deep", "broad"] {
        let args = vec!["solve", "pragmatic", PRAGMATIC_PROBLEM_PATH, "--search-mode", mode];
        get_solve_app().try_get_matches_from(args).unwrap();
    }
}

#[test]
fn can_specify_experimental_setting() {
    let args = vec!["solve", "pragmatic", PRAGMATIC_PROBLEM_PATH, "--experimental"];
    get_solve_app().try_get_matches_from(args).unwrap();
}

#[test]
fn can_specify_round_setting() {
    let args = vec!["solve", "solomon", SOLOMON_PROBLEM_PATH, "--round"];
    get_solve_app().try_get_matches_from(args).unwrap();
}

#[test]
fn can_specify_check_setting() {
    let args = vec!["solve", "solomon", SOLOMON_PROBLEM_PATH, "--check"];
    get_solve_app().try_get_matches_from(args).unwrap();
}

#[test]
fn can_specify_log_setting() {
    let args = vec!["solve", "solomon", SOLOMON_PROBLEM_PATH, "--log"];
    get_solve_app().try_get_matches_from(args).unwrap();
}

#[test]
fn can_specify_locations_setting() {
    let args = vec!["solve", "solomon", SOLOMON_PROBLEM_PATH, "--get-locations"];
    get_solve_app().try_get_matches_from(args).unwrap();
}

#[test]
fn can_specify_heuristic_setting() {
    for &(mode, result) in
        &[("default", Some(())), ("dynamic", Some(())), ("static", Some(())), ("ggg", None), ("multi", None)]
    {
        let args = vec!["solve", "pragmatic", PRAGMATIC_PROBLEM_PATH, "--heuristic", mode];
        assert_eq!(get_solve_app().try_get_matches_from(args).ok().map(|_| ()), result);
    }
}

#[test]
fn can_specify_parallelism() {
    for (params, result) in [
        (vec!["--parallelism", "3,1"], Ok(3_usize)),
        (vec!["--parallelism", "3"], Err("cannot parse parallelism parameter".into())),
    ] {
        let matches = get_solomon_matches(params.as_slice());

        let thread_pool_size = get_environment(&matches).map(|e| e.parallelism.thread_pool_size());

        assert_eq!(thread_pool_size, result);
    }
}

#[test]
fn can_use_init_size() {
    for (params, result) in [
        (vec!["--init-size", "1"], Ok(Some(1))),
        (vec!["--init-size", "0"], Err("init size must be an integer bigger than 0, got '0'".into())),
        (vec![], Ok(None)),
    ] {
        let matches = get_solomon_matches(params.as_slice());

        let init_size = get_init_size(&matches);

        assert_eq!(init_size, result);
    }
}

#[test]
fn can_specify_cv() {
    for (params, result) in vec![
        (vec!["--min-cv", "sample,200,0.05,true"], Ok(Some(("sample".to_string(), 200, 0.05, true)))),
        (vec!["--min-cv", "period,100,0.01,false"], Ok(Some(("period".to_string(), 100, 0.01, false)))),
        (vec!["--min-cv", "sample,200,0,tru"], Err("cannot parse min_cv parameter".into())),
        (vec!["--min-cv", "sampl,200,0,true"], Err("cannot parse min_cv parameter".into())),
        (vec!["--min-cv", "perio,200,0,true"], Err("cannot parse min_cv parameter".into())),
        (vec!["--min-cv", "200,0"], Err("cannot parse min_cv parameter".into())),
        (vec!["--min-cv", "0"], Err("cannot parse min_cv parameter".into())),
        (vec![], Ok(None)),
    ] {
        let matches = get_solomon_matches(params.as_slice());

        let min_cv = get_min_cv(&matches);

        assert_eq!(min_cv, result);
    }
}
