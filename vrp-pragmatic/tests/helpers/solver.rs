use crate::checker::CheckerContext;
use crate::format::problem::{Matrix, PragmaticProblem, Problem};
use crate::format::solution::{create_solution, Solution};
use std::cmp::Ordering::Less;
use std::sync::Arc;
use vrp_core::construction::heuristics::InsertionContext;
use vrp_core::models::Problem as CoreProblem;
use vrp_core::models::Solution as CoreSolution;
use vrp_core::solver::mutation::{Recreate, RecreateWithCheapest};
use vrp_core::solver::population::Elitism;
use vrp_core::solver::Builder;
use vrp_core::solver::RefinementContext;
use vrp_core::utils::DefaultRandom;

/// Runs solver with cheapest insertion heuristic.
pub fn solve_with_cheapest_insertion(problem: Problem, matrices: Option<Vec<Matrix>>) -> Solution {
    get_core_solution(problem, matrices, true, |problem: Arc<CoreProblem>| {
        let random = Arc::new(DefaultRandom::default());
        let population = Box::new(Elitism::new_with_defaults(problem.clone(), random.clone()));
        let mut refinement_ctx = RefinementContext::new(problem.clone(), population, None);

        RecreateWithCheapest::default()
            .run(&mut refinement_ctx, InsertionContext::new(problem.clone(), random))
            .solution
            .to_solution(problem.extras.clone())
    })
}

/// Runs solver with default metaheuristic and default amount of generations.
pub fn solve_with_metaheuristic(problem: Problem, matrices: Option<Vec<Matrix>>) -> Solution {
    solve_with_metaheuristic_and_iterations(problem, matrices, 200)
}

/// Runs solver with default metaheuristic and specified amount of generations.
pub fn solve_with_metaheuristic_and_iterations(
    problem: Problem,
    matrices: Option<Vec<Matrix>>,
    generations: usize,
) -> Solution {
    solve(problem, matrices, generations, true)
}

/// Runs solver with default metaheuristic and specified amount of generations without feasibility check.
pub fn solve_with_metaheuristic_and_iterations_without_check(
    problem: Problem,
    matrices: Option<Vec<Matrix>>,
    generations: usize,
) -> Solution {
    solve(problem, matrices, generations, false)
}

pub fn solve(problem: Problem, matrices: Option<Vec<Matrix>>, generations: usize, perform_check: bool) -> Solution {
    get_core_solution(problem, matrices, perform_check, |problem: Arc<CoreProblem>| {
        let (solution, _, _) = Builder::new(problem)
            .with_max_generations(Some(generations))
            .build()
            .unwrap_or_else(|err| panic!("cannot build solver: {}", err))
            .solve()
            .unwrap_or_else(|err| panic!("cannot solve the problem: {}", err));

        solution
    })
}

fn get_core_problem(problem: Problem, matrices: Option<Vec<Matrix>>) -> Arc<CoreProblem> {
    Arc::new(
        if let Some(matrices) = matrices { (problem, matrices).read_pragmatic() } else { problem.read_pragmatic() }
            .unwrap(),
    )
}

fn get_core_solution<F: Fn(Arc<CoreProblem>) -> CoreSolution>(
    problem: Problem,
    matrices: Option<Vec<Matrix>>,
    perform_check: bool,
    solve_func: F,
) -> Solution {
    let format_problem = problem.clone();
    let format_matrices = matrices.clone();

    let core_problem = get_core_problem(problem, matrices);

    let core_solution = solve_func(core_problem.clone());

    let format_solution = sort_all_data(create_solution(&core_problem, &core_solution, None));

    if perform_check {
        if let Some(err) =
            CheckerContext::new(core_problem, format_problem.clone(), format_matrices, format_solution.clone())
                .check()
                .err()
        {
            panic!("check failed: '{}', problem: {:?}, solution: {:?}", err, format_problem, format_solution);
        }
    }

    sort_all_data(format_solution)
}

/// Sorts some solution properties in lexicographical order to simplify test assertions.
fn sort_all_data(solution: Solution) -> Solution {
    let mut solution = solution;

    solution.tours.sort_by(|a, b| a.vehicle_id.partial_cmp(&b.vehicle_id).unwrap_or(Less));

    if let Some(ref mut unassigned) = solution.unassigned {
        unassigned.sort_by(|a, b| a.job_id.partial_cmp(&b.job_id).unwrap_or(Less));
    }

    solution
}
