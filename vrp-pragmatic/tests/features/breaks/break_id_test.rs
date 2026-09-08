use crate::format::problem::*;
use crate::format::solution::*;
use crate::format_time;
use crate::helpers::*;

fn get_break_ids(solution: &Solution) -> Vec<Option<String>> {
    solution
        .tours
        .iter()
        .flat_map(|tour| tour.stops.iter())
        .flat_map(|stop| stop.activities().iter())
        .filter(|activity| activity.activity_type == "break")
        .map(|activity| activity.break_id.clone())
        .collect()
}

fn create_problem(vehicle_break: VehicleBreak, jobs: Vec<Job>) -> Problem {
    Problem {
        plan: Plan { jobs, ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![VehicleShift { breaks: Some(vec![vehicle_break]), ..create_default_vehicle_shift() }],
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    }
}

#[test]
fn can_propagate_optional_break_id_to_solution() {
    let problem = create_problem(
        VehicleBreak::Optional {
            id: Some("lunch".to_string()),
            time: VehicleOptionalBreakTime::TimeWindow(vec![format_time(5.), format_time(10.)]),
            places: vec![VehicleOptionalBreakPlace {
                duration: 2.0,
                location: Some((6., 0.).to_loc()),
                tag: Some("break_tag".to_string()),
            }],
            policy: None,
        },
        vec![create_delivery_job("job1", (5., 0.)), create_delivery_job("job2", (10., 0.))],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(get_break_ids(&solution), vec![Some("lunch".to_string())]);
}

#[test]
fn can_propagate_required_break_id_to_solution() {
    let problem = create_problem(
        VehicleBreak::Required {
            id: Some("mandatory_rest".to_string()),
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(7.), latest: format_time(7.) },
            duration: 2.,
        },
        vec![create_delivery_job("job1", (5., 0.)), create_delivery_job("job2", (10., 0.))],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(get_break_ids(&solution), vec![Some("mandatory_rest".to_string())]);
}

#[test]
fn can_distinguish_multiple_breaks_by_id() {
    let problem = create_problem(
        VehicleBreak::Optional {
            id: Some("first_break".to_string()),
            time: VehicleOptionalBreakTime::TimeWindow(vec![format_time(5.), format_time(10.)]),
            places: vec![VehicleOptionalBreakPlace { duration: 2.0, location: Some((6., 0.).to_loc()), tag: None }],
            policy: None,
        },
        vec![create_delivery_job("job1", (5., 0.)), create_delivery_job("job2", (10., 0.))],
    );
    let problem = Problem {
        fleet: Fleet {
            vehicles: problem
                .fleet
                .vehicles
                .into_iter()
                .map(|vehicle| VehicleType {
                    shifts: vehicle
                        .shifts
                        .into_iter()
                        .map(|shift| VehicleShift {
                            breaks: shift.breaks.map(|mut breaks| {
                                breaks.push(VehicleBreak::Optional {
                                    id: Some("second_break".to_string()),
                                    time: VehicleOptionalBreakTime::TimeWindow(vec![
                                        format_time(12.),
                                        format_time(15.),
                                    ]),
                                    places: vec![VehicleOptionalBreakPlace {
                                        duration: 2.0,
                                        location: Some((10., 0.).to_loc()),
                                        tag: None,
                                    }],
                                    policy: None,
                                });
                                breaks
                            }),
                            ..shift
                        })
                        .collect(),
                    ..vehicle
                })
                .collect(),
            ..problem.fleet
        },
        ..problem
    };
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(get_break_ids(&solution), vec![Some("first_break".to_string()), Some("second_break".to_string())]);
}

#[test]
fn can_propagate_break_id_to_violation() {
    let problem = create_problem(
        VehicleBreak::Optional {
            id: Some("lunch".to_string()),
            time: VehicleOptionalBreakTime::TimeWindow(vec![format_time(5.), format_time(8.)]),
            places: vec![VehicleOptionalBreakPlace { duration: 2.0, location: Some((6., 0.).to_loc()), tag: None }],
            policy: None,
        },
        vec![create_delivery_job_with_duration("job1", (1., 0.), 10.)],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(
        solution.violations,
        Some(vec![Violation::Break {
            break_id: Some("lunch".to_string()),
            vehicle_id: "my_vehicle_1".to_string(),
            shift_index: 0
        }])
    );
}

#[test]
fn can_omit_break_id_when_not_specified() {
    let problem = create_problem(
        VehicleBreak::Optional {
            id: None,
            time: VehicleOptionalBreakTime::TimeWindow(vec![format_time(5.), format_time(10.)]),
            places: vec![VehicleOptionalBreakPlace { duration: 2.0, location: Some((6., 0.).to_loc()), tag: None }],
            policy: None,
        },
        vec![create_delivery_job("job1", (5., 0.)), create_delivery_job("job2", (10., 0.))],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(get_break_ids(&solution), vec![None]);
}
