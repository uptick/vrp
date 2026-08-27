use crate::format::problem::Objective::*;
use crate::format::problem::*;
use crate::format_time;
use crate::helpers::*;

#[test]
fn can_round_trip_prefer_early_tours_objective() {
    let json = r#"{"type":"prefer-early-tours"}"#;

    let objective: Objective = serde_json::from_str(json).expect("cannot deserialize objective");
    assert!(matches!(objective, PreferEarlyTours));

    let serialized = serde_json::to_string(&objective).expect("cannot serialize objective");
    assert_eq!(serialized, json);
}

/// Builds a problem with one job and two single-shift vehicles: `early` starts first but its depot
/// is far from the job, `late` starts a day later but is parked on top of it. Cost alone therefore
/// prefers `late`.
fn create_problem(objectives: Option<Vec<Objective>>, out_of_hours_travel: bool) -> Problem {
    let shift = |earliest: f64, latest: f64, location: (f64, f64)| VehicleShift {
        start: ShiftStart { earliest: format_time(earliest), latest: None, location: location.to_loc() },
        end: Some(ShiftEnd { earliest: None, latest: format_time(latest), location: location.to_loc() }),
        ..create_default_vehicle_shift()
    };

    let limits = || {
        out_of_hours_travel.then_some(VehicleLimits {
            max_distance: None,
            max_duration: None,
            tour_size: None,
            allow_out_of_hours_depot_travel: Some(true),
        })
    };

    Problem {
        plan: Plan { jobs: vec![create_delivery_job("job1", (100., 0.))], ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![
                VehicleType {
                    shifts: vec![shift(0., 1000., (0., 0.))],
                    limits: limits(),
                    ..create_default_vehicle("early")
                },
                VehicleType {
                    shifts: vec![shift(10000., 11000., (100., 0.))],
                    limits: limits(),
                    ..create_default_vehicle("late")
                },
            ],
            ..create_default_fleet()
        },
        objectives,
    }
}

fn solve(objectives: Option<Vec<Objective>>, out_of_hours_travel: bool) -> String {
    let problem = create_problem(objectives, out_of_hours_travel);
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    solution.tours.first().expect("no tours in solution").vehicle_id.clone()
}

#[test]
fn can_prefer_cheaper_later_shift_without_objective() {
    assert_eq!(solve(None, false), "late_1");
}

#[test]
fn can_prefer_earliest_shift_over_cost() {
    let vehicle_id = solve(objectives(), false);

    assert_eq!(vehicle_id, "early_1");
}

#[test]
fn can_prefer_earliest_shift_with_out_of_hours_depot_travel() {
    // `allow_out_of_hours_depot_travel` relaxes the shift's start bound to `None`, so the actor's
    // start time collapses to zero and every shift would look equally early. The objective reads
    // the first job arrival floor instead, which keeps the real shift start.
    let vehicle_id = solve(objectives(), true);

    assert_eq!(vehicle_id, "early_1");
}

fn objectives() -> Option<Vec<Objective>> {
    Some(vec![MinimizeUnassigned { breaks: None }, MinimizeTours, PreferEarlyTours, MinimizeCost])
}
