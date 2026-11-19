use crate::format::Location;
use crate::format::problem::*;
use crate::format_time;
use crate::helpers::*;

fn create_profile(adjustments: Option<TravelAdjustments>) -> MatrixProfile {
    MatrixProfile {
        name: "car".to_string(),
        speed: None,
        distance_multiplier: None,
        unknown_location_travel: None,
        adjustments,
    }
}

fn create_problem(jobs: Vec<Job>, start: Location, adjustments: Option<TravelAdjustments>) -> Problem {
    Problem {
        plan: Plan { jobs, ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![VehicleShift {
                    start: ShiftStart { earliest: format_time(0.), latest: None, location: start },
                    ..create_default_open_vehicle_shift()
                }],
                ..create_default_vehicle_type()
            }],
            profiles: vec![create_profile(adjustments)],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    }
}

#[test]
fn can_adjust_supplied_matrix_by_arrival_index() {
    // Depot at index 2, jobs at 0 and 1. Arriving at 0 takes 10 seconds longer, so the vehicle
    // goes 2 -> 1 (2s) -> 0 (1s + 10s) rather than 2 -> 0 (3s + 10s) -> 1 (3s).
    let jobs = vec![create_delivery_job_with_index("job1", 0), create_delivery_job_with_index("job2", 1)];
    let adjustments = TravelAdjustments {
        minimum: None,
        arrivals: Some(vec![ArrivalAdjustment {
            location: Location::Reference { index: 0 },
            buffer: Some(10.),
            round_up_to: None,
        }]),
    };
    let problem = create_problem(jobs, Location::Reference { index: 2 }, Some(adjustments));
    let matrix = Matrix {
        profile: Some("car".to_string()),
        timestamp: None,
        travel_times: vec![0, 3, 3, 1, 0, 3, 3, 2, 0],
        distances: vec![0, 3, 3, 1, 0, 3, 3, 2, 0],
        error_codes: None,
    };

    // The solution checker applies the same adjustments to the matrix it checks against.
    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(solution.unassigned.is_none());
    let job_order = solution.tours[0]
        .stops
        .iter()
        .flat_map(|stop| stop.activities().iter().map(|activity| activity.job_id.clone()))
        .collect::<Vec<_>>();
    assert_eq!(job_order, vec!["departure", "job2", "job1"]);
    // Durations are adjusted; distances never are.
    assert_eq!(solution.statistic.times.driving, 13);
    assert_eq!(solution.statistic.distance, 3);
}

#[test]
fn can_adjust_approximated_travel() {
    let jobs = || vec![create_delivery_job("job1", (1., 0.)), create_delivery_job("job2", (2., 0.))];
    let start = || Location::Coordinate { lat: 0., lng: 0. };
    let minimum = 100_000.;

    let unadjusted = solve_with_metaheuristic(create_problem(jobs(), start(), None), None);
    let adjusted = solve_with_metaheuristic(
        create_problem(jobs(), start(), Some(TravelAdjustments { minimum: Some(minimum), arrivals: None })),
        None,
    );

    // Two legs of about 11,100 seconds each, both raised to the minimum.
    assert!(unadjusted.statistic.times.driving < 2 * minimum as i64);
    assert_eq!(adjusted.statistic.times.driving, 2 * minimum as i64);
    assert_eq!(adjusted.statistic.distance, unadjusted.statistic.distance);
}
