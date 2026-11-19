use crate::format::Location;
use crate::format::problem::*;
use crate::format_time;
use crate::helpers::*;

const COORDINATES: [(f64, f64); 3] = [(1., 0.), (2., 0.), (0., 0.)];

/// Two deliveries and a depot, at `COORDINATES`. With `table`, every location is its index in
/// the problem's location table; without it, its coordinates.
fn create_problem(table: bool, profiles: Vec<MatrixProfile>, vehicle_profile: &str) -> Problem {
    let location = |index: usize| {
        if table {
            Location::Reference { index }
        } else {
            let (lat, lng) = COORDINATES[index];
            Location::Coordinate { lat, lng }
        }
    };
    let job = |id: &str, index: usize| {
        if table { create_delivery_job_with_index(id, index) } else { create_delivery_job(id, COORDINATES[index]) }
    };

    Problem {
        plan: Plan { jobs: vec![job("job1", 0), job("job2", 1)], ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                profile: create_vehicle_profile_with_name(vehicle_profile),
                shifts: vec![VehicleShift {
                    start: ShiftStart { earliest: format_time(0.), latest: None, location: location(2) },
                    ..create_default_open_vehicle_shift()
                }],
                ..create_default_vehicle_type()
            }],
            profiles,
            ..create_default_fleet()
        },
        locations: table.then(|| COORDINATES.iter().map(|&(lat, lng)| Some(GeoCoordinate { lat, lng })).collect()),
        ..create_empty_problem()
    }
}

fn profile(name: &str, speed: Option<f64>) -> MatrixProfile {
    MatrixProfile { name: name.to_string(), speed }
}

#[test]
fn can_approximate_from_location_table_as_from_coordinates() {
    let profiles = || vec![profile("car", Some(5.))];

    let from_coordinates = solve_with_metaheuristic(create_problem(false, profiles(), "car"), None);
    let from_table = solve_with_metaheuristic(create_problem(true, profiles(), "car"), None);

    assert!(from_table.unassigned.is_none());
    assert_eq!(from_table.statistic, from_coordinates.statistic);
}

#[test]
fn can_mix_routing_matrix_and_approximation_by_profile() {
    // "car" has a routing matrix it would take far longer to drive by; "walk" has none, so it
    // is approximated from the table.
    let car_matrix = Matrix {
        profile: Some("car".to_string()),
        timestamp: None,
        travel_times: vec![0, 1000, 1000, 1000, 0, 1000, 1000, 1000, 0],
        distances: vec![0, 1000, 1000, 1000, 0, 1000, 1000, 1000, 0],
        error_codes: None,
    };
    let walk_only = solve_with_metaheuristic(create_problem(false, vec![profile("walk", Some(2.))], "walk"), None);

    let mixed = solve_with_metaheuristic(
        create_problem(true, vec![profile("car", None), profile("walk", Some(2.))], "walk"),
        Some(vec![car_matrix]),
    );

    assert!(mixed.unassigned.is_none());
    assert_eq!(mixed.statistic, walk_only.statistic);
}
