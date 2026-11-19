use super::*;
use crate::format::CustomLocationType;
use crate::helpers::*;

#[test]
fn can_detect_duplicates() {
    let problem = Problem {
        fleet: Fleet {
            profiles: vec![
                MatrixProfile {
                    name: "my_vehicle".to_string(),
                    speed: None,
                    distance_multiplier: None,
                    unknown_location_travel: None,
                    adjustments: None,
                },
                MatrixProfile {
                    name: "my_vehicle".to_string(),
                    speed: None,
                    distance_multiplier: None,
                    unknown_location_travel: None,
                    adjustments: None,
                },
            ],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);

    let result = check_e1500_duplicated_profiles(&ctx);

    assert_eq!(result.err().map(|err| err.code), Some("E1500".to_string()));
}

#[test]
fn can_detect_empty_profiles() {
    let problem = Problem { fleet: Fleet { profiles: vec![], ..create_default_fleet() }, ..create_empty_problem() };
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);

    let result = check_e1501_empty_profiles(&ctx);

    assert_eq!(result.err().map(|err| err.code), Some("E1501".to_string()));
}

#[test]
fn can_detect_mixed_locations() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![create_delivery_job_with_index("job1", 0), create_delivery_job("job2", (1., 0.))],
            relations: None,
            ..create_empty_plan()
        },
        ..create_empty_problem()
    };
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);
    let location_types = (ctx.coord_index.has_coordinates(), ctx.coord_index.has_indices());

    let result = check_e1502_no_location_type_mix(&ctx, location_types);

    assert_eq!(result.err().map(|err| err.code), Some("E1502".to_string()));
}

#[test]
fn can_detect_missing_matrix_when_indices_used() {
    let problem = Problem {
        plan: Plan { jobs: vec![create_delivery_job_with_index("job1", 0)], ..create_empty_plan() },
        ..create_empty_problem()
    };
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);
    let location_types = (ctx.coord_index.has_coordinates(), ctx.coord_index.has_indices());

    let result = check_e1503_no_matrix_when_indices_used(&ctx, location_types);

    assert_eq!(result.err().map(|err| err.code), Some("E1503".to_string()));
}

#[test]
fn can_detect_index_mismatch() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![
                create_delivery_job_with_index("job1", 0),
                create_delivery_job_with_index("job2", 1),
                create_delivery_job_with_index("job3", 2),
            ],
            ..create_empty_plan()
        },
        ..create_empty_problem()
    };
    let matrices = vec![Matrix {
        profile: Some("car".to_owned()),
        timestamp: None,
        travel_times: vec![1; 4],
        distances: vec![1; 4],
        error_codes: None,
    }];
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, Some(&matrices), &coord_index);

    let result = check_e1504_index_size_mismatch(&ctx);

    assert_eq!(result.err().map(|err| err.code), Some("E1504".to_string()));
}

#[test]
fn can_detect_missing_profile() {
    let problem = Problem {
        fleet: Fleet {
            vehicles: vec![
                VehicleType { profile: create_vehicle_profile_with_name("car"), ..create_default_vehicle_type() },
                VehicleType { profile: create_vehicle_profile_with_name("truck"), ..create_default_vehicle_type() },
            ],
            profiles: vec![MatrixProfile {
                name: "car".to_string(),
                speed: None,
                distance_multiplier: None,
                unknown_location_travel: None,
                adjustments: None,
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);

    let result = check_e1505_profiles_exist(&ctx);

    assert_eq!(result.err().map(|err| err.code), Some("E1505".to_string()));
}

parameterized_test! {can_detect_invalid_approximation_settings, (speed, distance_multiplier, unknown_location_travel, expected), {
    can_detect_invalid_approximation_settings_impl(speed, distance_multiplier, unknown_location_travel, expected);
}}

can_detect_invalid_approximation_settings! {
    case01_defaults: (None, None, None, None),
    case02_valid: (Some(1.4), Some(1.3), Some((86400., 1000000.)), None),
    case03_zero_unknown_location: (None, None, Some((0., 0.)), None),
    case04_zero_speed: (Some(0.), None, None, Some("E1506")),
    case05_negative_distance_multiplier: (None, Some(-1.), None, Some("E1506")),
    case06_zero_distance_multiplier: (None, Some(0.), None, Some("E1506")),
    case07_negative_unknown_location_duration: (None, None, Some((-1., 0.)), Some("E1506")),
    case08_infinite_unknown_location_distance: (None, None, Some((0., f64::INFINITY)), Some("E1506")),
}

fn can_detect_invalid_approximation_settings_impl(
    speed: Option<f64>,
    distance_multiplier: Option<f64>,
    unknown_location_travel: Option<(f64, f64)>,
    expected: Option<&str>,
) {
    let unknown_location_travel =
        unknown_location_travel.map(|(duration, distance)| UnknownLocationTravel { duration, distance });
    let problem = Problem {
        fleet: Fleet {
            profiles: vec![MatrixProfile {
                name: "car".to_string(),
                speed,
                distance_multiplier,
                unknown_location_travel,
                adjustments: None,
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);

    let result = check_e1506_invalid_approximation_settings(&ctx);

    assert_eq!(result.err().map(|err| err.code), expected.map(|code| code.to_string()));
}

fn create_problem_with_adjustments(adjustments: TravelAdjustments) -> Problem {
    Problem {
        plan: Plan { jobs: vec![create_delivery_job("job1", (1., 0.))], ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![],
            profiles: vec![MatrixProfile {
                name: "car".to_string(),
                speed: None,
                distance_multiplier: None,
                unknown_location_travel: None,
                adjustments: Some(adjustments),
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    }
}

fn job_arrival(buffer: Option<f64>, round_up_to: Option<f64>) -> ArrivalAdjustment {
    ArrivalAdjustment { location: Location::Coordinate { lat: 1., lng: 0. }, buffer, round_up_to }
}

parameterized_test! {can_detect_invalid_travel_adjustments, (minimum, arrival, expected), {
    can_detect_invalid_travel_adjustments_impl(minimum, arrival, expected);
}}

can_detect_invalid_travel_adjustments! {
    case01_minimum_only: (Some(300.), None, None),
    case02_buffer: (None, Some(job_arrival(Some(60.), None)), None),
    case03_round_up: (None, Some(job_arrival(None, Some(600.))), None),
    case04_zero_minimum_and_buffer: (Some(0.), Some(job_arrival(Some(0.), None)), None),
    case05_negative_minimum: (Some(-1.), None, Some("E1507")),
    case06_both_buffer_and_round_up: (None, Some(job_arrival(Some(60.), Some(600.))), Some("E1507")),
    case07_neither_buffer_nor_round_up: (None, Some(job_arrival(None, None)), Some("E1507")),
    case08_negative_buffer: (None, Some(job_arrival(Some(-60.), None)), Some("E1507")),
    case09_zero_round_up: (None, Some(job_arrival(None, Some(0.))), Some("E1507")),
}

fn can_detect_invalid_travel_adjustments_impl(
    minimum: Option<f64>,
    arrival: Option<ArrivalAdjustment>,
    expected: Option<&str>,
) {
    let problem = create_problem_with_adjustments(TravelAdjustments { minimum, arrivals: arrival.map(|a| vec![a]) });
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);

    let result = check_e1507_invalid_travel_adjustments(&ctx);

    assert_eq!(result.err().map(|err| err.code), expected.map(|code| code.to_string()));
}

parameterized_test! {can_detect_unknown_arrival_location, (location, expected), {
    can_detect_unknown_arrival_location_impl(location, expected);
}}

can_detect_unknown_arrival_location! {
    case01_job_location: (Location::Coordinate { lat: 1., lng: 0. }, None),
    case02_not_in_problem: (Location::Coordinate { lat: 9., lng: 9. }, Some("E1508")),
    case03_unknown_location: (Location::Custom { r#type: CustomLocationType::Unknown }, Some("E1508")),
}

fn can_detect_unknown_arrival_location_impl(location: Location, expected: Option<&str>) {
    let arrival = ArrivalAdjustment { location, buffer: Some(60.), round_up_to: None };
    let problem = create_problem_with_adjustments(TravelAdjustments { minimum: None, arrivals: Some(vec![arrival]) });
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);

    let result = check_e1508_unknown_arrival_location(&ctx);

    assert_eq!(result.err().map(|err| err.code), expected.map(|code| code.to_string()));
}
