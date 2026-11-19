use super::*;
use crate::helpers::*;

#[test]
fn can_detect_duplicates() {
    let problem = Problem {
        fleet: Fleet {
            profiles: vec![
                MatrixProfile { name: "my_vehicle".to_string(), speed: None },
                MatrixProfile { name: "my_vehicle".to_string(), speed: None },
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
            profiles: vec![MatrixProfile { name: "car".to_string(), speed: None }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);

    let result = check_e1505_profiles_exist(&ctx);

    assert_eq!(result.err().map(|err| err.code), Some("E1505".to_string()));
}

fn create_problem_with_location_table(table_size: usize, jobs: Vec<Job>) -> Problem {
    Problem {
        plan: Plan { jobs, ..create_empty_plan() },
        // No vehicles: their depots would add coordinate locations.
        fleet: Fleet { vehicles: vec![], ..create_default_fleet() },
        locations: Some((0..table_size).map(|idx| Some(GeoCoordinate { lat: idx as f64, lng: 0. })).collect()),
        ..create_empty_problem()
    }
}

fn create_matrix_of_size(profile: Option<&str>, size: usize) -> Matrix {
    Matrix {
        profile: profile.map(|profile| profile.to_string()),
        timestamp: None,
        travel_times: vec![1; size * size],
        distances: vec![1; size * size],
        error_codes: None,
    }
}

#[test]
fn can_use_location_table_instead_of_matrix_with_indices() {
    let jobs = vec![create_delivery_job_with_index("job1", 0), create_delivery_job_with_index("job2", 1)];
    let problem = create_problem_with_location_table(2, jobs);
    let coord_index = CoordIndex::new(&problem);
    let matrices = vec![];
    let ctx = ValidationContext::new(&problem, Some(&matrices), &coord_index);
    let location_types = (ctx.coord_index.has_coordinates(), ctx.coord_index.has_indices());

    assert!(check_e1503_no_matrix_when_indices_used(&ctx, location_types).is_ok());
    assert!(check_e1504_index_size_mismatch(&ctx).is_ok());
}

#[test]
fn can_detect_location_table_size_mismatch_with_indices() {
    let jobs = vec![create_delivery_job_with_index("job1", 0), create_delivery_job_with_index("job2", 1)];
    let problem = create_problem_with_location_table(3, jobs);
    let coord_index = CoordIndex::new(&problem);
    let matrices = vec![];
    let ctx = ValidationContext::new(&problem, Some(&matrices), &coord_index);

    let result = check_e1504_index_size_mismatch(&ctx);

    assert_eq!(result.err().map(|err| err.code), Some("E1504".to_string()));
}

#[test]
fn can_detect_location_table_with_coordinates() {
    let problem = create_problem_with_location_table(1, vec![create_delivery_job("job1", (1., 0.))]);
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, None, &coord_index);
    let location_types = (ctx.coord_index.has_coordinates(), ctx.coord_index.has_indices());

    let result = check_e1506_location_table_with_coordinates(&ctx, location_types);

    assert_eq!(result.err().map(|err| err.code), Some("E1506".to_string()));
}

parameterized_test! {can_detect_matrix_size_differs_from_location_table, (matrix_size, expected), {
    can_detect_matrix_size_differs_from_location_table_impl(matrix_size, expected);
}}

can_detect_matrix_size_differs_from_location_table! {
    case01_same_size: (2, None),
    case02_smaller: (1, Some("E1507")),
    case03_larger: (3, Some("E1507")),
}

fn can_detect_matrix_size_differs_from_location_table_impl(matrix_size: usize, expected: Option<&str>) {
    let jobs = vec![create_delivery_job_with_index("job1", 0), create_delivery_job_with_index("job2", 1)];
    let problem = create_problem_with_location_table(2, jobs);
    let matrices = vec![create_matrix_of_size(Some("car"), matrix_size)];
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, Some(&matrices), &coord_index);

    let result = check_e1507_matrix_size_differs_from_location_table(&ctx);

    assert_eq!(result.err().map(|err| err.code), expected.map(|code| code.to_string()));
}

parameterized_test! {can_detect_unnamed_matrix_with_location_table, (profile, expected), {
    can_detect_unnamed_matrix_with_location_table_impl(profile, expected);
}}

can_detect_unnamed_matrix_with_location_table! {
    case01_named: (Some("car"), None),
    case02_unnamed: (None, Some("E1508")),
}

fn can_detect_unnamed_matrix_with_location_table_impl(profile: Option<&str>, expected: Option<&str>) {
    let problem = create_problem_with_location_table(1, vec![create_delivery_job_with_index("job1", 0)]);
    let matrices = vec![create_matrix_of_size(profile, 1)];
    let coord_index = CoordIndex::new(&problem);
    let ctx = ValidationContext::new(&problem, Some(&matrices), &coord_index);

    let result = check_e1508_unnamed_matrix_with_location_table(&ctx);

    assert_eq!(result.err().map(|err| err.code), expected.map(|code| code.to_string()));
}
