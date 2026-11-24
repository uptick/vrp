use super::*;
use crate::format::Location;
use vrp_core::models::common::Profile;
use vrp_core::models::problem::{MatrixData, create_matrix_transport_cost};

fn get_test_locations() -> Vec<Location> {
    vec![
        Location::Coordinate { lat: 52.52599, lng: 13.45413 },
        Location::Coordinate { lat: 52.5225, lng: 13.4095 },
        Location::Coordinate { lat: 52.5165, lng: 13.3808 },
    ]
}

#[test]
fn can_calculate_distance_between_two_locations() {
    let l1 = Location::Coordinate { lat: 52.52599, lng: 13.45413 };
    let l2 = Location::Coordinate { lat: 52.5165, lng: 13.3808 };

    let distance = get_haversine_distance(&l1, &l2);

    assert_eq!(distance.round(), 5078.);
}

#[test]
fn can_use_approximated_with_matrix_costs() {
    let profile = Profile::default();
    let locations = get_test_locations();
    let speed = 10.;
    let approx_data = get_approx_transportation(&locations, &[Approximation::Speed(speed)]);
    assert_eq!(approx_data.len(), 1);

    let (durations, distances) = approx_data.first().unwrap();
    let durations = durations.iter().map(|&d| d as Float).collect();
    let distances = distances.iter().map(|&d| d as Float).collect();

    let costs = create_matrix_transport_cost(vec![MatrixData::new(profile.index, None, durations, distances)])
        .expect("Cannot create matrix transport costs");

    vec![(0, 1, 3048.), (1, 2, 2056.), (2, 0, 5078.)].into_iter().for_each(|(from, to, expected)| {
        let distance = costs.distance_approx(&profile, from, to);
        let duration = costs.duration_approx(&profile, from, to);

        assert_eq!(distance.round(), expected);
        assert_eq!(duration.round(), (distance / speed).round());
    });
}

#[test]
fn can_interpolate_tables_from_origin_and_beyond_last_point() {
    let points = [[1000., 100.], [3000., 200.]];

    vec![(0., 0.), (500., 50.), (1000., 100.), (2000., 150.), (3000., 200.), (5000., 300.)].into_iter().for_each(
        |(x, expected)| {
            assert_eq!(interpolate(&points, x), expected, "at {x}");
        },
    );
    // one point is a line from the origin, extended
    assert_eq!(interpolate(&[[1000., 100.]], 2500.), 250.);
}

#[test]
fn can_use_tables_for_durations_and_distances() {
    let locations = get_test_locations();
    // invented tables: 2 minutes to the first kilometre, then a kilometre a minute; roads a third longer
    let tables =
        Approximation::Tables { durations: vec![[1000., 120.], [2000., 180.]], distances: Some(vec![[1000., 1333.]]) };
    let durations_only = Approximation::Tables { durations: vec![[1000., 120.], [2000., 180.]], distances: None };

    let approx_data = get_approx_transportation(&locations, &[tables, durations_only]);
    assert_eq!(approx_data.len(), 2);

    let straight = |from: usize, to: usize| get_haversine_distance(&locations[from], &locations[to]);
    let (durations, distances) = &approx_data[0];
    // 3048 and 2056 meters: on the second segment
    assert_eq!(durations[1], (120. + (straight(0, 1) - 1000.) * 60. / 1000.).round() as i64);
    assert_eq!(durations[5], (120. + (straight(1, 2) - 1000.) * 60. / 1000.).round() as i64);
    // 5078 meters: beyond the only distance point
    assert_eq!(distances[2], (straight(0, 2) * 1333. / 1000.).round() as i64);
    assert_eq!(durations[0], 0);
    assert_eq!(distances[0], 0);

    let (_, straight_distances) = &approx_data[1];
    assert_eq!(straight_distances[2], 5078);
}

#[test]
fn can_tell_approximations_apart_by_key() {
    let tables = |distances| Approximation::Tables { durations: vec![[1000., 120.]], distances };

    assert_eq!(Approximation::Speed(10.).key(), Approximation::Speed(10.).key());
    assert_ne!(Approximation::Speed(10.).key(), Approximation::Speed(8.).key());
    assert_eq!(tables(None).key(), tables(None).key());
    assert_ne!(tables(None).key(), tables(Some(vec![[1000., 1300.]])).key());
}
