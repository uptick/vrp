use super::*;
use crate::format::problem::*;
use crate::helpers::*;

/// Three jobs at indices 0, 1 and 2, so each index is a location in the problem.
fn create_coord_index() -> CoordIndex {
    let problem = Problem {
        plan: Plan {
            jobs: (0..3).map(|index| create_delivery_job_with_index(&format!("job{index}"), index)).collect(),
            ..create_empty_plan()
        },
        fleet: Fleet { vehicles: vec![], ..create_default_fleet() },
        ..create_empty_problem()
    };

    CoordIndex::new(&problem)
}

fn arrival(index: usize, buffer: Option<Float>, round_up_to: Option<Float>) -> ArrivalAdjustment {
    ArrivalAdjustment { location: Location::Reference { index }, buffer, round_up_to }
}

fn create_adjustments(minimum: Option<Float>, arrivals: Vec<ArrivalAdjustment>) -> ProfileAdjustments {
    ProfileAdjustments::new(&TravelAdjustments { minimum, arrivals: Some(arrivals) }, &create_coord_index())
}

parameterized_test! {can_adjust_leg_duration, (minimum, arrivals, from, to, duration, expected), {
    can_adjust_leg_duration_impl(minimum, arrivals, from, to, duration, expected);
}}

can_adjust_leg_duration! {
    case01_minimum_raises_short_leg: (Some(300.), vec![], 0, 1, 120., 300.),
    case02_minimum_keeps_long_leg: (Some(300.), vec![], 0, 1, 420., 420.),
    case03_buffer_on_arrival: (None, vec![arrival(1, Some(60.), None)], 0, 1, 120., 180.),
    case04_no_buffer_on_departure: (None, vec![arrival(1, Some(60.), None)], 1, 0, 120., 120.),
    case05_round_up: (None, vec![arrival(1, None, Some(600.))], 0, 1, 601., 1200.),
    case06_round_up_keeps_multiple: (None, vec![arrival(1, None, Some(600.))], 0, 1, 600., 600.),
    case07_minimum_before_rules: (Some(300.), vec![arrival(1, Some(60.), None)], 0, 1, 120., 360.),
    case08_rules_in_order: (None, vec![arrival(1, Some(60.), None), arrival(1, None, Some(600.))], 0, 1, 590., 1200.),
    case09_rules_in_other_order: (None, vec![arrival(1, None, Some(600.)), arrival(1, Some(60.), None)], 0, 1, 590., 660.),
    case10_diagonal_untouched: (Some(300.), vec![arrival(1, Some(60.), None)], 1, 1, 0., 0.),
    case11_unreachable_untouched: (Some(300.), vec![arrival(1, Some(60.), None)], 0, 1, -1., -1.),
}

fn can_adjust_leg_duration_impl(
    minimum: Option<Float>,
    arrivals: Vec<ArrivalAdjustment>,
    from: usize,
    to: usize,
    duration: Float,
    expected: Float,
) {
    let adjustments = create_adjustments(minimum, arrivals);

    assert_eq!(adjustments.apply(from, to, duration), expected);
}

#[test]
fn can_adjust_matrix_row_major_by_arrival() {
    // A buffer on arrivals at location 1: column 1 of the row-major matrix, never row 1.
    let adjustments = create_adjustments(None, vec![arrival(1, Some(60.), None)]);
    let mut durations = vec![0., 10., 20., 30., 0., 40., 50., 60., 0.];

    adjustments.apply_to_matrix(&mut durations, 3);

    assert_eq!(durations, vec![0., 70., 20., 30., 0., 40., 50., 120., 0.]);
}

#[test]
fn can_resolve_arrival_rules_by_coordinate() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![create_delivery_job("job1", (1., 0.)), create_delivery_job("job2", (2., 0.))],
            ..create_empty_plan()
        },
        fleet: Fleet { vehicles: vec![], ..create_default_fleet() },
        ..create_empty_problem()
    };
    let coord_index = CoordIndex::new(&problem);
    let location = Location::Coordinate { lat: 2., lng: 0. };
    let to = coord_index.get_by_loc(&location).unwrap();
    let from = coord_index.get_by_loc(&Location::Coordinate { lat: 1., lng: 0. }).unwrap();
    let adjustments = ProfileAdjustments::new(
        &TravelAdjustments {
            minimum: None,
            arrivals: Some(vec![ArrivalAdjustment { location, buffer: Some(60.), round_up_to: None }]),
        },
        &coord_index,
    );

    assert_eq!(adjustments.apply(from, to, 100.), 160.);
    assert_eq!(adjustments.apply(to, from, 100.), 100.);
}
