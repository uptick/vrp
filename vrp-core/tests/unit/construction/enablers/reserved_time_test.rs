use super::*;
use crate::construction::enablers::ScheduleActivityState;
use crate::construction::features::TransportFeatureBuilder;
use crate::construction::heuristics::*;
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::problem::*;
use crate::helpers::models::solution::*;
use crate::models::problem::*;
use crate::models::{Feature, ViolationCode};
use rosomaxa::prelude::Float;

const VIOLATION_CODE: ViolationCode = ViolationCode(1);

type VehicleData = (Location, Location, Timestamp, Timestamp);
type ActivityData = (Location, (Timestamp, Timestamp), Duration);

fn create_detail(
    locations: (Option<Location>, Option<Location>),
    time: Option<(Timestamp, Timestamp)>,
) -> VehicleDetail {
    let (start_location, end_location) = locations;
    VehicleDetail {
        start: start_location.map(|location| VehiclePlace {
            location,
            time: time.map_or(Default::default(), |(start, _)| TimeInterval { earliest: Some(start), latest: None }),
        }),
        end: end_location.map(|location| VehiclePlace {
            location,
            time: time.map_or(Default::default(), |(_, end)| TimeInterval { earliest: None, latest: Some(end) }),
        }),
    }
}

parameterized_test! {can_search_for_reserved_time, (times, tests), {
    can_search_for_reserved_time_impl(times, tests);
}}

can_search_for_reserved_time! {
    case01: (vec![((5., 5.), 5.), ((20., 20.), 10.)],
        vec![((6., 6.), Some(0)), ((2., 6.), Some(0)), ((10., 11.), None), ((2., 4.), None),
             ((10., 21.), Some(1)), ((25., 27.), Some(1)), ((29., 31.), Some(1)),
             ((0., 3.), None), ((31., 33.), None)]),
    case02: (vec![((0.,0.), 10.), ((5., 5.), 10.)], vec![]),
}

fn can_search_for_reserved_time_impl(
    times: Vec<((Timestamp, Timestamp), Duration)>,
    tests: Vec<((Timestamp, Timestamp), Option<usize>)>,
) {
    let route_ctx = RouteContextBuilder::default().build();
    let reserved_times = vec![(
        route_ctx.route().actor.clone(),
        times
            .iter()
            .cloned()
            .map(|((start, end), duration)| ReservedTimeSpan {
                time: TimeSpan::Window(TimeWindow::new(start, end)),
                duration,
                id: None,
            })
            .collect::<Vec<_>>(),
    )]
    .into_iter()
    .collect();

    let reserved_time_fn = create_reserved_times_fn(reserved_times);

    match reserved_time_fn {
        Ok(reserved_time_fn) => {
            tests.iter().enumerate().for_each(|(test_idx, ((s, e), expected))| {
                let interval = TimeWindow::new(*s, *e);
                let expected = expected.and_then(|idx| times.get(idx)).map(|((s, e), _)| TimeWindow::new(*s, *e));

                let result = (reserved_time_fn)(route_ctx.route(), &interval);

                assert_eq!(result.map(|r| r.time), expected, "test {test_idx} is failed");
            });
        }
        _ => {
            assert!(tests.is_empty())
        }
    }
}

fn create_feature_and_route(
    vehicle_detail_data: VehicleData,
    activities: Vec<ActivityData>,
    reserved_time: ReservedTimeSpan,
) -> (ReservedTimesFn, Feature, RouteContext) {
    let (location_start, location_end, time_start, time_end) = vehicle_detail_data;

    let activities = activities.into_iter().map(|(loc, (start, end), dur)| {
        ActivityBuilder::with_location_tw_and_duration(loc, TimeWindow::new(start, end), dur).build()
    });
    let fleet = FleetBuilder::default()
        .add_driver(test_driver())
        .add_vehicles(vec![
            TestVehicleBuilder::default()
                .id("v1")
                .details(vec![create_detail((Some(location_start), Some(location_end)), Some((time_start, time_end)))])
                .build(),
        ])
        .build();
    let reserved_times_idx =
        vec![(fleet.actors.first().unwrap().clone(), vec![reserved_time])].into_iter().collect::<HashMap<_, _>>();
    let mut route_ctx = RouteContextBuilder::default()
        .with_route(RouteBuilder::default().with_vehicle(&fleet, "v1").add_activities(activities).build())
        .build();
    let feature = TransportFeatureBuilder::new("minimize_costs")
        .set_violation_code(VIOLATION_CODE)
        .set_transport_cost(Arc::new(
            DynamicTransportCost::new(reserved_times_idx.clone(), Arc::new(TestTransportCost::default())).unwrap(),
        ))
        .set_activity_cost(Arc::new(DynamicActivityCost::new(reserved_times_idx.clone()).unwrap()))
        .build_minimize_cost()
        .unwrap();
    feature.state.as_ref().unwrap().accept_route_state(&mut route_ctx);

    (create_reserved_times_fn(reserved_times_idx).unwrap(), feature, route_ctx)
}

fn get_schedules(route_ctx: &RouteContext) -> Vec<(Timestamp, Timestamp)> {
    route_ctx.route().tour.all_activities().map(|a| (a.schedule.arrival, a.schedule.departure)).collect()
}

parameterized_test! {can_update_state_for_reserved_time, (vehicle_detail_data, reserved_time, activities, late_arrival_expected, expected_schedules), {
    let reserved_time = ReservedTimeSpan {
        time: TimeSpan::Window(TimeWindow::new(reserved_time.0, reserved_time.0)),
        duration: reserved_time.1 - reserved_time.0,
        id: None,
    };
    can_update_state_for_reserved_time_impl(vehicle_detail_data, reserved_time, activities, late_arrival_expected, expected_schedules);
}}

can_update_state_for_reserved_time! {
    case01_single_outside: ((0, 0, 0., 100.), (25., 30.),
              vec![(10, (0., 100.), 10.)],
              vec![Some(0.), Some(80.), None],
              vec![(0., 0.), (10., 20.), (35., 35.)]),

    case02_single_inside: ((0, 0, 0., 100.), (25., 30.),
              vec![(20, (0., 25.), 10.)],
              vec![Some(0.), Some(15.), None],
              vec![(0., 0.), (20., 40.), (60., 60.)]),

    case03_two_inside_travel: ((0, 0, 0., 100.), (25., 30.),
              vec![(10, (0., 20.), 10.), (20, (0., 40.), 10.)],
              vec![Some(0.), Some(15.), Some(40.), None],
              vec![(0., 0.), (10., 20.), (35., 45.), (65., 65.)]),

    case04_two_inside_service: ((0, 0, 0., 100.), (35., 40.),
              vec![(10, (0., 20.), 10.), (20, (0., 50.), 10.)],
              vec![Some(0.), Some(20.), Some(50.), None],
              vec![(0., 0.), (10., 20.), (30., 50.), (70., 70.)]),
}

fn can_update_state_for_reserved_time_impl(
    vehicle_detail_data: VehicleData,
    reserved_time: ReservedTimeSpan,
    activities: Vec<ActivityData>,
    late_arrival_expected: Vec<Option<Float>>,
    expected_schedules: Vec<(Timestamp, Timestamp)>,
) {
    let (_, _, route_ctx) = create_feature_and_route(vehicle_detail_data, activities, reserved_time);

    let schedules = get_schedules(&route_ctx);
    let late_arrival_result = (0..route_ctx.route().tour.total())
        .map(|activity_idx| route_ctx.state().get_schedule_at(activity_idx).map(|state| state.latest_arrival))
        .collect::<Vec<_>>();

    assert_eq!(schedules, expected_schedules);
    assert_eq!(late_arrival_result, late_arrival_expected);
}

parameterized_test! {can_evaluate_activity, (vehicle_detail_data, reserved_time, target, activities, expected_schedules), {
    let reserved_time = ReservedTimeSpan {
        time: TimeSpan::Window(TimeWindow::new(reserved_time.0, reserved_time.0)),
        duration: reserved_time.1 - reserved_time.0,
        id: None,
    };
    can_evaluate_activity_impl(vehicle_detail_data, reserved_time, target, activities, expected_schedules);
}}

can_evaluate_activity! {
    case01_break_starts_at_target_then_next_at_end:
        ((0, 0, 0., 100.), (10., 20.),
        (10, (0., 100.), 10.),
        vec![(20, (0., 40.), 10.)],
        vec![(0., 0.), (10., 30.), (40., 50.), (70., 70.)]),

    case02_break_starts_at_target_then_next_is_late:
        ((0, 0, 0., 100.), (10., 20.),
        (10, (0., 100.), 10.),
        vec![(20, (0., 39.), 10.)],
        vec![]),

    case03_break_with_waiting_time:
        ((0, 0, 0., 100.), (10., 20.),
        (10, (20., 100.), 10.),
        vec![(20, (0., 100.), 10.)],
        vec![(0., 0.), (10., 30.), (40., 50.), (70., 70.)]),

    case04_break_during_traveling:
        ((0, 0, 0., 100.), (5., 10.),
        (10, (0., 100.), 10.),
        vec![(20, (0., 100.), 10.)],
        vec![(0., 0.), (15., 25.), (35., 45.), (65., 65.)]),

    case05_break_on_whole_time_window_exclusive:
        ((0, 0, 0., 100.), (10., 20.),
        (10, (0., 20.), 10.),
        vec![(20, (0., 100.), 10.)],
        vec![(0., 0.), (10., 30.), (40., 50.), (70., 70.)]),

    case06_break_on_whole_time_window_inclusive:
        ((0, 0, 0., 100.), (10., 21.),
        (10, (0., 20.), 10.),
        vec![(20, (0., 100.), 10.)],
        vec![]),

    case07_break_on_whole_time_window_inclusive:
        ((0, 0, 0., 100.), (9., 21.),
        (10, (10., 20.), 10.),
        vec![(20, (0., 100.), 10.)],
        vec![]),

    case08_break_constraints_next:
        ((0, 0, 0., 100.), (50., 60.),
        (30, (0., 100.), 10.),
        vec![(20, (0., 50.), 10.)],
        vec![]),

    case09_break_constraints_next:
        ((0, 0, 0., 100.), (45., 60.),
        (30, (0., 100.), 10.),
        vec![(20, (0., 50.), 10.)],
        vec![]),
}

fn can_evaluate_activity_impl(
    vehicle_detail_data: VehicleData,
    reserved_time: ReservedTimeSpan,
    target: ActivityData,
    activities: Vec<ActivityData>,
    expected_schedules: Vec<(Timestamp, Timestamp)>,
) {
    let (_, feature, mut route_ctx) = create_feature_and_route(vehicle_detail_data, activities, reserved_time);
    let (feature_constraint, feature_state) = (feature.constraint.unwrap(), feature.state.unwrap());
    let (loc, (start, end), dur) = target;
    let prev = route_ctx.route().tour.get(0).unwrap();
    let target = ActivityBuilder::with_location_tw_and_duration(loc, TimeWindow::new(start, end), dur).build();
    let next = route_ctx.route().tour.get(1);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let activity_ctx = ActivityContext { index: 0, prev, target: &target, next };

    let is_violation =
        feature_constraint.evaluate(&MoveContext::activity(&solution_ctx, &route_ctx, &activity_ctx)).is_some();

    assert_eq!(is_violation, expected_schedules.is_empty());
    if !is_violation {
        route_ctx.route_mut().tour.insert_at(target, 1);
        feature_state.accept_route_state(&mut route_ctx);
        assert_eq!(get_schedules(&route_ctx), expected_schedules)
    }
}

parameterized_test! {can_avoid_reserved_time_when_driving, (vehicle_detail_data, reserved_time, activities, expected_schedules), {
    can_avoid_reserved_time_when_driving_impl(vehicle_detail_data, reserved_time, activities, expected_schedules);
}}

can_avoid_reserved_time_when_driving! {
    case01_should_move_duration_to_serving: (
        (0, 0, 0., 100.), (10., 40., 5.),
        vec![(10, (0., 100.), 10.), (50, (0., 100.), 10.)],
        vec![(0., 0.), (10., 25.), (65., 75.), (125., 125.)]
    ),
    case02_should_keep_duration_at_driving: (
        (0, 0, 0., 100.), (30., 40., 5.),
        vec![(10, (0., 100.), 10.), (50, (0., 100.), 10.)],
        vec![(0., 0.), (10., 20.), (65., 75.), (125., 125.)]
    ),
}

fn can_avoid_reserved_time_when_driving_impl(
    vehicle_detail_data: VehicleData,
    reserved_time: (Timestamp, Timestamp, Duration),
    activities: Vec<ActivityData>,
    expected_schedules: Vec<(Timestamp, Timestamp)>,
) {
    let reserved_time = ReservedTimeSpan {
        time: TimeSpan::Offset(TimeOffset::new(reserved_time.0, reserved_time.1)),
        duration: reserved_time.2,
        id: None,
    };
    let (reserved_times_fn, _, mut route_ctx) =
        create_feature_and_route(vehicle_detail_data, activities, reserved_time);

    avoid_reserved_time_when_driving(route_ctx.route_mut(), &reserved_times_fn);

    assert_eq!(get_schedules(&route_ctx), expected_schedules)
}

fn create_slot(earliest: Timestamp, latest: Timestamp, duration: Duration) -> ReservedSlot {
    ReservedSlot { earliest, latest, duration }
}

fn get_service_start(slots: &[ReservedSlot], arrival: Timestamp, tw: &TimeWindow, duration: Duration) -> Timestamp {
    estimate_service_start(slots.iter().copied(), arrival, tw, duration)
        .map_or(arrival.max(tw.start), |(service_start, _)| service_start)
}

fn get_latest_arrival(
    slots: &[ReservedSlot],
    latest_service_start: Timestamp,
    tw: &TimeWindow,
    duration: Duration,
) -> ControlFlow<Timestamp, Timestamp> {
    let times = ActorReservedTimes { slots: slots.to_vec(), latest_ends: get_latest_ends(slots), is_offset: false };

    estimate_latest_arrival(
        times.slots_until(latest_service_start.max(tw.start) + duration, 0.),
        latest_service_start,
        tw,
        duration,
    )
}

fn get_latest_ends(slots: &[ReservedSlot]) -> Vec<Timestamp> {
    slots
        .iter()
        .scan(Timestamp::MIN, |max_end, slot| {
            *max_end = max_end.max(slot.latest_end());
            Some(*max_end)
        })
        .collect()
}

parameterized_test! {can_estimate_service_start_around_reserved_time, (slots, arrival, tw, duration, expected), {
    let slots = slots.into_iter().map(|(earliest, latest, duration)| create_slot(earliest, latest, duration)).collect::<Vec<_>>();
    let result = estimate_service_start(slots.into_iter(), arrival, &TimeWindow::new(tw.0, tw.1), duration);

    assert_eq!(result, expected);
}}

can_estimate_service_start_around_reserved_time! {
    case01_service_is_over_before: (vec![(7., 7., 2.)], 2., (0., 100.), 5., None),
    case02_service_starts_after: (vec![(7., 7., 2.)], 9., (0., 100.), 5., None),
    case03_waits_for_exact_time: (vec![(7., 7., 2.)], 5., (0., 100.), 3., Some((9., 2.))),
    case04_takes_on_arrival: (vec![(4., 7., 2.)], 5., (0., 100.), 3., Some((7., 2.))),
    case05_fits_into_waiting: (vec![(7., 7., 2.)], 5., (9., 100.), 3., Some((9., 0.))),
    case06_overlaps_waiting: (vec![(7., 7., 2.)], 5., (8., 100.), 3., Some((9., 2.))),
    case07_is_not_taken_too_early: (vec![(2., 10.5, 2.)], 0., (10., 100.), 1., Some((11.5, 2.))),
    case08_runs_into_the_next_one: (vec![(6., 6., 2.), (9., 9., 2.)], 5., (0., 100.), 3., Some((11., 4.))),
    case09_skips_the_next_one: (vec![(6., 6., 2.), (20., 20., 2.)], 5., (0., 100.), 3., Some((8., 2.))),
    case10_takes_the_next_one_while_waiting: (vec![(1., 1., 1.), (2., 11., 2.)], 0., (10., 100.), 3., Some((10., 0.))),
}

parameterized_test! {can_estimate_latest_arrival_around_reserved_time, (slots, latest_service_start, tw, duration, expected), {
    let slots = slots.into_iter().map(|(earliest, latest, duration)| create_slot(earliest, latest, duration)).collect::<Vec<_>>();
    let result = get_latest_arrival(&slots, latest_service_start, &TimeWindow::new(tw.0, tw.1), duration);

    assert_eq!(result, expected);
}}

can_estimate_latest_arrival_around_reserved_time! {
    case01_service_is_over_before: (vec![(7., 7., 2.)], 2., (0., 100.), 5., ControlFlow::Continue(2.)),
    case02_service_starts_after: (vec![(7., 7., 2.)], 9., (0., 100.), 5., ControlFlow::Continue(9.)),
    // NOTE break cannot start earlier, so arriving earlier does not help: finish the service before it
    case03_finishes_before_exact_time: (vec![(7., 7., 2.)], 8., (0., 100.), 3., ControlFlow::Continue(4.)),
    case04_takes_before_service: (vec![(4., 7., 2.)], 7., (0., 100.), 3., ControlFlow::Continue(5.)),
    case05_cannot_take_earlier: (vec![(4., 7., 2.)], 5.5, (0., 100.), 3., ControlFlow::Continue(4.)),
    case06_fits_into_waiting: (vec![(7., 7., 2.)], 9., (9., 100.), 3., ControlFlow::Continue(9.)),
    case07_cannot_fit_into_waiting: (vec![(7., 7., 2.)], 8., (9., 100.), 3., ControlFlow::Break(4.)),
    case08_cannot_finish_before: (vec![(7., 7., 2.)], 8., (5., 100.), 3., ControlFlow::Break(4.)),
    case09_after_two: (vec![(6., 6., 2.), (9., 9., 2.)], 11., (0., 100.), 3., ControlFlow::Continue(11.)),
    case10_before_two: (vec![(6., 6., 2.), (9., 9., 2.)], 10., (0., 100.), 3., ControlFlow::Continue(3.)),
    case11_between_two: (vec![(6., 6., 2.), (20., 20., 2.)], 19., (0., 100.), 3., ControlFlow::Continue(17.)),
    case12_takes_first_and_finishes_before_second: (vec![(4., 6., 2.), (8., 9., 2.)], 9.5, (0., 100.), 3., ControlFlow::Continue(4.)),
}

#[test]
fn can_estimate_latest_arrival_as_the_opposite_of_service_start() {
    // NOTE half steps are used to get the times which do not match reserved time bounds
    let times = |from: i32, to: i32| (from * 2..=to * 2).map(|time| time as Timestamp / 2.);
    let slots_variants = [
        vec![create_slot(10., 10., 3.)],
        vec![create_slot(6., 10., 3.)],
        vec![create_slot(6., 10., 0.5)],
        vec![create_slot(4., 4., 2.), create_slot(10., 10., 3.)],
        vec![create_slot(3., 5., 2.), create_slot(8., 11., 1.)],
        vec![create_slot(2., 2., 1.), create_slot(4., 6., 2.), create_slot(12., 12., 4.)],
    ];

    for slots in slots_variants.iter() {
        // NOTE no arrival falls on a reserved time which has started already, as travel is delayed by it
        let is_reachable = |arrival: Timestamp| slots.iter().all(|s| arrival <= s.latest || arrival >= s.latest_end());

        for duration in [0.5, 1., 3., 6.] {
            for tw_start in [0., 5., 9., 11., 14.] {
                let tw = TimeWindow::new(tw_start, 100.);

                for latest_service_start in times(0, 24) {
                    let is_in_time = |arrival: Timestamp| {
                        is_reachable(arrival)
                            && get_service_start(slots, arrival, &tw, duration) <= latest_service_start
                    };
                    let message = format!("{slots:?}, duration={duration}, tw={tw:?}, start={latest_service_start}");

                    match get_latest_arrival(slots, latest_service_start, &tw, duration) {
                        ControlFlow::Continue(latest_arrival) if latest_service_start >= tw.start => {
                            assert!(is_in_time(latest_arrival), "arriving at {latest_arrival} is late: {message}");
                            // NOTE an earlier arrival is in time too, while a later one is not
                            assert!(
                                times(-8, 24).all(|arrival| !is_reachable(arrival)
                                    || is_in_time(arrival) == (arrival <= latest_arrival)),
                                "{latest_arrival} is not the latest arrival: {message}"
                            );
                        }
                        // NOTE service cannot start before time window whenever the vehicle arrives
                        _ => assert!(times(-8, 24).all(|arrival| !is_in_time(arrival)), "can be in time: {message}"),
                    }
                }
            }
        }
    }
}
