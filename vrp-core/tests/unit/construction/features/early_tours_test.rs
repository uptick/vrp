use super::*;
use crate::construction::enablers::FirstJobArrivalFloorDimension;
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::problem::*;
use crate::helpers::models::solution::*;
use crate::models::common::TimeInterval;
use crate::models::problem::{VehicleDetail, VehiclePlace};
use std::collections::HashSet;

const DAY: Timestamp = 86400.;

/// Builds a fleet where vehicle `i` is a shift starting on day `i`, mimicking a multi-day plan.
fn create_fleet(days: usize) -> Fleet {
    (0..days)
        .fold(FleetBuilder::default().add_driver(test_driver()), |builder, day| {
            builder.add_vehicle(
                TestVehicleBuilder::default()
                    .id(format!("day_{day}").as_str())
                    .details(vec![VehicleDetail {
                        start: Some(VehiclePlace {
                            location: 0,
                            time: TimeInterval { earliest: Some(day as Timestamp * DAY), latest: None },
                        }),
                        end: None,
                    }])
                    .build(),
            )
        })
        .build()
}

fn create_objective(fleet: &Fleet) -> Arc<dyn FeatureObjective> {
    create_objective_with(fleet, HashSet::default())
}

fn create_objective_with(fleet: &Fleet, existing_jobs: HashSet<Job>) -> Arc<dyn FeatureObjective> {
    create_prefer_early_tours_feature("prefer_early_tours", get_earliest_shift_start(fleet), existing_jobs)
        .unwrap()
        .objective
        .unwrap()
}

/// Creates a route on the given day's shift with `job_count` jobs in its tour.
fn create_route_ctx(fleet: &Fleet, day: usize, job_count: usize) -> RouteContext {
    let route = RouteBuilder::default()
        .with_vehicle(fleet, format!("day_{day}").as_str())
        .add_activities((0..job_count).map(|idx| ActivityBuilder::with_location(idx + 1).build()))
        .build();

    RouteContextBuilder::default().with_route(route).build()
}

/// Creates a job of the given kind: an `existing` one, one `bound` to a vehicle the way a break is,
/// or otherwise a plain new job the solver is free to schedule.
fn create_job(kind: &str, id: &str) -> Job {
    let mut builder = TestSingleBuilder::default();
    builder.id(id).location(Some(1));
    if kind == "bound" {
        builder.dimens_mut().set_vehicle_id("day_4".to_string());
    }

    builder.build_as_job_ref()
}

/// Creates a route on the given day's shift serving the given jobs.
fn create_route_ctx_for(fleet: &Fleet, day: usize, jobs: &[Job]) -> RouteContext {
    let route = RouteBuilder::default()
        .with_vehicle(fleet, format!("day_{day}").as_str())
        .add_activities(jobs.iter().map(|job| {
            ActivityBuilder::with_location(1).job(Some(job.as_single().expect("not a single").clone())).build()
        }))
        .build();

    RouteContextBuilder::default().with_route(route).build()
}

parameterized_test! {can_estimate_delay_of_opening_a_shift, (day, expected), {
    can_estimate_delay_of_opening_a_shift_impl(day, expected);
}}

can_estimate_delay_of_opening_a_shift! {
    case_01_earliest_shift_is_free: (0, 0.),
    case_02_next_day: (1, DAY),
    case_03_last_day: (4, 4. * DAY),
}

fn can_estimate_delay_of_opening_a_shift_impl(day: usize, expected: Cost) {
    let fleet = create_fleet(5);
    let objective = create_objective(&fleet);
    let route_ctx = create_route_ctx(&fleet, day, 0);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let job = TestSingleBuilder::default().location(Some(1)).build_as_job_ref();

    let result = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job));

    assert_eq!(result, expected);
}

parameterized_test! {can_estimate_nothing_for_a_shift_already_in_use, day, {
    can_estimate_nothing_for_a_shift_already_in_use_impl(day);
}}

can_estimate_nothing_for_a_shift_already_in_use! {
    case_01_earliest_shift: 0,
    case_02_last_day: 4,
}

fn can_estimate_nothing_for_a_shift_already_in_use_impl(day: usize) {
    let fleet = create_fleet(5);
    let objective = create_objective(&fleet);
    let route_ctx = create_route_ctx(&fleet, day, 2);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let job = TestSingleBuilder::default().location(Some(1)).build_as_job_ref();

    let result = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job));

    assert_eq!(result, 0.);
}

#[test]
fn can_ignore_activity_position_within_tour() {
    let fleet = create_fleet(5);
    let objective = create_objective(&fleet);
    let route_ctx = create_route_ctx(&fleet, 4, 2);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;

    let result = objective.estimate(&MoveContext::activity(
        &solution_ctx,
        &route_ctx,
        &ActivityContext {
            index: 0,
            prev: &ActivityBuilder::with_location(1).build(),
            target: &ActivityBuilder::with_location(2).build(),
            next: None,
        },
    ));

    assert_eq!(result, 0.);
}

parameterized_test! {can_estimate_fitness, (days_with_jobs, expected), {
    can_estimate_fitness_impl(days_with_jobs, expected);
}}

can_estimate_fitness! {
    case_01_all_work_in_first_shift: (vec![(0, 4)], 0.),
    case_02_front_loaded: (vec![(0, 2), (1, 2)], DAY),
    case_03_back_loaded: (vec![(3, 2), (4, 2)], 7. * DAY),
    // the same two days cost the same however the work is split between them: the objective picks
    // which days are worked, not what goes on each
    case_04_split_does_not_matter: (vec![(0, 1), (1, 3)], DAY),
    case_05_empty_late_route_is_free: (vec![(0, 4), (4, 0)], 0.),
}

fn can_estimate_fitness_impl(days_with_jobs: Vec<(usize, usize)>, expected: Cost) {
    let fleet = create_fleet(5);
    let objective = create_objective(&fleet);
    let routes = days_with_jobs.iter().map(|&(day, jobs)| create_route_ctx(&fleet, day, jobs)).collect();
    let insertion_ctx = TestInsertionContextBuilder::default().with_routes(routes).build();

    let result = objective.fitness(&insertion_ctx);

    assert_eq!(result, expected);
}

#[test]
fn can_use_first_job_arrival_floor_when_shift_start_is_relaxed() {
    // `allow_out_of_hours_depot_travel` relaxes the shift's start bound, so `detail.time.start`
    // collapses to zero and the floor carries the real shift start.
    let vehicle = {
        let mut builder = TestVehicleBuilder::default();
        builder.id("relaxed").details(vec![VehicleDetail {
            start: Some(VehiclePlace { location: 0, time: TimeInterval { earliest: None, latest: None } }),
            end: None,
        }]);
        builder.dimens_mut().set_first_job_arrival_floor(3. * DAY);
        builder.build()
    };
    let fleet = FleetBuilder::default().add_driver(test_driver()).add_vehicle(vehicle).build();
    let actor = get_test_actor_from_fleet(&fleet, "relaxed");

    assert_eq!(actor.detail.time.start, 0.);
    assert_eq!(get_shift_start(actor.as_ref()), 3. * DAY);
}

parameterized_test! {can_estimate_delay_only_for_work_it_can_move, (kind, expected), {
    can_estimate_delay_only_for_work_it_can_move_impl(kind, expected);
}}

can_estimate_delay_only_for_work_it_can_move! {
    case_01_new_job_pays_for_the_shift: ("new", 4. * DAY),
    case_02_existing_job_does_not: ("existing", 0.),
    case_03_vehicle_bound_job_does_not: ("bound", 0.),
}

/// The shift holds only an existing appointment, so it is not yet paid for.
fn can_estimate_delay_only_for_work_it_can_move_impl(kind: &str, expected: Cost) {
    let fleet = create_fleet(5);
    let existing = create_job("existing", "existing");
    let objective = create_objective_with(&fleet, HashSet::from([existing.clone()]));
    let route_ctx = create_route_ctx_for(&fleet, 4, std::slice::from_ref(&existing));
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let job = if kind == "existing" { existing } else { create_job(kind, "candidate") };

    let result = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job));

    assert_eq!(result, expected);
}

parameterized_test! {can_estimate_fitness_ignoring_work_it_cannot_move, (days, expected), {
    can_estimate_fitness_ignoring_work_it_cannot_move_impl(days, expected);
}}

can_estimate_fitness_ignoring_work_it_cannot_move! {
    case_01_existing_work_alone_is_free: (vec![(4, vec!["existing"])], 0.),
    case_02_vehicle_bound_work_alone_is_free: (vec![(4, vec!["bound"])], 0.),
    case_03_new_work_on_the_earliest_shift: (vec![(0, vec!["new"]), (4, vec!["existing"])], 0.),
    // the regression pair: joining the shift an appointment holds open must not beat opening an
    // earlier one, which it does when the appointment's shift is counted (4 days against 1 + 4)
    case_04_new_work_joining_the_existing_shift: (vec![(4, vec!["existing", "new"])], 4. * DAY),
    case_05_new_work_on_its_own_earlier_day: (vec![(1, vec!["new"]), (4, vec!["existing"])], DAY),
}

fn can_estimate_fitness_ignoring_work_it_cannot_move_impl(days: Vec<(usize, Vec<&str>)>, expected: Cost) {
    let fleet = create_fleet(5);
    let existing = create_job("existing", "existing");
    let objective = create_objective_with(&fleet, HashSet::from([existing.clone()]));
    let routes = days
        .into_iter()
        .map(|(day, kinds)| {
            let jobs = kinds
                .iter()
                .map(|kind| match *kind {
                    "existing" => existing.clone(),
                    kind => create_job(kind, &format!("{kind}_{day}")),
                })
                .collect::<Vec<_>>();

            create_route_ctx_for(&fleet, day, &jobs)
        })
        .collect();

    let result = objective.fitness(&TestInsertionContextBuilder::default().with_routes(routes).build());

    assert_eq!(result, expected);
}
