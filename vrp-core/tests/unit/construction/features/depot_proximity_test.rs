use super::*;
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::problem::{TestSingleBuilder, TestTransportCost, test_multi_job_with_locations};
use crate::helpers::models::solution::*;

// The default actor's depot (shift start) is at location 0 and `TestTransportCost` reports the
// travel duration between two locations as their absolute difference, so a job at location `L` is
// scored `L` from the depot.

fn create_objective() -> Arc<dyn FeatureObjective> {
    create_objective_with_cap(None)
}

fn create_objective_with_cap(cap: Option<Cost>) -> Arc<dyn FeatureObjective> {
    create_minimize_depot_travel_time_feature("min_depot_travel", TestTransportCost::new_shared(), Arc::new(|d| d), cap)
        .unwrap()
        .objective
        .unwrap()
}

parameterized_test! {can_estimate_depot_to_job_duration, (location, expected), {
    can_estimate_depot_to_job_duration_impl(location, expected);
}}

can_estimate_depot_to_job_duration! {
    case_01_near: (Some(3), 3.),
    case_02_far: (Some(30), 30.),
    case_03_at_depot: (Some(0), 0.),
    case_04_no_location: (None, 0.),
}

fn can_estimate_depot_to_job_duration_impl(location: Option<Location>, expected: Cost) {
    let objective = create_objective();
    let route_ctx = RouteContextBuilder::default().build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let job = TestSingleBuilder::default().location(location).build_as_job_ref();

    let result = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job));

    assert_eq!(result, expected);
}

#[test]
fn can_estimate_nearest_of_multiple_places_for_single_job() {
    let objective = create_objective();
    let route_ctx = RouteContextBuilder::default().build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let job = TestSingleBuilder::default()
        .places(vec![(Some(30), 0., vec![]), (Some(7), 0., vec![]), (Some(15), 0., vec![])])
        .build_as_job_ref();

    let result = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job));

    assert_eq!(result, 7.);
}

#[test]
fn can_estimate_nearest_of_multiple_singles_for_multi_job() {
    let objective = create_objective();
    let route_ctx = RouteContextBuilder::default().build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let job = Job::Multi(test_multi_job_with_locations(vec![vec![Some(30)], vec![Some(9)]]));

    let result = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job));

    assert_eq!(result, 9.);
}

#[test]
fn can_score_far_job_higher_than_near_job() {
    let objective = create_objective();
    let route_ctx = RouteContextBuilder::default().build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let near = TestSingleBuilder::default().location(Some(5)).build_as_job_ref();
    let far = TestSingleBuilder::default().location(Some(50)).build_as_job_ref();

    let near_cost = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &near));
    let far_cost = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &far));

    assert!(far_cost > near_cost, "expected far job ({far_cost}) to be scored higher than near job ({near_cost})");
}

#[test]
fn can_ignore_activity_context() {
    let objective = create_objective();
    // Use a route with a start activity (the depot) so the activity context has a valid `prev`.
    let route_ctx = RouteContextBuilder::default().with_route(RouteBuilder::default().build()).build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let target = ActivityBuilder::with_location(40).build();
    let activity_ctx =
        ActivityContext { index: 1, prev: route_ctx.route().tour.get(0).unwrap(), target: &target, next: None };

    let result = objective.estimate(&MoveContext::activity(&solution_ctx, &route_ctx, &activity_ctx));

    assert_eq!(result, Cost::default());
}

#[test]
fn can_sum_fitness_over_routes() {
    let objective = create_objective();
    let route_ctx = RouteContextBuilder::default()
        .with_route(
            RouteBuilder::default()
                .add_activity(ActivityBuilder::with_location(4).build())
                .add_activity(ActivityBuilder::with_location(6).build())
                .build(),
        )
        .build();
    let insertion_ctx = TestInsertionContextBuilder::default().with_routes(vec![route_ctx]).build();

    let result = objective.fitness(&insertion_ctx);

    assert_eq!(result, 10.);
}

#[test]
fn capped_estimate_rewards_near_job_and_penalises_far_job() {
    // cap = 10: a job within the radius gets a negative (rewarded) insertion cost, a job beyond it
    // a positive (discouraged) one. The break-even is exactly at the cap distance.
    let objective = create_objective_with_cap(Some(10.));
    let route_ctx = RouteContextBuilder::default().build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;

    let near = TestSingleBuilder::default().location(Some(3)).build_as_job_ref();
    let far = TestSingleBuilder::default().location(Some(30)).build_as_job_ref();

    let near_cost = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &near));
    let far_cost = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &far));

    assert_eq!(near_cost, 3. - 10.); // -7: within radius, rewarded
    assert_eq!(far_cost, 30. - 10.); // +20: beyond radius, discouraged
}

#[test]
fn capped_fitness_penalises_unassigned_jobs() {
    // One assigned job at location 4 (travel 4) plus two unassigned jobs, cap = 10.
    // fitness = 4 (assigned travel) + 2 * 10 (unassigned penalty) = 24.
    let objective = create_objective_with_cap(Some(10.));
    let route_ctx = RouteContextBuilder::default()
        .with_route(RouteBuilder::default().add_activity(ActivityBuilder::with_location(4).build()).build())
        .build();
    let insertion_ctx = TestInsertionContextBuilder::default()
        .with_routes(vec![route_ctx])
        .with_unassigned(vec![
            (TestSingleBuilder::default().id("u1").build_as_job_ref(), UnassignmentInfo::Unknown),
            (TestSingleBuilder::default().id("u2").build_as_job_ref(), UnassignmentInfo::Unknown),
        ])
        .build();

    let result = objective.fitness(&insertion_ctx);

    assert_eq!(result, 24.);
}
