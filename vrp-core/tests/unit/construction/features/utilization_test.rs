use super::*;
use crate::construction::clustering::vicinity::ClusterInfo;
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::problem::{DEFAULT_ACTIVITY_TIME_WINDOW, TestSingleBuilder, TestTransportCost};
use crate::helpers::models::solution::*;
use crate::models::solution::Commute;

// The default actor starts and ends its shift at location 0, and `TestTransportCost` reports the
// travel duration between two locations as their absolute difference. Jobs are named after their
// value, so a job without a numeric id has no value.

const BREAK_ID: &str = "break";

#[derive(Clone, Copy)]
enum Kind {
    Utilization,
    NetValue,
}

fn create_objective(kind: Kind, travel_penalty: Float) -> Arc<dyn FeatureObjective> {
    create_objective_with_parking(kind, travel_penalty, 0.)
}

fn create_objective_with_parking(
    kind: Kind,
    travel_penalty: Float,
    cluster_parking: Duration,
) -> Arc<dyn FeatureObjective> {
    let transport = TestTransportCost::new_shared();
    let feature = match kind {
        Kind::Utilization => create_maximize_utilization_feature(
            "max_utilization",
            travel_penalty,
            transport,
            Arc::new(|single| single.dimens.get_job_id().is_none_or(|id| id != BREAK_ID)),
            cluster_parking,
        ),
        Kind::NetValue => create_maximize_net_value_feature(
            "max_net_value",
            travel_penalty,
            transport,
            Arc::new(|job| job.dimens().get_job_id().and_then(|id| id.parse::<Float>().ok()).unwrap_or(0.)),
            cluster_parking,
        ),
    };

    feature.unwrap().objective.unwrap()
}

/// A job activity whose job is worth `value` and takes `duration` to serve.
fn job_activity(location: Location, value: Float, duration: Duration) -> Activity {
    let job =
        TestSingleBuilder::default().id(&value.to_string()).location(Some(location)).duration(duration).build_shared();

    ActivityBuilder::with_location_tw_and_duration(location, DEFAULT_ACTIVITY_TIME_WINDOW, duration)
        .job(Some(job))
        .build()
}

fn create_insertion_ctx(activities: Vec<Activity>) -> InsertionContext {
    let route_ctx =
        RouteContextBuilder::default().with_route(RouteBuilder::default().add_activities(activities).build()).build();

    TestInsertionContextBuilder::default().with_routes(vec![route_ctx]).build()
}

parameterized_test! {can_score_tour, (kind, travel_penalty, expected), {
    // 0 -> 10 -> 30 -> 0 drives 60, takes 100 + 50 to serve and is worth 7 + 3
    let insertion_ctx = create_insertion_ctx(vec![job_activity(10, 7., 100.), job_activity(30, 3., 50.)]);

    assert_eq!(create_objective(kind, travel_penalty).fitness(&insertion_ctx), expected);
}}

can_score_tour! {
    case_01_utilization_no_penalty: (Kind::Utilization, 0., -150.),
    case_02_utilization: (Kind::Utilization, 0.25, 0.25 * 60. - 150.),
    case_03_utilization_travel_outweighs_service: (Kind::Utilization, 5., 5. * 60. - 150.),
    case_04_net_value: (Kind::NetValue, 0.5, 0.5 * 60. - 10.),
}

#[test]
fn can_ignore_tour_without_jobs() {
    let route_ctx = RouteContextBuilder::default()
        .with_route(
            RouteBuilder::default()
                .with_start(ActivityBuilder::with_location(0).job(None).build())
                .with_end(ActivityBuilder::with_location(50).job(None).build())
                .build(),
        )
        .build();
    let insertion_ctx = TestInsertionContextBuilder::default().with_routes(vec![route_ctx]).build();

    assert_eq!(create_objective(Kind::Utilization, 1.).fitness(&insertion_ctx), 0.);
}

parameterized_test! {can_count_break_as_travel_stop_only, (kind, expected), {
    let break_activity = ActivityBuilder::with_location_tw_and_duration(20, DEFAULT_ACTIVITY_TIME_WINDOW, 30.)
        .job(Some(TestSingleBuilder::default().id(BREAK_ID).duration(30.).build_shared()))
        .build();
    let insertion_ctx = create_insertion_ctx(vec![job_activity(10, 7., 100.), break_activity]);

    // the break is neither service nor value, but the detour to reach it is still driven
    assert_eq!(create_objective(kind, 1.).fitness(&insertion_ctx), expected);
}}

can_count_break_as_travel_stop_only! {
    case_01_utilization: (Kind::Utilization, 40. - 100.),
    case_02_net_value: (Kind::NetValue, 40. - 7.),
}

parameterized_test! {can_match_fitness_difference_with_insertion_estimate, kind, {
    let objective = create_objective(kind, 0.25);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let route_ctx = RouteContextBuilder::default().build();
    let (prev, target, next) = (job_activity(10, 7., 100.), job_activity(40, 9., 70.), job_activity(30, 3., 50.));
    let job = Job::Single(target.job.clone().unwrap());
    let activity_ctx = ActivityContext { index: 2, prev: &prev, target: &target, next: Some(&next) };

    let estimate = objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job))
        + objective.estimate(&MoveContext::activity(&solution_ctx, &route_ctx, &activity_ctx));
    let before = objective.fitness(&create_insertion_ctx(vec![prev.clone(), next.clone()]));
    let after = objective.fitness(&create_insertion_ctx(vec![prev, target, next]));

    assert_eq!(after - before, estimate);
}}

can_match_fitness_difference_with_insertion_estimate! {
    case_01_utilization: Kind::Utilization,
    case_02_net_value: Kind::NetValue,
}

#[test]
fn can_estimate_net_value_once_per_job() {
    let objective = create_objective(Kind::NetValue, 0.);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let route_ctx = RouteContextBuilder::default().build();
    let (prev, target) = (job_activity(10, 0., 0.), job_activity(25, 80., 50.));
    let job = Job::Single(target.job.clone().unwrap());
    let activity_ctx = ActivityContext { index: 1, prev: &prev, target: &target, next: None };

    // so a job with several activities is not credited for each of them
    assert_eq!(objective.estimate(&MoveContext::route(&solution_ctx, &route_ctx, &job)), -80.);
    assert_eq!(objective.estimate(&MoveContext::activity(&solution_ctx, &route_ctx, &activity_ctx)), 0.);
}

#[test]
fn can_estimate_insertion_at_end_of_open_tour() {
    let objective = create_objective(Kind::Utilization, 1.);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let route_ctx = RouteContextBuilder::default().build();
    let (prev, target) = (job_activity(10, 0., 0.), job_activity(25, 0., 50.));
    let activity_ctx = ActivityContext { index: 1, prev: &prev, target: &target, next: None };

    // only the leg in
    assert_eq!(objective.estimate(&MoveContext::activity(&solution_ctx, &route_ctx, &activity_ctx)), 15. - 50.);
}

parameterized_test! {can_score_dropping_job_against_its_detour, (kind, travel_penalty, expected_drop_is_better), {
    // `far` adds 60 of travel (0 -> 10 -> 40 -> 0 against 0 -> 10 -> 0), takes 50 and is worth 50
    let objective = create_objective(kind, travel_penalty);
    let served = create_insertion_ctx(vec![job_activity(10, 100., 100.), job_activity(40, 50., 50.)]);
    let dropped = create_insertion_ctx(vec![job_activity(10, 100., 100.)]);

    assert_eq!(objective.fitness(&dropped) < objective.fitness(&served), expected_drop_is_better);
}}

can_score_dropping_job_against_its_detour! {
    case_01_utilization_detour_outweighs_service: (Kind::Utilization, 1., true),
    case_02_utilization_just_under_break_even: (Kind::Utilization, 0.8, false),
    case_03_net_value_detour_outweighs_value: (Kind::NetValue, 1., true),
    case_04_net_value_just_under_break_even: (Kind::NetValue, 0.8, false),
}

// Clusters: a stop at location 10 where two jobs are served on foot. The first takes 30 plus 5 of
// parking, the second 20, and walking between them takes 10, so the stop lasts 65. It is worth 40.

const CLUSTER_PARKING: Duration = 5.;

fn cluster_activity() -> Activity {
    let member = |id: &str, service_time: Duration| ClusterInfo {
        job: TestSingleBuilder::default().id(id).build_as_job_ref(),
        service_time,
        place_idx: 0,
        commute: Commute::default(),
    };
    let mut cluster = TestSingleBuilder::default();
    cluster.id("40").location(Some(10)).duration(65.);
    cluster.dimens_mut().set_cluster_info(vec![member("first", 30. + CLUSTER_PARKING), member("second", 20.)]);

    ActivityBuilder::with_location_tw_and_duration(10, DEFAULT_ACTIVITY_TIME_WINDOW, 65.)
        .job(Some(cluster.build_shared()))
        .build()
}

parameterized_test! {can_count_walking_and_parking_in_cluster_as_travel, (kind, cluster_parking, expected), {
    let insertion_ctx = create_insertion_ctx(vec![cluster_activity()]);

    assert_eq!(create_objective_with_parking(kind, 1., cluster_parking).fitness(&insertion_ctx), expected);
}}

can_count_walking_and_parking_in_cluster_as_travel! {
    // driving 20 plus walking 10 and parking 5, against 50 of service
    case_01_utilization: (Kind::Utilization, CLUSTER_PARKING, (20. + 10. + 5.) - 50.),
    // without the parking time, it is mistaken for service
    case_02_utilization_parking_unknown: (Kind::Utilization, 0., (20. + 10.) - 55.),
    case_03_net_value: (Kind::NetValue, CLUSTER_PARKING, (20. + 10. + 5.) - 40.),
}

#[test]
fn can_estimate_cluster_insertion_with_walking_and_parking_as_travel() {
    let objective = create_objective_with_parking(Kind::Utilization, 1., CLUSTER_PARKING);
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let route_ctx = RouteContextBuilder::default().build();
    let (prev, target, next) = (job_activity(0, 0., 0.), cluster_activity(), job_activity(0, 0., 0.));
    let activity_ctx = ActivityContext { index: 1, prev: &prev, target: &target, next: Some(&next) };

    let result = objective.estimate(&MoveContext::activity(&solution_ctx, &route_ctx, &activity_ctx));

    assert_eq!(result, (20. + 10. + 5.) - 50.);
}
