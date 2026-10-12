use super::*;
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::problem::*;
use crate::helpers::models::solution::*;
use crate::models::common::Schedule;

// The test fleet's single shift runs from 0 to `DEFAULT_ACTOR_TIME_WINDOW.end` (1000).
const DUE_DATE: Timestamp = 100.;

fn objective_with(measure: DueDateMeasure, due_date: Option<Timestamp>, fleet: &Fleet) -> Arc<dyn FeatureObjective> {
    let job_due_date_fn: JobDueDateFn = Arc::new(move |_| due_date);

    create_due_date_feature("due_date", measure, job_due_date_fn, TestTransportCost::new_shared(), fleet)
        .unwrap()
        .objective
        .unwrap()
}

fn objective(measure: DueDateMeasure, due_date: Option<Timestamp>) -> Arc<dyn FeatureObjective> {
    objective_with(measure, due_date, &test_fleet())
}

/// A route serving each given job at the given arrival time.
fn route_serving(jobs: Vec<(Arc<Single>, Timestamp)>) -> RouteContext {
    let activities = jobs
        .into_iter()
        .map(|(job, arrival)| {
            ActivityBuilder::default().job(Some(job)).schedule(Schedule::new(arrival, arrival)).build()
        })
        .collect::<Vec<_>>();

    RouteContextBuilder::default()
        .with_route(
            RouteBuilder::default()
                .with_start(ActivityBuilder::default().job(None).build())
                .add_activities(activities)
                .build(),
        )
        .build()
}

fn route_with_arrivals(arrivals: &[Timestamp]) -> RouteContext {
    route_serving(arrivals.iter().map(|&arrival| (TestSingleBuilder::default().build_shared(), arrival)).collect())
}

fn context_with(routes: Vec<RouteContext>, dropped: bool) -> InsertionContext {
    let unassigned = if dropped {
        vec![(TestSingleBuilder::default().id("dropped").build_as_job_ref(), UnassignmentInfo::Unknown)]
    } else {
        vec![]
    };

    TestInsertionContextBuilder::default().with_routes(routes).with_unassigned(unassigned).build()
}

parameterized_test! {can_score_served_jobs, (measure, arrivals, expected), {
    let ctx = context_with(vec![route_with_arrivals(&arrivals)], false);

    assert_eq!(objective(measure, Some(DUE_DATE)).fitness(&ctx), expected);
}}

can_score_served_jobs! {
    // arriving exactly on the due date is on time
    case_01_late_jobs_on_time: (DueDateMeasure::LateJobs, vec![50., 100.], 0.),
    case_02_late_jobs_some_late: (DueDateMeasure::LateJobs, vec![50., 101., 5000.], 2.),
    case_03_tardiness_on_time: (DueDateMeasure::Tardiness, vec![50., 100.], 0.),
    // early arrival is no credit against late
    case_04_tardiness_some_late: (DueDateMeasure::Tardiness, vec![0., 110., 130.], 40.),
    case_05_window_use: (DueDateMeasure::WindowUse, vec![0., 50.], 0.5),
    case_06_window_use_capped_when_late: (DueDateMeasure::WindowUse, vec![150.], 1.),
}

#[test]
fn can_disagree_between_late_jobs_and_tardiness() {
    // why the count ranks above the sum: it prefers one very late job to three slightly late
    let one_very_late = context_with(vec![route_with_arrivals(&[50., 50., 1000.])], false);
    let three_slightly_late = context_with(vec![route_with_arrivals(&[110., 110., 110.])], false);

    let late_jobs = objective(DueDateMeasure::LateJobs, Some(DUE_DATE));
    let tardiness = objective(DueDateMeasure::Tardiness, Some(DUE_DATE));

    assert!(late_jobs.fitness(&one_very_late) < late_jobs.fitness(&three_slightly_late));
    assert!(tardiness.fitness(&three_slightly_late) < tardiness.fitness(&one_very_late));
}

parameterized_test! {can_score_a_dropped_job_as_arriving_at_the_plan_end, (measure, due_date, expected), {
    assert_eq!(objective(measure, Some(due_date)).fitness(&context_with(vec![], true)), expected);
}}

can_score_a_dropped_job_as_arriving_at_the_plan_end! {
    case_01_late_jobs_due_in_plan: (DueDateMeasure::LateJobs, DUE_DATE, 1.),
    // a later plan can still serve it on time
    case_02_late_jobs_due_after_plan: (DueDateMeasure::LateJobs, 2000., 0.),
    case_03_tardiness_due_in_plan: (DueDateMeasure::Tardiness, DUE_DATE, 900.),
    case_04_window_use_due_in_plan: (DueDateMeasure::WindowUse, DUE_DATE, 1.),
    case_05_window_use_due_soon_after_plan: (DueDateMeasure::WindowUse, 1250., 0.8),
    case_06_window_use_due_long_after_plan: (DueDateMeasure::WindowUse, 4000., 0.25),
}

#[test]
fn can_end_an_open_ended_plan_at_its_latest_arrival() {
    // no served job is ever scored as arriving after a dropped one
    let fleet = FleetBuilder::default().add_driver(test_driver()).add_vehicle(test_ovrp_vehicle("v1")).build();
    let ctx = context_with(vec![route_with_arrivals(&[300., 500.])], true);

    let tardiness = objective_with(DueDateMeasure::Tardiness, Some(DUE_DATE), &fleet);

    assert_eq!(tardiness.fitness(&ctx), 200. + 400. + 400.);
}

parameterized_test! {can_open_the_window_at_the_jobs_time_window, (window_start, arrival, expected), {
    let job = TestSingleBuilder::default().times(vec![TimeWindow::new(window_start, 1000.)]).build_shared();
    let ctx = context_with(vec![route_serving(vec![(job, arrival)])], false);

    assert_eq!(objective(DueDateMeasure::WindowUse, Some(DUE_DATE)).fitness(&ctx), expected);
}}

can_open_the_window_at_the_jobs_time_window! {
    case_01_halfway_through_a_later_window: (60., 80., 0.5),
    // due before it may be served: it is overdue, and there is no window to use
    case_02_window_opens_after_due: (200., 200., 1.),
}

#[test]
fn can_ignore_jobs_without_a_due_date() {
    let ctx = context_with(vec![route_with_arrivals(&[5000.])], true);

    for measure in [DueDateMeasure::LateJobs, DueDateMeasure::Tardiness, DueDateMeasure::WindowUse] {
        assert_eq!(objective(measure, None).fitness(&ctx), 0.);
    }
}

parameterized_test! {can_estimate_an_insertion, (measure, prev_departure, expected), {
    // TestTransportCost adds no travel between identical locations, so the target arrives when
    // the previous activity departs
    let route_ctx = RouteContextBuilder::default().build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let prev = ActivityBuilder::default().job(None).schedule(Schedule::new(prev_departure, prev_departure)).build();
    let target = ActivityBuilder::default().job(Some(TestSingleBuilder::default().build_shared())).build();
    let activity_ctx = ActivityContext { index: 0, prev: &prev, target: &target, next: None };

    let estimate = objective(measure, Some(DUE_DATE)).estimate(&MoveContext::activity(&solution_ctx, &route_ctx, &activity_ctx));

    assert_eq!(estimate, expected);
}}

can_estimate_an_insertion! {
    case_01_late_jobs_on_time: (DueDateMeasure::LateJobs, 100., 0.),
    case_02_late_jobs_late: (DueDateMeasure::LateJobs, 160., 1.),
    case_03_tardiness: (DueDateMeasure::Tardiness, 160., 60.),
    case_04_window_use: (DueDateMeasure::WindowUse, 25., 0.25),
}

#[test]
fn can_skip_estimating_a_route_level_move() {
    // arrival is not known until a position in the route is proposed
    let route_ctx = RouteContextBuilder::default().build();
    let solution_ctx = TestInsertionContextBuilder::default().build().solution;
    let job = TestSingleBuilder::default().build_as_job_ref();

    let estimate = objective(DueDateMeasure::Tardiness, Some(DUE_DATE)).estimate(&MoveContext::route(
        &solution_ctx,
        &route_ctx,
        &job,
    ));

    assert_eq!(estimate, 0.);
}
