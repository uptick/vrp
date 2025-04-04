use super::*;
use crate::construction::heuristics::UnassignmentInfo;
use crate::helpers::construction::clustering::vicinity::*;
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::domain::*;
use crate::helpers::models::problem::*;
use crate::helpers::models::solution::*;
use crate::helpers::solver::create_default_refinement_ctx;
use crate::models::problem::{Job, JobIdDimension};
use crate::models::solution::{Commute, CommuteInfo};
use crate::prelude::ViolationCode;

fn create_test_jobs() -> Vec<Job> {
    vec![
        TestSingleBuilder::default().id("job1").location(Some(1)).duration(2.).build_as_job_ref(),
        TestSingleBuilder::default().id("job2").location(Some(2)).duration(2.).build_as_job_ref(),
        TestSingleBuilder::default().id("job3").location(Some(3)).duration(2.).build_as_job_ref(),
        TestSingleBuilder::default().id("job4_outlier").location(Some(20)).duration(2.).build_as_job_ref(),
    ]
}

fn create_problems(config: ClusterConfig, jobs: Vec<Job>) -> (Arc<Problem>, Arc<Problem>) {
    let environment = Arc::new(Environment::default());

    let orig_problem =
        ProblemBuilder::default().with_goal(create_goal_context_with_vicinity(vec![])).with_jobs(jobs).build();
    let orig_problem = Arc::new(Problem {
        extras: Arc::new({
            let mut extras = orig_problem.extras.as_ref().clone();
            extras.set_cluster_config(Arc::new(config));
            extras
        }),
        ..orig_problem
    });

    let refinement_cxt = RefinementContext { environment, ..create_default_refinement_ctx(orig_problem.clone()) };

    let new_refinement_cxt = VicinityClustering::default().pre_process(refinement_cxt);

    (orig_problem, new_refinement_cxt.problem)
}

#[test]
fn can_create_problem_with_clusters_on_pre_process() {
    let (_, problem) = create_problems(create_cluster_config(), create_test_jobs());

    let jobs = problem.jobs.all().iter().collect::<Vec<_>>();
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().any(|job| get_job_id(job) == "job4_outlier"));
    let jobs = jobs
        .iter()
        .find(|job| get_job_id(job) == "job3")
        .and_then(|job| job.dimens().get_cluster_info().cloned())
        .unwrap()
        .into_iter()
        .map(|info| get_job_id(&info.job).clone())
        .collect::<Vec<_>>();
    assert_eq!(jobs, vec!["job3".to_string(), "job2".to_string(), "job1".to_string()]);
}

parameterized_test! {can_unwrap_clusters_in_route_on_post_process, (visiting, duration, expected), {
    can_unwrap_clusters_in_route_on_post_process_impl(visiting, duration, expected);
}}

can_unwrap_clusters_in_route_on_post_process! {
    case_01: (VisitPolicy::ClosedContinuation, 10., vec![("job3", (3., 5.)), ("job2", (5., 8.)), ("job1", (8., 13.))]),
    case_02: (VisitPolicy::OpenContinuation, 8., vec![("job3", (3., 5.)), ("job2", (5., 8.)), ("job1", (8., 11.))]),
    case_03: (VisitPolicy::Return, 12., vec![("job3", (3., 5.)), ("job2", (5., 9.)), ("job1", (9., 15.))]),
}

fn can_unwrap_clusters_in_route_on_post_process_impl(
    visiting: VisitPolicy,
    duration: Float,
    expected: Vec<(&str, (Float, Float))>,
) {
    let problem_jobs = create_test_jobs();
    let (_, new_problem) = create_problems(ClusterConfig { visiting, ..create_cluster_config() }, problem_jobs);
    let clustered_single =
        new_problem.jobs.all().iter().find(|job| get_job_id(job) == "job3").unwrap().to_single().clone();
    let clustered_time = clustered_single.places.first().unwrap().clone().times.first().unwrap().to_time_window(0.);
    let insertion_ctx = InsertionContext {
        problem: new_problem.clone(),
        ..TestInsertionContextBuilder::default()
            .with_routes(vec![
                RouteContextBuilder::default()
                    .with_route(
                        RouteBuilder::default()
                            .with_vehicle(new_problem.fleet.as_ref(), "v1")
                            .with_start(ActivityBuilder::default().schedule(Schedule::new(0., 0.)).job(None).build())
                            .with_end(ActivityBuilder::default().schedule(Schedule::new(0., 0.)).job(None).build())
                            .add_activity(Activity {
                                place: Place {
                                    idx: 0,
                                    location: 3,
                                    duration: DEFAULT_JOB_DURATION * 3.,
                                    time: clustered_time,
                                },
                                schedule: Schedule::new(3., 3. + duration),
                                job: Some(clustered_single),
                                commute: Some(Commute {
                                    forward: CommuteInfo { location: 3, duration: 0., distance: 0. },
                                    backward: CommuteInfo { location: 3, duration: 0., distance: 0. },
                                }),
                            })
                            .build(),
                    )
                    .build(),
            ])
            .build()
    };

    let insertion_ctx = VicinityClustering::default().post_process(insertion_ctx);

    assert_eq!(insertion_ctx.problem.jobs.size(), 4);
    assert_eq!(insertion_ctx.solution.routes.len(), 1);
    let route_ctx = insertion_ctx.solution.routes.first().unwrap();
    assert_eq!(route_ctx.route().tour.job_activity_count(), 3);
    assert_eq!(route_ctx.route().tour.total(), 5);
    let job_activities = route_ctx.route().tour.all_activities().skip(1).take(3).collect::<Vec<_>>();
    assert_eq!(job_activities.len(), expected.len());
    job_activities.into_iter().zip(expected).for_each(|(activity, (id, (arrival, departure)))| {
        assert_eq!(activity.job.as_ref().unwrap().dimens.get_job_id().unwrap(), id);
        assert_eq!(activity.schedule.arrival, arrival);
        assert_eq!(activity.schedule.departure, departure);
    });
}

#[test]
fn can_unwrap_clusters_in_unassigned_on_post_process() {
    let (_, new_problem) = create_problems(create_cluster_config(), create_test_jobs());
    let clustered_job = new_problem.jobs.all().iter().find(|job| get_job_id(job) == "job3").unwrap().clone();
    let unclustered_job = new_problem.jobs.all().iter().find(|job| get_job_id(job) == "job4_outlier").unwrap().clone();
    let insertion_ctx = InsertionContext {
        problem: new_problem,
        ..TestInsertionContextBuilder::default()
            .with_unassigned(
                vec![
                    (clustered_job, UnassignmentInfo::Simple(ViolationCode(1))),
                    (unclustered_job, UnassignmentInfo::Simple(ViolationCode(2))),
                ]
                .into_iter()
                .collect(),
            )
            .build()
    };

    let insertion_ctx = VicinityClustering::default().post_process(insertion_ctx);

    assert_eq!(insertion_ctx.solution.unassigned.len(), 4);
}
