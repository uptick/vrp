use super::*;
use crate::format::problem::Objective::*;

parameterized_test! {can_solve_clustered_jobs_with_travel_penalised_objectives, objectives, {
    can_solve_clustered_jobs_with_travel_penalised_objectives_impl(objectives);
}}

can_solve_clustered_jobs_with_travel_penalised_objectives! {
    case_01_utilization: vec![
        MinimizeUnassigned { breaks: None },
        MaximizeValue { breaks: None },
        MaximizeUtilization { travel_penalty: 0.25 },
        MinimizeCost,
    ],
    case_02_net_value: vec![
        MinimizeUnassigned { breaks: None },
        MaximizeNetValue { travel_penalty: 0.25 },
        MinimizeCost,
    ],
}

fn can_solve_clustered_jobs_with_travel_penalised_objectives_impl(objectives: Vec<Objective>) {
    let mut problem = create_test_problem(
        &[(1., "delivery"), (2., "delivery"), (3., "delivery"), (10., "delivery")],
        10,
        Clustering::Vicinity {
            profile: VehicleProfile { matrix: "car".to_string(), scale: None },
            threshold: VicinityThresholdPolicy {
                duration: 3.,
                distance: 3.,
                min_shared_time: None,
                smallest_time_window: None,
                max_jobs_per_cluster: None,
            },
            visiting: VicinityVisitPolicy::Continue,
            serving: VicinityServingPolicy::Original { parking: 2. },
            filtering: None,
        },
    );
    problem.plan.jobs.iter_mut().for_each(|job| job.value = Some(1.));
    problem.objectives = Some(objectives);
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(solution.unassigned.is_none(), "unassigned jobs: {:?}", solution.unassigned);
    let has_walked = solution
        .tours
        .iter()
        .flat_map(|tour| tour.stops.iter())
        .flat_map(|stop| stop.activities().iter())
        .any(|activity| activity.commute.is_some());
    assert!(has_walked, "no jobs were clustered");
}
