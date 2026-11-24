use crate::format::problem::Objective::*;
use crate::format::problem::*;
use crate::format_time;
use crate::helpers::*;

parameterized_test! {can_round_trip_objective, json, {
    let objective: Objective = serde_json::from_str(json).expect("cannot deserialize objective");

    assert_eq!(serde_json::to_string(&objective).expect("cannot serialize objective"), json);
}}

can_round_trip_objective! {
    case_01_utilization: r#"{"type":"maximize-utilization","travelPenalty":0.25}"#,
    case_02_net_value: r#"{"type":"maximize-net-value","travelPenalty":0.25}"#,
}

fn create_valued_job(id: &str, location: (f64, f64), value: f64, duration: f64) -> Job {
    Job { value: Some(value), ..create_delivery_job_with_duration(id, location, duration) }
}

fn create_problem(jobs: Vec<Job>, shift_end: f64, objectives: Vec<Objective>) -> Problem {
    Problem {
        plan: Plan { jobs, ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![VehicleShift {
                    end: Some(ShiftEnd { earliest: None, latest: format_time(shift_end), location: (0., 0.).to_loc() }),
                    ..create_default_vehicle_shift()
                }],
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        objectives: Some(objectives),
    }
}

fn solve_for_served_job_ids(problem: Problem) -> Vec<String> {
    let matrix = create_matrix_from_problem(&problem);
    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    let mut ids = solution
        .tours
        .iter()
        .flat_map(|tour| tour.stops.iter())
        .flat_map(|stop| stop.activities().iter())
        .map(|activity| activity.job_id.clone())
        .filter(|job_id| job_id != "departure" && job_id != "arrival")
        .collect::<Vec<_>>();
    ids.sort();

    ids
}

parameterized_test! {can_prefer_longer_job_when_only_one_fits, (objectives, expected), {
    // both jobs are 10 from the depot; serving both takes 40 travel and 400 service, past the
    // shift end at 400
    let jobs = vec![
        create_delivery_job_with_duration("short", (10., 0.), 100.),
        create_delivery_job_with_duration("long", (0., 10.), 300.),
    ];

    assert_eq!(solve_for_served_job_ids(create_problem(jobs, 400., objectives)), vec![expected]);
}}

can_prefer_longer_job_when_only_one_fits! {
    // cost alone picks the job which takes less time
    case_01_cost_only: (vec![MinimizeUnassigned { breaks: None }, MinimizeCost], "short"),
    case_02_utilization: (
        vec![MinimizeUnassigned { breaks: None }, MaximizeUtilization { travel_penalty: 0.25 }, MinimizeCost],
        "long"),
}

parameterized_test! {can_trade_value_against_travel_when_only_one_fits, (travel_penalty, expected), {
    // `near` is worth 10 for 20 of travel, `far` 12 for 200; serving both takes 200 travel and
    // 400 service, past the shift end at 450
    let jobs = vec![create_valued_job("near", (10., 0.), 10., 200.), create_valued_job("far", (100., 0.), 12., 200.)];
    let objectives = vec![MinimizeUnassigned { breaks: None }, MaximizeNetValue { travel_penalty }, MinimizeCost];

    assert_eq!(solve_for_served_job_ids(create_problem(jobs, 450., objectives)), vec![expected]);
}}

can_trade_value_against_travel_when_only_one_fits! {
    case_01_no_penalty: (0., "far"),
    case_02_penalty: (0.1, "near"),
}

#[test]
fn can_solve_with_both_objectives_in_weighted_sum() {
    // as separate types, both can sit in one goal
    let problem = create_problem(
        vec![create_valued_job("job1", (10., 0.), 10., 100.)],
        1000.,
        vec![
            MinimizeUnassigned { breaks: None },
            MultiObjective {
                strategy: MultiStrategy::WeightedSum { weights: vec![1., 1.] },
                objectives: vec![
                    MaximizeNetValue { travel_penalty: 0.1 },
                    MaximizeUtilization { travel_penalty: 0.25 },
                ],
            },
            MinimizeCost,
        ],
    );

    assert_eq!(solve_for_served_job_ids(problem), vec!["job1"]);
}
