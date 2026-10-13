use crate::format::problem::Objective::*;
use crate::format::problem::*;
use crate::helpers::*;

#[test]
fn can_round_trip_due_date_objectives() {
    for json in [
        r#"{"type":"minimize-late-jobs"}"#,
        r#"{"type":"minimize-tardiness"}"#,
        r#"{"type":"minimize-due-window-use"}"#,
    ] {
        let objective: Objective = serde_json::from_str(json).expect("cannot deserialize objective");

        assert_eq!(serde_json::to_string(&objective).expect("cannot serialize objective"), json);
    }
}

#[test]
fn can_round_trip_job_due_date() {
    let json = r#"{"id":"job1","deliveries":[{"places":[{"location":{"lat":1.0,"lng":0.0},"duration":1.0}]}],"due":"1970-01-01T00:16:40Z"}"#;

    let job: Job = serde_json::from_str(json).expect("cannot deserialize job");
    assert_eq!(job.due.as_deref(), Some("1970-01-01T00:16:40Z"));

    let serialized = serde_json::to_string(&job).expect("cannot serialize job");
    assert_eq!(serialized, json);
}

fn create_problem(objectives: Vec<Objective>) -> Problem {
    Problem {
        plan: Plan {
            // On an open shift cost prefers job1 first (0->1->5 is 5, 0->5->1 is 9), but job2
            // arrives at 5 only when served first, and is due at 5.
            jobs: vec![
                create_delivery_job_with_due_date("job1", (1., 0.), 100.),
                create_delivery_job_with_due_date("job2", (5., 0.), 5.),
            ],
            ..create_empty_plan()
        },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![create_default_open_vehicle_shift()],
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        objectives: Some(objectives),
        ..create_empty_problem()
    }
}

fn served_job_ids(objectives: Vec<Objective>) -> Vec<String> {
    let problem = create_problem(objectives);
    let matrix = create_matrix_from_problem(&problem);

    solve_with_metaheuristic(problem, Some(vec![matrix]))
        .tours
        .first()
        .expect("expected a tour")
        .stops
        .iter()
        .flat_map(|stop| stop.activities().iter())
        .map(|activity| activity.job_id.clone())
        .filter(|job_id| job_id != "departure" && job_id != "arrival")
        .collect()
}

#[test]
fn can_serve_earlier_due_job_first() {
    let served = served_job_ids(vec![
        MinimizeUnassigned { breaks: None },
        MinimizeTours,
        MinimizeLateJobs,
        MinimizeTardiness,
        MinimizeCost,
    ]);

    assert_eq!(served, vec!["job2".to_string(), "job1".to_string()]);
}

#[test]
fn ignores_due_dates_without_the_objectives() {
    // the control for the test above
    let served = served_job_ids(vec![MinimizeUnassigned { breaks: None }, MinimizeTours, MinimizeCost]);

    assert_eq!(served, vec!["job1".to_string(), "job2".to_string()]);
}

/// Scenarios over four weeks of weekday shifts, 08:00 to 16:00, with day 0 a Monday.
mod across_a_month {
    use super::*;
    use crate::{format_time, parse_time};

    const DAY: f64 = 86400.;
    const HOUR: f64 = 3600.;

    /// The end of the given day, as a due date meaning "any time that day" is sent.
    fn end_of_day(day: usize) -> f64 {
        (day + 1) as f64 * DAY - 1.
    }

    fn job(id: &str, x: f64, hours: f64, due: Option<f64>, not_before: Option<f64>) -> Job {
        let place = JobPlace {
            duration: hours * HOUR,
            times: not_before.map(|start| vec![vec![format_time(start), format_time(100. * DAY)]]),
            ..create_job_place((x, 0.), None)
        };

        Job {
            services: Some(vec![JobTask { places: vec![place], demand: None, order: None }]),
            due: due.map(format_time),
            ..create_job(id)
        }
    }

    /// Undated jobs near the depot which take a whole shift each.
    fn fillers(count: usize) -> impl Iterator<Item = Job> {
        (0..count).map(|idx| job(&format!("filler{idx}"), 1., 7., None, None))
    }

    fn solve(jobs: Vec<Job>) -> (Vec<(String, usize)>, Vec<String>) {
        let objectives = vec![
            MinimizeUnassigned { breaks: None },
            MinimizeLateJobs,
            MinimizeTardiness,
            MinimizeDueWindowUse,
            MinimizeCost,
        ];

        solve_plan(Plan { jobs, ..create_empty_plan() }, objectives)
    }

    /// Returns the day each non-filler job is served on, sorted by job id, and unassigned job ids.
    fn solve_plan(plan: Plan, objectives: Vec<Objective>) -> (Vec<(String, usize)>, Vec<String>) {
        let shifts = (0..28)
            .filter(|day| day % 7 < 5)
            .map(|day| {
                let start = day as f64 * DAY + 8. * HOUR;
                VehicleShift {
                    start: ShiftStart { earliest: format_time(start), latest: None, location: (0., 0.).to_loc() },
                    end: Some(ShiftEnd {
                        earliest: None,
                        latest: format_time(start + 8. * HOUR),
                        location: (0., 0.).to_loc(),
                    }),
                    ..create_default_vehicle_shift()
                }
            })
            .collect();
        let problem = Problem {
            plan,
            fleet: Fleet {
                vehicles: vec![VehicleType { shifts, ..create_default_vehicle_type() }],
                ..create_default_fleet()
            },
            objectives: Some(objectives),
            ..create_empty_problem()
        };
        let matrix = create_matrix_from_problem(&problem);
        let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

        let mut days = solution
            .tours
            .iter()
            .flat_map(|tour| tour.stops.iter())
            .flat_map(|stop| {
                let day = (parse_time(&stop.schedule().arrival) / DAY) as usize;
                stop.activities()
                    .iter()
                    .filter(|activity| activity.activity_type == "service" && !activity.job_id.starts_with("filler"))
                    .map(move |activity| (activity.job_id.clone(), day))
            })
            .collect::<Vec<_>>();
        days.sort();
        let unassigned = solution.unassigned.unwrap_or_default().into_iter().map(|job| job.job_id).collect();

        (days, unassigned)
    }

    #[test]
    fn spreads_weeklies_across_their_weeks() {
        // A due date alone puts all four on one visit, each on time. With its week as a time
        // window too, window use serves each at the start of its week.
        let jobs = (0..4)
            .map(|week| {
                let monday = (7 * week) as f64 * DAY;
                job(&format!("weekly{week}"), 5., 1., Some(end_of_day(7 * week + 4)), Some(monday))
            })
            .collect();

        let days = solve(jobs).0.into_iter().map(|(_, day)| day).collect::<Vec<_>>();

        assert_eq!(days, vec![0, 7, 14, 21]);
    }

    #[test]
    fn serves_a_dated_job_in_the_first_shift() {
        let jobs = std::iter::once(job("annual", 50., 7., Some(end_of_day(17)), None)).chain(fillers(5)).collect();

        assert_eq!(solve(jobs).0, vec![("annual".to_string(), 0)]);
    }

    parameterized_test! {keeps_dated_work_over_undated_work_in_a_full_month, due, {
        // 21 jobs for 20 shifts, and cost alone would drop the dated one as the furthest away
        let jobs = std::iter::once(job("dated", 50., 7., Some(due), None)).chain(fillers(20)).collect();

        let (days, unassigned) = solve(jobs);

        assert_eq!(days.len(), 1, "{unassigned:?}");
    }}

    keeps_dated_work_over_undated_work_in_a_full_month! {
        case_01_overdue: -5. * DAY,
        case_02_due_after_the_last_shift: end_of_day(25),
        case_03_due_a_week_after_the_plan: end_of_day(32),
    }

    #[test]
    fn keeps_a_job_due_after_the_last_shift_with_a_horizon() {
        // The job is due after the plan's last shift ends, so without a horizon dropping it is
        // not late. Window use would keep it anyway, so it is left out to test the horizon alone.
        let jobs = std::iter::once(job("dated", 50., 7., Some(end_of_day(25)), None)).chain(fillers(20)).collect();
        let plan = Plan { jobs, horizon: Some(Horizon { end: format_time(26. * DAY) }), ..create_empty_plan() };

        let (days, _) = solve_plan(
            plan,
            vec![MinimizeUnassigned { breaks: None }, MinimizeLateJobs, MinimizeTardiness, MinimizeCost],
        );

        assert_eq!(days.len(), 1);
    }
}
