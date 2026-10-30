use super::*;
use crate::helpers::*;

fn assert_result(code: &str, action: &str, result: Option<FormatError>) {
    assert_eq!(result.clone().map(|err| err.code), Some(code.to_string()));
    assert!(result.map_or("".to_string(), |err| err.action).contains(action));
}

parameterized_test! {can_detect_reserved_ids, (job_id, expected), {
    can_detect_reserved_ids_impl(job_id.to_string(), expected);
}}

can_detect_reserved_ids! {
    case01: ("job1", None),
    case02: ("departure", Some("departure")),
    case03: ("arrival", Some("arrival")),
    case04: ("break", Some("break")),
    case05: ("reload", Some("reload")),
}

fn can_detect_reserved_ids_impl(job_id: String, expected: Option<&str>) {
    let problem = Problem {
        plan: Plan { jobs: vec![create_delivery_job(job_id.as_str(), (1., 0.))], ..create_empty_plan() },
        fleet: create_default_fleet(),
        ..create_empty_problem()
    };

    let result = check_e1104_no_reserved_ids(&ValidationContext::new(&problem, None, &CoordIndex::new(&problem))).err();

    if let Some(action) = expected {
        assert_result("E1104", action, result);
    } else {
        assert!(result.is_none());
    }
}

#[test]
fn can_detect_empty_job() {
    let problem = Problem {
        plan: Plan { jobs: vec![Job { deliveries: Some(vec![]), ..create_job("job1") }], ..create_empty_plan() },
        ..create_empty_problem()
    };

    let result = check_e1105_empty_jobs(&ValidationContext::new(&problem, None, &CoordIndex::new(&problem))).err();

    assert_result("E1105", "job1", result);
}

#[test]
fn can_detect_negative_duration() {
    let problem = Problem {
        plan: Plan { jobs: vec![create_delivery_job_with_duration("job1", (1., 0.), -10.)], ..create_empty_plan() },
        ..create_empty_problem()
    };

    let result =
        check_e1106_negative_duration(&ValidationContext::new(&problem, None, &CoordIndex::new(&problem))).err();

    assert_result("E1106", "job1", result);
}

#[test]
fn can_detect_negative_demand() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![create_delivery_job_with_demand("job1", (1., 0.), vec![0, -1])],
            ..create_empty_plan()
        },
        ..create_empty_problem()
    };

    let result = check_e1107_negative_demand(&ValidationContext::new(&problem, None, &CoordIndex::new(&problem))).err();

    assert_result("E1107", "job1", result);
}

#[test]
fn can_iterate_all_task_kinds_in_validation_order() {
    let task_with_duration = |duration| JobTask {
        places: vec![JobPlace { duration, ..create_job_place((duration, 0.), None) }],
        demand: None,
        order: None,
    };
    let problem = Problem {
        plan: Plan {
            jobs: vec![Job {
                pickups: Some(vec![task_with_duration(1.)]),
                deliveries: Some(vec![task_with_duration(2.)]),
                replacements: Some(vec![task_with_duration(3.)]),
                services: Some(vec![task_with_duration(4.)]),
                ..create_job("job1")
            }],
            ..create_empty_plan()
        },
        ..create_empty_problem()
    };

    let durations = job_tasks(&problem.plan.jobs[0]).map(|task| task.places[0].duration).collect::<Vec<_>>();

    assert_eq!(durations, vec![1., 2., 3., 4.]);
}

parameterized_test! {can_detect_invalid_due_date, (due, is_invalid), {
    let job = Job { due: due.map(str::to_string), ..create_delivery_job("job1", (1., 0.)) };
    let problem = Problem { plan: Plan { jobs: vec![job], ..create_empty_plan() }, ..create_empty_problem() };

    let result = check_e1108_invalid_due_date(&ValidationContext::new(&problem, None, &CoordIndex::new(&problem)));

    assert_eq!(result.err().map(|err| err.code), is_invalid.then(|| "E1108".to_string()));
}}

can_detect_invalid_due_date! {
    case01_absent: (None, false),
    case02_rfc3339: (Some("2024-07-04T17:00:00+10:00"), false),
    // a due date needs a time: "any time that day" is sent as the end of the day
    case03_date_only: (Some("2024-07-04"), true),
    case04_out_of_range: (Some("2024-13-04T17:00:00Z"), true),
    case05_not_a_date: (Some("tomorrow"), true),
}
