use crate::format::problem::PragmaticProblem;
use crate::format::problem::*;
use crate::format::solution::*;
use crate::format_time;
use crate::helpers::*;
use crate::parse_time;

fn create_crew_job(id: &str, location: (f64, f64), crew: usize, duration: f64) -> Job {
    let mut job = create_service_job(id, location);
    job.services.as_mut().unwrap()[0].places[0].duration = duration;
    Job { crew: Some(crew), ..job }
}

fn create_vehicle_at(id: &str, location: (f64, f64), shift: (f64, f64)) -> VehicleType {
    VehicleType {
        shifts: vec![VehicleShift {
            start: ShiftStart { earliest: format_time(shift.0), latest: None, location: location.to_loc() },
            end: Some(ShiftEnd { earliest: None, latest: format_time(shift.1), location: location.to_loc() }),
            ..create_default_vehicle_shift()
        }],
        ..create_default_vehicle(id)
    }
}

fn create_crew_problem(jobs: Vec<Job>, vehicles: Vec<VehicleType>) -> Problem {
    Problem {
        plan: Plan { jobs, ..create_empty_plan() },
        fleet: Fleet { vehicles, ..create_default_fleet() },
        ..create_empty_problem()
    }
}

/// Returns (vehicle id, start time) for every tour serving the job.
fn get_crew_members(solution: &Solution, job_id: &str) -> Vec<(String, f64)> {
    let mut members = Vec::new();
    for tour in solution.tours.iter() {
        for stop in tour.stops.iter() {
            for activity in stop.activities().iter().filter(|activity| activity.job_id == job_id) {
                let start = activity.time.as_ref().map_or(stop.schedule().arrival.as_str(), |time| &time.start);
                members.push((tour.vehicle_id.clone(), parse_time(start)));
            }
        }
    }
    members.sort_by(|(a, _), (b, _)| a.cmp(b));

    members
}

fn get_unassigned_codes(solution: &Solution, job_id: &str) -> Option<Vec<String>> {
    solution.unassigned.as_ref().and_then(|unassigned| {
        unassigned
            .iter()
            .find(|job| job.job_id == job_id)
            .map(|job| job.reasons.iter().map(|reason| reason.code.clone()).collect())
    })
}

#[test]
fn can_serve_crew_job_by_two_vehicles_at_the_same_time() {
    let problem = create_crew_problem(
        vec![
            create_crew_job("crew", (5., 0.), 2, 10.),
            create_service_job("job1", (1., 0.)),
            create_service_job("job2", (9., 0.)),
        ],
        vec![create_vehicle_at("v1", (0., 0.), (0., 1000.)), create_vehicle_at("v2", (10., 0.), (0., 1000.))],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    let members = get_crew_members(&solution, "crew");
    assert_eq!(members.len(), 2);
    assert_ne!(members[0].0, members[1].0);
    assert_eq!(members[0].1, members[1].1);
    assert!(solution.unassigned.is_none());
}

#[test]
fn can_wait_for_the_crew_member_who_arrives_last() {
    let problem = create_crew_problem(
        vec![create_crew_job("crew", (2., 0.), 2, 10.)],
        vec![create_vehicle_at("v1", (0., 0.), (0., 1000.)), create_vehicle_at("v2", (20., 0.), (0., 1000.))],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    // v1 arrives at 2 and waits for v2, who arrives at 18
    assert_eq!(get_crew_members(&solution, "crew"), vec![("v1_1".to_string(), 18.), ("v2_1".to_string(), 18.)]);
}

#[test]
fn can_serve_crew_of_three() {
    let problem = create_crew_problem(
        vec![create_crew_job("crew", (5., 5.), 3, 10.), create_service_job("job1", (1., 1.))],
        vec![
            create_vehicle_at("v1", (0., 0.), (0., 1000.)),
            create_vehicle_at("v2", (10., 0.), (0., 1000.)),
            create_vehicle_at("v3", (0., 10.), (0., 1000.)),
        ],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    let members = get_crew_members(&solution, "crew");
    assert_eq!(members.iter().map(|(vehicle, _)| vehicle.as_str()).collect::<Vec<_>>(), vec!["v1_1", "v2_1", "v3_1"]);
    assert!(members.iter().all(|(_, start)| *start == members[0].1));
}

#[test]
fn can_unassign_crew_job_when_only_one_vehicle_can_serve_it() {
    let crew_job =
        Job { skills: Some(all_of_skills(vec!["licence".to_string()])), ..create_crew_job("crew", (5., 0.), 2, 10.) };
    let skilled_vehicle =
        VehicleType { skills: Some(vec!["licence".to_string()]), ..create_vehicle_at("v1", (0., 0.), (0., 1000.)) };
    let problem = create_crew_problem(
        vec![crew_job, create_service_job("job1", (1., 0.))],
        vec![skilled_vehicle, create_vehicle_at("v2", (10., 0.), (0., 1000.))],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(get_crew_members(&solution, "crew").is_empty());
    assert!(get_unassigned_codes(&solution, "crew").unwrap().contains(&"CREW_CONSTRAINT".to_string()));
    assert_eq!(get_crew_members(&solution, "job1").len(), 1);
}

#[test]
fn can_reject_crew_larger_than_fleet() {
    let problem = create_crew_problem(
        vec![create_crew_job("crew", (5., 0.), 2, 10.)],
        vec![create_vehicle_at("v1", (0., 0.), (0., 1000.))],
    );

    let result = problem.read_pragmatic();

    let codes = result.err().map(|errors| errors.into_iter().map(|err| err.code).collect::<Vec<_>>());
    assert_eq!(codes, Some(vec!["E1109".to_string()]));
}

#[test]
fn can_unassign_crew_job_when_vehicles_cannot_be_there_together() {
    let problem = create_crew_problem(
        vec![create_crew_job("crew", (5., 0.), 2, 10.), create_service_job("job1", (1., 0.))],
        vec![create_vehicle_at("v1", (0., 0.), (0., 100.)), create_vehicle_at("v2", (10., 0.), (200., 300.))],
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(get_crew_members(&solution, "crew").is_empty());
    assert!(get_unassigned_codes(&solution, "crew").is_some());
    assert_eq!(get_crew_members(&solution, "job1").len(), 1);
}

#[test]
fn can_solve_many_crew_jobs_with_regular_ones() {
    let crew_jobs = (0..6).map(|idx| create_crew_job(&format!("crew{idx}"), (idx as f64 * 3., 5.), 2, 5.));
    let jobs = (0..20).map(|idx| create_service_job(&format!("job{idx}"), ((idx % 10) as f64 * 2., (idx / 10) as f64)));
    let problem = create_crew_problem(
        crew_jobs.chain(jobs).collect(),
        vec![
            create_vehicle_at("v1", (0., 0.), (0., 200.)),
            create_vehicle_at("v2", (10., 0.), (0., 200.)),
            create_vehicle_at("v3", (20., 0.), (0., 200.)),
        ],
    );
    let matrix = create_matrix_from_problem(&problem);

    // the checker run by the helper validates crews: different vehicles, same start, all or nothing
    let solution = solve_with_metaheuristic_and_iterations(problem, Some(vec![matrix]), 500);

    (0..6).for_each(|idx| {
        let members = get_crew_members(&solution, &format!("crew{idx}"));
        assert!(members.is_empty() || members.len() == 2);
    });
    assert!(solution.unassigned.is_none(), "unassigned: {:?}", solution.unassigned);
}
