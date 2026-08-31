use crate::format::Location;
use crate::format::problem::*;
use crate::format_time;
use crate::helpers::*;

fn create_shift_start() -> ShiftStart {
    ShiftStart { earliest: format_time(0.), latest: Some(format_time(0.)), location: (0., 0.).to_loc() }
}

fn create_problem(jobs: Vec<Job>, vehicle_break: VehicleBreak, is_open: bool) -> Problem {
    let vehicle_shift = if is_open { create_default_open_vehicle_shift() } else { create_default_vehicle_shift() };
    Problem {
        plan: Plan { jobs, ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                costs: create_default_vehicle_costs(),
                shifts: vec![VehicleShift {
                    start: create_shift_start(),
                    breaks: Some(vec![vehicle_break]),
                    ..vehicle_shift
                }],
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    }
}

#[test]
fn can_assign_break_during_travel() {
    let is_open = false;
    let problem = create_problem(
        vec![create_delivery_job("job1", (5., 0.)), create_delivery_job("job2", (10., 0.))],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(7.), latest: format_time(7.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(
        solution,
        SolutionBuilder::default()
            .tour(
                TourBuilder::default()
                    .stops(vec![
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(0., 0.)
                            .load(vec![2])
                            .build_departure(),
                        StopBuilder::default()
                            .coordinate((5., 0.))
                            .schedule_stamp(5., 6.)
                            .load(vec![1])
                            .distance(5)
                            .build_single("job1", "delivery"),
                        StopBuilder::new_transit().schedule_stamp(7., 9.).load(vec![1]).build_single("break", "break"),
                        StopBuilder::default()
                            .coordinate((10., 0.))
                            .schedule_stamp(13., 14.)
                            .load(vec![0])
                            .distance(10)
                            .build_single("job2", "delivery"),
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(24., 24.)
                            .load(vec![0])
                            .distance(20)
                            .build_arrival(),
                    ])
                    .statistic(StatisticBuilder::default().driving(20).serving(2).break_time(2).build())
                    .build()
            )
            .build()
    );
}

#[test]
fn can_defer_break_until_service_is_finished() {
    let is_open = false;
    let problem = create_problem(
        vec![create_delivery_job_with_duration("job1", (5., 0.), 3.)],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(7.), latest: format_time(10.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    // NOTE the break becomes due at 7, while the job is served: it waits until the service is over
    assert_eq!(
        solution,
        SolutionBuilder::default()
            .tour(
                TourBuilder::default()
                    .stops(vec![
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(0., 0.)
                            .load(vec![1])
                            .build_departure(),
                        StopBuilder::default()
                            .coordinate((5., 0.))
                            .schedule_stamp(5., 10.)
                            .load(vec![0])
                            .distance(5)
                            .activity(
                                ActivityBuilder::delivery()
                                    .job_id("job1")
                                    .coordinate((5., 0.))
                                    .time_stamp(5., 8.)
                                    .build()
                            )
                            .activity(ActivityBuilder::break_type().time_stamp(8., 10.).build())
                            .build(),
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(15., 15.)
                            .load(vec![0])
                            .distance(10)
                            .build_arrival(),
                    ])
                    .statistic(StatisticBuilder::default().driving(10).serving(3).break_time(2).build())
                    .build()
            )
            .build()
    );
}

#[test]
fn can_take_break_before_service_when_it_is_due_on_arrival() {
    let is_open = false;
    let problem = create_problem(
        vec![create_delivery_job_with_duration("job1", (5., 0.), 3.)],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(5.), latest: format_time(7.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    // NOTE the break becomes due when the vehicle arrives, so it is taken before the work starts
    assert_eq!(
        solution,
        SolutionBuilder::default()
            .tour(
                TourBuilder::default()
                    .stops(vec![
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(0., 0.)
                            .load(vec![1])
                            .build_departure(),
                        StopBuilder::default()
                            .coordinate((5., 0.))
                            .schedule_stamp(5., 10.)
                            .load(vec![0])
                            .distance(5)
                            .activity(ActivityBuilder::break_type().time_stamp(5., 7.).build())
                            .activity(
                                ActivityBuilder::delivery()
                                    .job_id("job1")
                                    .coordinate((5., 0.))
                                    .time_stamp(7., 10.)
                                    .build()
                            )
                            .build(),
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(15., 15.)
                            .load(vec![0])
                            .distance(10)
                            .build_arrival(),
                    ])
                    .statistic(StatisticBuilder::default().driving(10).serving(3).break_time(2).build())
                    .build()
            )
            .build()
    );
}

#[test]
fn can_take_break_during_waiting_time() {
    let is_open = false;
    let problem = create_problem(
        vec![create_delivery_job_with_times("job1", (5., 0.), vec![(10, 100)], 2.)],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(5.), latest: format_time(8.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    // NOTE the vehicle arrives at 5 and waits till 10 for the job's time window, so the break is taken
    //      while it is idle and costs nothing: the waiting time is reduced by it instead
    assert_eq!(
        solution,
        SolutionBuilder::default()
            .tour(
                TourBuilder::default()
                    .stops(vec![
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(0., 0.)
                            .load(vec![1])
                            .build_departure(),
                        StopBuilder::default()
                            .coordinate((5., 0.))
                            .schedule_stamp(5., 12.)
                            .load(vec![0])
                            .distance(5)
                            .activity(ActivityBuilder::break_type().time_stamp(5., 7.).build())
                            .activity(
                                ActivityBuilder::delivery()
                                    .job_id("job1")
                                    .coordinate((5., 0.))
                                    .time_stamp(10., 12.)
                                    .build()
                            )
                            .build(),
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(17., 17.)
                            .load(vec![0])
                            .distance(10)
                            .build_arrival(),
                    ])
                    .statistic(StatisticBuilder::default().driving(10).serving(2).waiting(3).break_time(2).build())
                    .build()
            )
            .build()
    );
}

#[test]
fn can_reject_job_which_cannot_be_served_around_break() {
    let is_open = false;
    let problem = create_problem(
        vec![create_delivery_job_with_duration("job1", (5., 0.), 3.)],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(7.), latest: format_time(7.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    // NOTE the job is served from 5 till 8, so the break cannot be taken at 7 without interrupting it
    assert!(solution.tours.is_empty());
    assert_eq!(solution.unassigned.as_ref().map(|unassigned| unassigned.len()), Some(1));
    assert_eq!(
        solution.unassigned.as_ref().and_then(|unassigned| unassigned.first()).map(|job| job.job_id.as_str()),
        Some("job1")
    );
}

#[test]
fn can_handle_required_break_when_its_start_falls_at_activity_end() {
    let is_open = true;
    let problem = create_problem(
        vec![create_delivery_job("job1", (5., 0.)), create_delivery_job("job2", (10., 0.))],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(6.), latest: format_time(6.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(
        solution,
        SolutionBuilder::default()
            .tour(
                TourBuilder::default()
                    .stops(vec![
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(0., 0.)
                            .load(vec![2])
                            .build_departure(),
                        StopBuilder::default()
                            .coordinate((5., 0.))
                            .schedule_stamp(5., 6.)
                            .load(vec![1])
                            .distance(5)
                            .build_single("job1", "delivery"),
                        StopBuilder::new_transit().schedule_stamp(6., 8.).load(vec![1]).build_single("break", "break"),
                        StopBuilder::default()
                            .coordinate((10., 0.))
                            .schedule_stamp(13., 14.)
                            .load(vec![0])
                            .distance(10)
                            .build_single("job2", "delivery"),
                    ])
                    .statistic(StatisticBuilder::default().driving(10).serving(2).break_time(2).build())
                    .build()
            )
            .build()
    );
}

#[test]
fn can_skip_break_if_it_becomes_due_after_tour_end() {
    let is_open = true;
    let problem = create_problem(
        vec![create_delivery_job("job1", (5., 0.))],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(20.), latest: format_time(22.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(get_ids_from_tour(&solution.tours[0]).iter().flatten().all(|id| id != "break"));
}

#[test]
fn can_take_break_on_the_road_when_it_is_due_while_driving() {
    let is_open = true;
    let problem = create_problem(
        vec![create_delivery_job("job1", (5., 0.)), create_delivery_job("job2", (10., 0.))],
        VehicleBreak::Required {
            time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(4.), latest: format_time(7.) },
            duration: 2.,
        },
        is_open,
    );
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    // NOTE the break becomes due at 4, when the vehicle is still on its way to the first job
    assert_eq!(
        solution,
        SolutionBuilder::default()
            .tour(
                TourBuilder::default()
                    .stops(vec![
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(0., 0.)
                            .load(vec![2])
                            .build_departure(),
                        StopBuilder::new_transit().schedule_stamp(4., 6.).load(vec![2]).build_single("break", "break"),
                        StopBuilder::default()
                            .coordinate((5., 0.))
                            .schedule_stamp(7., 8.)
                            .load(vec![1])
                            .distance(5)
                            .build_single("job1", "delivery"),
                        StopBuilder::default()
                            .coordinate((10., 0.))
                            .schedule_stamp(13., 14.)
                            .load(vec![0])
                            .distance(10)
                            .build_single("job2", "delivery"),
                    ])
                    .statistic(StatisticBuilder::default().driving(10).serving(2).break_time(2).build())
                    .build()
            )
            .build()
    );
}

#[test]
fn can_handle_required_break_with_infeasible_sequence_relation() {
    let create_test_job = |index: usize, duration: f64, times: (String, String)| Job {
        services: Some(vec![JobTask {
            places: vec![JobPlace {
                location: Location::Reference { index },
                duration,
                times: Some(vec![vec![times.0, times.1]]),
                tag: None,
            }],
            demand: None,
            order: None,
        }]),
        ..create_job(index.to_string().as_str())
    };

    let problem = Problem {
        plan: Plan {
            jobs: vec![
                create_test_job(0, 10800., (format_time(0.), format_time(86399.))),
                create_test_job(1, 3600., (format_time(81000.), format_time(81000.))),
                create_test_job(2, 1800., (format_time(86400. + 900.), format_time(86400. + 900.))),
                create_test_job(3, 5400., (format_time(75600.), format_time(75600.))),
                create_test_job(4, 1800., (format_time(86400. + 2700.), format_time(86400. + 2700.))),
            ],
            relations: Some(vec![Relation {
                type_field: RelationType::Sequence,
                jobs: to_strings(vec!["3", "1", "2", "4"]),
                vehicle_id: "my_vehicle_1".to_string(),
                shift_index: Some(0),
            }]),
            ..create_empty_plan()
        },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![VehicleShift {
                    start: ShiftStart {
                        earliest: format_time(86400. + 28800.),
                        latest: Some(format_time(86400. + 28800.)),
                        location: Location::Reference { index: 5 },
                    },
                    end: Some(ShiftEnd {
                        earliest: None,
                        latest: format_time(86400. + 57600.),
                        location: Location::Reference { index: 5 },
                    }),
                    breaks: Some(vec![VehicleBreak::Required {
                        time: VehicleRequiredBreakTime::OffsetTime { earliest: 15303., latest: 15303. },
                        duration: 1800.,
                    }]),
                    ..create_default_vehicle_shift()
                }],
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };

    let matrix = Matrix {
        profile: Some("car".to_string()),
        timestamp: None,
        travel_times: vec![
            0, 635, 24, 580, 27, 2232, 625, 0, 650, 76, 653, 2507, 24, 660, 0, 605, 3, 2257, 570, 95, 595, 0, 598,
            2449, 27, 663, 3, 608, 0, 2260, 2232, 2545, 2257, 2515, 2260, 0,
        ],
        distances: vec![
            0, 8888, 192, 8510, 215, 52931, 8896, 0, 9088, 450, 9111, 56579, 192, 9080, 0, 8702, 23, 53123, 8518, 450,
            8710, 0, 8733, 60163, 215, 9103, 23, 8725, 0, 53146, 52996, 56684, 53188, 60477, 53211, 0,
        ],
        error_codes: None,
    };

    let solution = solve_with_metaheuristic_and_iterations_without_check(problem, Some(vec![matrix]), 200);

    // With proper constraint handling via ControlFlow, job "0" (with very wide time window) cannot be
    // scheduled due to the required break at specific time conflicting with the strict sequence relation.
    assert!(!solution.tours.is_empty());
    assert_eq!(solution.unassigned.as_ref().map(|u| u.len()), Some(1));
    assert_eq!(solution.unassigned.as_ref().and_then(|u| u.first()).map(|j| j.job_id.as_str()), Some("0"));
}
