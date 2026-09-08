use crate::format::problem::*;
use crate::format::solution::*;
use crate::format_time;
use crate::helpers::*;
use vrp_core::prelude::Float;

fn create_vehicle_type_with_max_duration_limit(max_duration: Float) -> VehicleType {
    VehicleType {
        limits: Some(VehicleLimits {
            max_distance: None,
            max_duration: Some(max_duration),
            tour_size: None,
            allow_out_of_hours_depot_travel: None,
        }),
        ..create_default_vehicle_type()
    }
}

#[test]
fn can_limit_one_job_by_max_duration() {
    let problem = Problem {
        plan: Plan { jobs: vec![create_delivery_job("job1", (100., 0.))], ..create_empty_plan() },
        fleet: Fleet { vehicles: vec![create_vehicle_type_with_max_duration_limit(99.)], ..create_default_fleet() },
        ..create_empty_problem()
    };
    let matrix = Matrix {
        profile: Some("car".to_owned()),
        timestamp: None,
        travel_times: vec![1, 100, 100, 1],
        distances: vec![1, 1, 1, 1],
        error_codes: None,
    };

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(solution.unassigned.iter().len(), 1);
}

#[test]
fn can_skip_job_from_multiple_because_of_max_duration() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![
                create_delivery_job_with_duration("job1", (1., 0.), 10.),
                create_delivery_job_with_duration("job2", (2., 0.), 10.),
                create_delivery_job_with_duration("job3", (3., 0.), 10.),
                create_delivery_job_with_duration("job4", (4., 0.), 10.),
                create_delivery_job_with_duration("job5", (5., 0.), 10.),
            ],
            ..create_empty_plan()
        },
        fleet: Fleet { vehicles: vec![create_vehicle_type_with_max_duration_limit(40.)], ..create_default_fleet() },
        ..create_empty_problem()
    };
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
                            .load(vec![3])
                            .build_departure(),
                        StopBuilder::default()
                            .coordinate((3., 0.))
                            .schedule_stamp(3., 13.)
                            .load(vec![2])
                            .distance(3)
                            .build_single("job3", "delivery"),
                        StopBuilder::default()
                            .coordinate((2., 0.))
                            .schedule_stamp(14., 24.)
                            .load(vec![1])
                            .distance(4)
                            .build_single("job2", "delivery"),
                        StopBuilder::default()
                            .coordinate((1., 0.))
                            .schedule_stamp(25., 35.)
                            .load(vec![0])
                            .distance(5)
                            .build_single("job1", "delivery"),
                        StopBuilder::default()
                            .coordinate((0., 0.))
                            .schedule_stamp(36., 36.)
                            .load(vec![0])
                            .distance(6)
                            .build_arrival(),
                    ])
                    .statistic(StatisticBuilder::default().driving(6).serving(30).build())
                    .build()
            )
            .unassigned(Some(vec![
                UnassignedJob {
                    job_id: "job4".to_string(),
                    reasons: vec![UnassignedJobReason {
                        code: "MAX_DURATION_CONSTRAINT".to_string(),
                        description: "cannot be assigned due to max duration constraint of vehicle".to_string(),
                        details: Some(vec![UnassignedJobDetail {
                            vehicle_id: "my_vehicle_1".to_string(),
                            shift_index: 0
                        }]),
                    }]
                },
                UnassignedJob {
                    job_id: "job5".to_string(),
                    reasons: vec![UnassignedJobReason {
                        code: "MAX_DURATION_CONSTRAINT".to_string(),
                        description: "cannot be assigned due to max duration constraint of vehicle".to_string(),
                        details: Some(vec![UnassignedJobDetail {
                            vehicle_id: "my_vehicle_1".to_string(),
                            shift_index: 0
                        }]),
                    }]
                }
            ]))
            .build()
    );
}

#[test]
fn allow_out_of_hours_depot_travel_moves_shift_bounds_onto_first_and_last_jobs() {
    // Depot at (0,0), first job at (10,0) → 10 units of travel.
    // Shift bounds are [100, 200]. With the flag, the vehicle should be able to leave the
    // depot before t=100 so it arrives at the first job at t>=100. The last job departure
    // must be <= 200, but the depot return may happen later.
    let problem = Problem {
        plan: Plan { jobs: vec![create_delivery_job_with_duration("job1", (10., 0.), 5.)], ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![VehicleShift {
                    start: ShiftStart { earliest: format_time(100.), latest: None, location: (0., 0.).to_loc() },
                    end: Some(ShiftEnd { earliest: None, latest: format_time(200.), location: (0., 0.).to_loc() }),
                    breaks: None,
                    reloads: None,
                    recharges: None,
                }],
                limits: Some(VehicleLimits {
                    max_distance: None,
                    max_duration: None,
                    tour_size: None,
                    allow_out_of_hours_depot_travel: Some(true),
                }),
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(solution.unassigned.is_none() || solution.unassigned.unwrap().is_empty());
    assert_eq!(solution.tours.len(), 1);

    let tour = &solution.tours[0];
    let depot_departure = crate::parse_time(tour.stops[0].schedule().departure.as_str());
    let first_job_arrival = crate::parse_time(tour.stops[1].schedule().arrival.as_str());

    // The depot was left before the shift started (at t<100) so that the first job is
    // reached at the shift start. Without the flag, depot departure would be pinned at >=100.
    assert!(depot_departure < 100., "expected depot to depart before shift start (t<100), got t={depot_departure}");
    assert!(
        first_job_arrival >= 100.,
        "expected first-job arrival at/after shift start (t>=100), got t={first_job_arrival}"
    );
}

#[test]
fn allow_out_of_hours_depot_travel_still_enforces_shift_start_latest() {
    // Depot at (0,0), job at (200,0) → 200 units of travel.
    // Shift start window [100, 150]. Even leaving the depot at t=0, first-job arrival
    // would be t=200, which exceeds start.latest=150. The flag relaxes the depot-start
    // window but the travel-limit constraint must still reject this insertion.
    let problem = Problem {
        plan: Plan { jobs: vec![create_delivery_job_with_duration("job1", (200., 0.), 5.)], ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![VehicleShift {
                    start: ShiftStart {
                        earliest: format_time(100.),
                        latest: Some(format_time(150.)),
                        location: (0., 0.).to_loc(),
                    },
                    end: Some(ShiftEnd { earliest: None, latest: format_time(1000.), location: (0., 0.).to_loc() }),
                    breaks: None,
                    reloads: None,
                    recharges: None,
                }],
                limits: Some(VehicleLimits {
                    max_distance: None,
                    max_duration: None,
                    tour_size: None,
                    allow_out_of_hours_depot_travel: Some(true),
                }),
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert_eq!(solution.unassigned.iter().flatten().count(), 1);
    assert!(solution.tours.is_empty(), "expected no tours since the only job cannot be assigned");
}

fn create_single_long_job_problem(allow_out_of_hours_depot_travel: bool) -> Problem {
    // Depot at (0,0), job at (10,0) → 10 units of travel each way.
    // Shift window is [100, 200] (span 100). The job's service duration is 150, which
    // exceeds the entire shift span on its own — it cannot fit between the shift start
    // and shift end regardless of when the depot is left. This mirrors "a 12h task on a
    // 9-5 (8h) shift": it must never be assigned, with OR without the flag.
    Problem {
        plan: Plan { jobs: vec![create_delivery_job_with_duration("job1", (10., 0.), 150.)], ..create_empty_plan() },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![VehicleShift {
                    start: ShiftStart { earliest: format_time(100.), latest: None, location: (0., 0.).to_loc() },
                    end: Some(ShiftEnd { earliest: None, latest: format_time(200.), location: (0., 0.).to_loc() }),
                    breaks: None,
                    reloads: None,
                    recharges: None,
                }],
                limits: Some(VehicleLimits {
                    max_distance: None,
                    max_duration: None,
                    tour_size: None,
                    allow_out_of_hours_depot_travel: Some(allow_out_of_hours_depot_travel),
                }),
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    }
}

// Regression test for the `allow_out_of_hours_depot_travel` over-long-job bug.
//
// A job whose service duration exceeds the whole shift span cannot be served regardless of the
// flag: out-of-hours travel only lets the depot legs spill outside the shift, the work itself
// must still fit inside the shift window. Previously the flag relaxed the depot-start window to
// `{earliest: None, latest: None}` (→ `Actor.detail.time.start = 0`), so the transport
// time-window constraint evaluated insertions against a `[0, end.latest]` operating window and
// scheduled over-long jobs anyway. The fix floors the first-job arrival to the
// `FirstJobArrivalFloor` inside the feasibility check, so the shift-end overrun is detected.
#[test]
fn allow_out_of_hours_depot_travel_rejects_job_longer_than_shift() {
    // The over-long (150-unit) job cannot fit in the 100-unit shift with the flag OFF or ON.
    for allow_out_of_hours_depot_travel in [false, true] {
        let problem = create_single_long_job_problem(allow_out_of_hours_depot_travel);
        let matrix = create_matrix_from_problem(&problem);
        let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

        assert_eq!(
            solution.unassigned.iter().flatten().count(),
            1,
            "over-long job must be unassigned (allow_out_of_hours_depot_travel={allow_out_of_hours_depot_travel})"
        );
        assert!(
            solution.tours.is_empty(),
            "no tour should be produced (allow_out_of_hours_depot_travel={allow_out_of_hours_depot_travel})"
        );
    }
}

#[test]
fn can_serve_job_when_it_starts_late() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![create_delivery_job_with_times("job1", (1., 0.), vec![(100, 200)], 10.)],
            ..create_empty_plan()
        },
        fleet: Fleet { vehicles: vec![create_vehicle_type_with_max_duration_limit(50.)], ..create_default_fleet() },
        ..create_empty_problem()
    };
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(solution.unassigned.is_none());
    assert!(!solution.tours.is_empty());
}
