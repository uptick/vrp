use crate::format::problem::Objective::*;
use crate::format::problem::*;
use crate::helpers::*;

#[test]
fn can_round_trip_minimize_depot_travel_time_objective() {
    let json = r#"{"type":"minimize-depot-travel-time"}"#;

    let objective: Objective = serde_json::from_str(json).expect("cannot deserialize objective");
    assert!(matches!(objective, MinimizeDepotTravelTime { cap: None }));

    let serialized = serde_json::to_string(&objective).expect("cannot serialize objective");
    assert_eq!(serialized, json);
}

#[test]
fn can_round_trip_minimize_depot_travel_time_objective_with_cap() {
    let json = r#"{"type":"minimize-depot-travel-time","cap":1800.0}"#;

    let objective: Objective = serde_json::from_str(json).expect("cannot deserialize objective");
    assert!(matches!(objective, MinimizeDepotTravelTime { cap: Some(c) } if c == 1800.0));

    let serialized = serde_json::to_string(&objective).expect("cannot serialize objective");
    assert_eq!(serialized, json);
}

#[test]
fn can_build_goal_with_minimize_depot_travel_time_objective() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![create_delivery_job("job1", (1., 0.)), create_delivery_job("job2", (2., 0.))],
            ..create_empty_plan()
        },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                shifts: vec![create_default_open_vehicle_shift()],
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        objectives: Some(vec![
            MinimizeUnassigned { breaks: None },
            MinimizeDepotTravelTime { cap: Some(1800.0) },
            MinimizeCost,
        ]),
        ..create_empty_problem()
    };
    let matrix = create_matrix_from_problem(&problem);

    let core_problem = (problem, vec![matrix]).read_pragmatic();

    assert!(core_problem.is_ok(), "failed to build goal: {:?}", core_problem.err());
}
