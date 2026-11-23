use super::*;
use crate::helpers::*;

#[test]
fn can_read_vicinity_thresholds() {
    let problem = Problem {
        plan: Plan {
            clustering: Some(Clustering::Vicinity {
                profile: VehicleProfile { matrix: "car".to_string(), scale: None },
                threshold: VicinityThresholdPolicy {
                    duration: 300.,
                    distance: 400.,
                    min_shared_time: None,
                    smallest_time_window: None,
                    max_jobs_per_cluster: None,
                },
                visiting: VicinityVisitPolicy::Continue,
                serving: VicinityServingPolicy::Original { parking: 0. },
                filtering: None,
            }),
            ..create_empty_plan()
        },
        fleet: Fleet { vehicles: vec![create_default_vehicle_type()], ..create_default_fleet() },
        ..create_empty_problem()
    };

    let config = create_cluster_config(&problem).expect("cannot create config").expect("no config");

    assert_eq!(config.threshold.moving_duration, 300.);
    assert_eq!(config.threshold.moving_distance, 400.);
}
