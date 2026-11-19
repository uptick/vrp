use super::*;
use crate::format::problem::*;
use crate::helpers::*;

#[test]
fn can_return_profile_unknown_location_travel_for_any_leg_with_unknown_end() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![
                create_delivery_job("job1", (1., 0.)),
                Job {
                    deliveries: Some(vec![JobTask {
                        places: vec![JobPlace {
                            times: None,
                            location: ApiLocation::Custom { r#type: CustomLocationType::Unknown },
                            duration: 1.,
                            tag: None,
                        }],
                        demand: Some(vec![1]),
                        order: None,
                    }]),
                    ..create_job("job2")
                },
            ],
            ..create_empty_plan()
        },
        ..create_empty_problem()
    };
    let coord_index = Arc::new(CoordIndex::new(&problem));
    let known = coord_index.get_by_loc(&ApiLocation::Coordinate { lat: 1., lng: 0. }).unwrap();
    let unknown = coord_index.get_by_loc(&ApiLocation::Custom { r#type: CustomLocationType::Unknown }).unwrap();
    let fallback =
        UnknownLocationFallback::with_unknown_location_travel(coord_index, vec![(0., 0.), (86400., 1000000.)]);
    let profile = |index| Profile::new(index, None);

    for (from, to) in [(known, unknown), (unknown, known), (unknown, unknown)] {
        assert_eq!(fallback.duration(&profile(1), from, to), 86400.);
        assert_eq!(fallback.distance(&profile(1), from, to), 1000000.);
        assert_eq!(fallback.duration(&profile(0), from, to), 0.);
        // A profile with no unknown location travel given gets zero.
        assert_eq!(fallback.distance(&profile(2), from, to), 0.);
    }
}
