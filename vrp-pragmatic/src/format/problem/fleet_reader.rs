#[cfg(test)]
#[path = "../../../tests/unit/format/problem/fleet_reader_test.rs"]
mod fleet_reader_test;

use super::*;
use crate::Location as ApiLocation;
use crate::format::{UnknownLocationFallback, create_profile_adjustments};
use crate::utils::{get_approx_transportation, get_approx_travel};
use std::collections::HashSet;
use vrp_core::construction::enablers::{FirstJobArrivalFloorDimension, create_typed_actor_groups};
use vrp_core::construction::features::{VehicleCapacityDimension, VehicleSkillsDimension};
use vrp_core::models::common::*;
use vrp_core::models::problem::*;

pub(super) fn get_profile_index_map(api_problem: &ApiProblem) -> HashMap<String, usize> {
    api_problem.fleet.profiles.iter().fold(Default::default(), |mut acc, profile| {
        if !acc.contains_key(&profile.name) {
            acc.insert(profile.name.clone(), acc.len());
        }
        acc
    })
}

/// Creates the routing costs from the supplied `matrices` or, when `approximate` is set, from
/// travel approximated from the problem's coordinates.
pub(super) fn create_transport_costs(
    api_problem: &ApiProblem,
    matrices: &[Matrix],
    coord_index: Arc<CoordIndex>,
    approximate: bool,
) -> GenericResult<Arc<dyn TransportCost>> {
    let matrix_profiles = get_profile_index_map(api_problem);
    let mut matrix_data = if approximate {
        create_approx_matrix_data(api_problem, coord_index.as_ref(), &matrix_profiles)
    } else {
        create_supplied_matrix_data(matrices, &matrix_profiles)?
    };

    // Adjust each profile's legs after its travel is known, whether supplied or approximated.
    let adjustments = create_profile_adjustments(api_problem, coord_index.as_ref(), &matrix_profiles);
    matrix_data.iter_mut().for_each(|data| {
        if let Some(Some(profile_adjustments)) = adjustments.get(data.index) {
            let size = (data.durations.len() as Float).sqrt().round() as usize;
            profile_adjustments.apply_to_matrix(&mut data.durations, size);
        }
    });

    let matrix_indices = matrix_data.iter().map(|data| data.index).collect::<HashSet<_>>().len();
    if matrix_profiles.len() != matrix_indices {
        return Err("amount of fleet profiles does not match matrix profiles".into());
    }

    if coord_index.has_custom() {
        let unknown_location_travel = get_profile_unknown_location_travel(api_problem, &matrix_profiles);
        create_matrix_transport_cost_with_fallback(
            matrix_data,
            UnknownLocationFallback::with_unknown_location_travel(coord_index, unknown_location_travel),
        )
    } else {
        create_matrix_transport_cost(matrix_data)
    }
}

fn create_supplied_matrix_data(
    matrices: &[Matrix],
    matrix_profiles: &HashMap<String, usize>,
) -> GenericResult<Vec<MatrixData>> {
    if !matrices.iter().all(|m| m.profile.is_some()) && !matrices.iter().all(|m| m.profile.is_none()) {
        return Err("all matrices should have profile set or none of them".into());
    }

    if matrices.iter().any(|m| m.profile.is_none()) && matrices.iter().any(|m| m.timestamp.is_some()) {
        return Err("when timestamp is set, all matrices should have profile set".into());
    }

    if matrix_profiles.len() > matrices.len() {
        return Err(format!(
            "not enough routing matrices specified for fleet profiles defined: \
             {} must be less or equal to {}",
            matrix_profiles.len(),
            matrices.len()
        )
        .into());
    }

    matrices
        .iter()
        .enumerate()
        .map(|(idx, matrix)| {
            let profile = matrix.profile.as_ref().and_then(|p| matrix_profiles.get(p)).cloned().unwrap_or(idx);
            (profile, matrix.timestamp.clone(), matrix)
        })
        .map(|(profile, timestamp, matrix)| {
            let (durations, distances) = if let Some(error_codes) = &matrix.error_codes {
                let capacity = matrix.distances.len();

                let mut durations: Vec<Duration> = Vec::with_capacity(capacity);
                let mut distances: Vec<Distance> = Vec::with_capacity(capacity);
                let err_fn = |i| move || GenericError::from(format!("invalid matrix index: {i}"));

                for (i, error) in error_codes.iter().enumerate() {
                    if *error > 0 {
                        durations.push(-1.);
                        distances.push(-1.);
                    } else {
                        durations.push(*matrix.travel_times.get(i).ok_or_else(err_fn(i))? as Float);
                        distances.push(*matrix.distances.get(i).ok_or_else(err_fn(i))? as Float);
                    }
                }

                (durations, distances)
            } else {
                (
                    matrix.travel_times.iter().map(|d| *d as Float).collect(),
                    matrix.distances.iter().map(|d| *d as Float).collect(),
                )
            };

            Ok(MatrixData::new(profile, timestamp.map(|t| parse_time(&t)), durations, distances))
        })
        .collect::<Result<Vec<_>, GenericError>>()
}

/// The travel of a leg with an unknown end, as `(duration, distance)` by profile index.
fn get_profile_unknown_location_travel(
    api_problem: &ApiProblem,
    profile_indices: &HashMap<String, usize>,
) -> Vec<(Float, Float)> {
    let mut unknown_location_travel = vec![(0., 0.); profile_indices.len()];
    api_problem.fleet.profiles.iter().rev().for_each(|profile| {
        if let (Some(&index), Some(travel)) =
            (profile_indices.get(&profile.name), profile.unknown_location_travel.as_ref())
        {
            unknown_location_travel[index] = (travel.duration, travel.distance);
        }
    });

    unknown_location_travel
}

pub(super) fn read_fleet(
    api_problem: &ApiProblem,
    props: &ProblemProperties,
    coord_index: &CoordIndex,
) -> Result<CoreFleet, MultiFormatError> {
    let profile_indices = get_profile_index_map(api_problem);
    let mut vehicles: Vec<Arc<Vehicle>> = Default::default();

    api_problem.fleet.vehicles.iter().for_each(|vehicle| {
        let costs = Costs {
            fixed: vehicle.costs.fixed.unwrap_or(0.),
            per_distance: vehicle.costs.distance,
            per_driving_time: vehicle.costs.time,
            per_waiting_time: vehicle.costs.time,
            per_service_time: vehicle.costs.time,
        };

        let index = *profile_indices.get(&vehicle.profile.matrix).unwrap();
        let profile = Profile::new(index, vehicle.profile.scale);

        let tour_size = vehicle.limits.as_ref().and_then(|limits| limits.tour_size);

        let allow_out_of_hours_depot_travel =
            vehicle.limits.as_ref().and_then(|limits| limits.allow_out_of_hours_depot_travel).unwrap_or(false);

        for (shift_index, shift) in vehicle.shifts.iter().enumerate() {
            let shift_start_earliest = parse_time(&shift.start.earliest);
            let shift_start_latest = shift.start.latest.as_ref().map(|time| parse_time(time));

            // When `allow_out_of_hours_depot_travel` is set, the shift start applies to the first
            // job's arrival rather than to the depot departure. Relax the depot start window;
            // the floor is re-applied in `apply_first_job_arrival_floor` (via `FirstJobArrivalFloor`
            // dim) and `start.latest` is enforced in the travel-limit constraint. The end side
            // is NOT relaxed — the back-propagated `latest_arrival` bounds the last-job departure.
            let start_time = if allow_out_of_hours_depot_travel {
                TimeInterval { earliest: None, latest: None }
            } else {
                TimeInterval { earliest: Some(shift_start_earliest), latest: shift_start_latest }
            };
            let start_location = coord_index.get_by_loc(&shift.start.location).unwrap();

            let end = shift.end.as_ref().map(|end| {
                let location = coord_index.get_by_loc(&end.location).unwrap();
                let time = TimeInterval { earliest: None, latest: Some(parse_time(&end.latest)) };
                VehiclePlace { location, time }
            });

            let details =
                vec![VehicleDetail { start: Some(VehiclePlace { location: start_location, time: start_time }), end }];

            vehicle.vehicle_ids.iter().for_each(|vehicle_id| {
                let mut dimens: Dimensions = Default::default();

                dimens
                    .set_vehicle_type(vehicle.type_id.clone())
                    .set_shift_index(shift_index)
                    .set_vehicle_id(vehicle_id.to_string());

                if let Some(tour_size) = tour_size {
                    dimens.set_tour_size(tour_size);
                }

                if allow_out_of_hours_depot_travel {
                    dimens.set_first_job_arrival_floor(shift_start_earliest);
                    if let Some(latest) = shift_start_latest {
                        dimens.set_shift_start_latest(latest);
                    }
                }

                if props.has_multi_dimen_capacity {
                    dimens.set_vehicle_capacity(MultiDimLoad::new(vehicle.capacity.clone()));
                } else {
                    dimens.set_vehicle_capacity(SingleDimLoad::new(*vehicle.capacity.first().unwrap()));
                }

                if let Some(skills) = vehicle.skills.as_ref() {
                    dimens.set_vehicle_skills(skills.iter().cloned().collect::<HashSet<_>>());
                }

                vehicles.push(Arc::new(Vehicle {
                    profile: profile.clone(),
                    costs: costs.clone(),
                    dimens,
                    details: details.clone(),
                }));
            });
        }
    });

    let drivers = vec![Arc::new(Driver {
        costs: Costs {
            fixed: 0.0,
            per_distance: 0.0,
            per_driving_time: 0.0,
            per_waiting_time: 0.0,
            per_service_time: 0.0,
        },
        dimens: Default::default(),
        details: vec![],
    })];

    // NOTE: `CoreFleet::new` requires at least one vehicle. Reject an empty fleet here rather than
    // letting that invariant trip an assertion, so that a definition which parses but cannot be
    // solved is reported like any other format problem. Validation normally catches this first and
    // says which vehicle type is at fault; this is the backstop for anything it does not cover.
    if vehicles.is_empty() {
        return Err(vec![FormatError::new(
            "E1309".to_string(),
            "fleet has no vehicles".to_string(),
            "add a vehicle type with at least one vehicle id and one shift".to_string(),
        )]
        .into());
    }

    Ok(CoreFleet::new(drivers, vehicles, |actors| {
        create_typed_actor_groups(actors, |a| {
            a.vehicle.dimens.get_vehicle_type().cloned().expect("vehicle has no type defined")
        })
    }))
}

/// Creates a matrices using approximation.
pub fn create_approx_matrices(problem: &ApiProblem) -> Vec<Matrix> {
    create_approx_matrices_with_index(problem, &CoordIndex::new(problem))
}

const DEFAULT_SPEED: Float = 10.;
const DEFAULT_DISTANCE_MULTIPLIER: Float = 1.;

fn get_approx_settings(profile: &MatrixProfile) -> (Float, Float) {
    (profile.speed.unwrap_or(DEFAULT_SPEED), profile.distance_multiplier.unwrap_or(DEFAULT_DISTANCE_MULTIPLIER))
}

/// Each pair of speed and distance multiplier, once.
fn get_unique_approx_settings(settings: impl Iterator<Item = (Float, Float)>) -> Vec<(Float, Float)> {
    let mut unique: Vec<(Float, Float)> = Vec::new();
    settings.for_each(|pair| {
        if !unique.iter().any(|&known| known.0.to_bits() == pair.0.to_bits() && known.1.to_bits() == pair.1.to_bits()) {
            unique.push(pair);
        }
    });

    unique
}

/// The locations approximation works over: every location in the index except unknown ones, in
/// index order.
fn get_approx_locations(coord_index: &CoordIndex) -> Vec<ApiLocation> {
    coord_index.unique().into_iter().filter(|location| !matches!(location, ApiLocation::Custom { .. })).collect()
}

pub(super) fn create_approx_matrices_with_index(problem: &ApiProblem, coord_index: &CoordIndex) -> Vec<Matrix> {
    let settings = get_unique_approx_settings(problem.fleet.profiles.iter().map(get_approx_settings));
    let approx_data = get_approx_transportation(&get_approx_locations(coord_index), settings.as_slice());

    problem
        .fleet
        .profiles
        .iter()
        .map(move |profile| {
            let profile_settings = get_approx_settings(profile);
            let idx = settings.iter().position(|&s| s == profile_settings).expect("Cannot find profile settings");

            Matrix {
                profile: Some(profile.name.clone()),
                timestamp: None,
                travel_times: approx_data[idx].0.clone(),
                distances: approx_data[idx].1.clone(),
                error_codes: None,
            }
        })
        .collect()
}

/// Approximates each profile's travel straight into the routing costs' own form, with no integer
/// matrices in between. A profile index takes the settings of the first profile with its name, and
/// profiles which share settings share one approximation: the last of them takes it without a copy.
fn create_approx_matrix_data(
    problem: &ApiProblem,
    coord_index: &CoordIndex,
    profile_indices: &HashMap<String, usize>,
) -> Vec<MatrixData> {
    let mut settings_by_index = vec![(DEFAULT_SPEED, DEFAULT_DISTANCE_MULTIPLIER); profile_indices.len()];
    problem.fleet.profiles.iter().rev().for_each(|profile| {
        if let Some(&index) = profile_indices.get(&profile.name) {
            settings_by_index[index] = get_approx_settings(profile);
        }
    });

    let settings = get_unique_approx_settings(settings_by_index.iter().copied());
    let position = |pair: (Float, Float)| settings.iter().position(|&known| known == pair).expect("known settings");

    let mut remaining_uses = vec![0_usize; settings.len()];
    settings_by_index.iter().for_each(|&pair| remaining_uses[position(pair)] += 1);

    let mut travel = get_approx_travel(&get_approx_locations(coord_index), settings.as_slice())
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();

    settings_by_index
        .iter()
        .enumerate()
        .map(|(index, &pair)| {
            let position = position(pair);
            remaining_uses[position] -= 1;
            let (durations, distances) =
                if remaining_uses[position] == 0 { travel[position].take() } else { travel[position].clone() }
                    .expect("approximated travel");

            MatrixData::new(index, None, durations, distances)
        })
        .collect()
}
