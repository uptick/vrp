#[cfg(test)]
#[path = "../../tests/unit/validation/routing_test.rs"]
mod routing_test;

use super::*;
use crate::utils::combine_error_results;
use std::collections::HashSet;
use vrp_core::prelude::Float;

/// Checks that no duplicated profile names specified.
fn check_e1500_duplicated_profiles(ctx: &ValidationContext) -> Result<(), FormatError> {
    get_duplicates(ctx.problem.fleet.profiles.iter().map(|p| &p.name)).map_or(Ok(()), |names| {
        Err(FormatError::new(
            "E1500".to_string(),
            "duplicated profile names".to_string(),
            format!("remove duplicates of profiles with the names: '{}'", names.join(", ")),
        ))
    })
}

/// Checks that profiles collection is not empty.
fn check_e1501_empty_profiles(ctx: &ValidationContext) -> Result<(), FormatError> {
    if ctx.problem.fleet.profiles.is_empty() {
        Err(FormatError::new(
            "E1501".to_string(),
            "empty profile collection".to_string(),
            "specify at least one profile".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Checks that only one type of location is used.
fn check_e1502_no_location_type_mix(_ctx: &ValidationContext, location_types: (bool, bool)) -> Result<(), FormatError> {
    let (has_coordinates, has_indices) = location_types;

    if has_coordinates && has_indices {
        Err(FormatError::new(
            "E1502".to_string(),
            "mixing different location types".to_string(),
            "use either coordinates or indices for all locations".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Checks that routing matrix is supplied when location indices are used.
fn check_e1503_no_matrix_when_indices_used(
    ctx: &ValidationContext,
    location_types: (bool, bool),
) -> Result<(), FormatError> {
    let (_, has_indices) = location_types;

    if has_indices && ctx.matrices.is_none_or(|matrices| matrices.is_empty()) {
        Err(FormatError::new(
            "E1503".to_string(),
            "location indices requires routing matrix to be specified".to_string(),
            "either use coordinates everywhere or specify routing matrix".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Checks that coord index has a proper maximum index for
fn check_e1504_index_size_mismatch(ctx: &ValidationContext) -> Result<(), FormatError> {
    let max_index = ctx.coord_index.max_matrix_index();

    let (matrix_size, is_correct_index) = ctx
        .matrices
        .and_then(|matrices| matrices.first())
        .map(|matrix| (matrix.distances.len() as Float).sqrt().round() as usize)
        .map_or((0_usize, true), |matrix_size| (matrix_size, max_index + 1 == matrix_size));

    if !is_correct_index {
        Err(FormatError::new(
            "E1504".to_string(),
            "amount of locations does not match matrix dimension".to_string(),
            format!(
                "check matrix size: max location index '{max_index}' + 1 should be equal to matrix size ('{matrix_size}')"
            ),
        ))
    } else {
        Ok(())
    }
}

/// Checks that no duplicated profile names specified.
fn check_e1505_profiles_exist(ctx: &ValidationContext) -> Result<(), FormatError> {
    let known_matrix_profiles = ctx.problem.fleet.profiles.iter().map(|p| p.name.clone()).collect::<HashSet<_>>();

    let unknown_vehicle_profiles = ctx
        .problem
        .fleet
        .vehicles
        .iter()
        .map(|vehicle| vehicle.profile.matrix.clone())
        .chain(ctx.problem.plan.clustering.iter().map(|clustering| match clustering {
            Clustering::Vicinity { profile, .. } => profile.matrix.clone(),
        }))
        .filter(|matrix| !known_matrix_profiles.contains(matrix))
        .collect::<HashSet<_>>();

    if unknown_vehicle_profiles.is_empty() {
        Ok(())
    } else {
        let unknown_profiles = unknown_vehicle_profiles.into_iter().collect::<Vec<_>>();
        Err(FormatError::new(
            "E1505".to_string(),
            "unknown matrix profile name in vehicle or vicinity clustering profile".to_string(),
            format!("ensure that matrix profiles '{}' are defined in profiles", unknown_profiles.join(", ")),
        ))
    }
}

/// Checks that profile approximation tables are well formed.
fn check_e1506_approximation_tables(ctx: &ValidationContext) -> Result<(), FormatError> {
    let is_valid_table = |table: &[[Float; 2]]| {
        !table.is_empty()
            && table.iter().all(|[x, y]| x.is_finite() && y.is_finite() && *x > 0. && *y > 0.)
            && table.windows(2).all(|pair| pair[0][0] < pair[1][0] && pair[0][1] < pair[1][1])
    };

    let invalid_profiles = ctx
        .problem
        .fleet
        .profiles
        .iter()
        .filter(|profile| {
            profile.approximation.as_ref().is_some_and(|approximation| {
                profile.speed.is_some()
                    || !is_valid_table(&approximation.durations)
                    || approximation.distances.as_ref().is_some_and(|distances| !is_valid_table(distances))
            })
        })
        .map(|profile| profile.name.clone())
        .collect::<Vec<_>>();

    if invalid_profiles.is_empty() {
        Ok(())
    } else {
        Err(FormatError::new(
            "E1506".to_string(),
            "invalid approximation in matrix profile".to_string(),
            format!(
                "for profiles '{}': remove speed when approximation is set, and give each table at least one \
                 point, with positive values strictly increasing in both straight-line meters and value",
                invalid_profiles.join(", ")
            ),
        ))
    }
}

/// Checks that a profile's speed is a finite number above zero, and its unknown location travel
/// a finite duration and distance of zero or more.
fn check_e1507_invalid_profile_travel(ctx: &ValidationContext) -> Result<(), FormatError> {
    let is_positive = |value: Option<Float>| value.is_none_or(|value| value.is_finite() && value > 0.);
    let is_non_negative = |value: Float| value.is_finite() && value >= 0.;

    let invalid_profiles = ctx
        .problem
        .fleet
        .profiles
        .iter()
        .filter(|profile| {
            !is_positive(profile.speed)
                || profile
                    .unknown_location_travel
                    .as_ref()
                    .is_some_and(|travel| !is_non_negative(travel.duration) || !is_non_negative(travel.distance))
        })
        .map(|profile| profile.name.as_str())
        .collect::<Vec<_>>();

    if invalid_profiles.is_empty() {
        Ok(())
    } else {
        Err(FormatError::new(
            "E1507".to_string(),
            "invalid speed or unknown location travel in profile".to_string(),
            format!(
                "give profiles '{}' a speed above zero, and unknown location travel of zero or more",
                invalid_profiles.join(", ")
            ),
        ))
    }
}

/// Validates routing rules.
pub fn validate_routing(ctx: &ValidationContext) -> Result<(), MultiFormatError> {
    let location_types = (ctx.coord_index.has_coordinates(), ctx.coord_index.has_indices());

    combine_error_results(&[
        check_e1500_duplicated_profiles(ctx),
        check_e1501_empty_profiles(ctx),
        check_e1502_no_location_type_mix(ctx, location_types),
        check_e1503_no_matrix_when_indices_used(ctx, location_types),
        check_e1504_index_size_mismatch(ctx),
        check_e1505_profiles_exist(ctx),
        check_e1506_approximation_tables(ctx),
        check_e1507_invalid_profile_travel(ctx),
    ])
    .map_err(From::from)
}
