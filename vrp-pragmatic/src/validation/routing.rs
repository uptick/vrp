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
    let has_location_table = ctx.problem.locations.is_some();

    if has_indices && !has_location_table && ctx.matrices.is_none_or(|matrices| matrices.is_empty()) {
        Err(FormatError::new(
            "E1503".to_string(),
            "location indices requires routing matrix to be specified".to_string(),
            "either use coordinates everywhere, specify routing matrix or specify a location table".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Checks that coord index has a proper maximum index for
fn check_e1504_index_size_mismatch(ctx: &ValidationContext) -> Result<(), FormatError> {
    let max_index = ctx.coord_index.max_matrix_index();

    // A profile with no routing matrix is approximated from the location table, at its size.
    let (matrix_size, is_correct_index) = ctx
        .matrices
        .and_then(|matrices| matrices.first())
        .map(get_matrix_size)
        .or_else(|| ctx.problem.locations.as_ref().map(|table| table.len()))
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

/// Checks that a location table is used with location indices only: it gives the coordinates of
/// each index, so a coordinate location would have no place in it.
fn check_e1506_location_table_with_coordinates(
    ctx: &ValidationContext,
    location_types: (bool, bool),
) -> Result<(), FormatError> {
    let (has_coordinates, _) = location_types;

    if ctx.problem.locations.is_some() && has_coordinates {
        Err(FormatError::new(
            "E1506".to_string(),
            "location table used with coordinate locations".to_string(),
            "refer to every location by its index in the location table".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Checks that every routing matrix has one row per location in the location table.
fn check_e1507_matrix_size_differs_from_location_table(ctx: &ValidationContext) -> Result<(), FormatError> {
    let Some(table_size) = ctx.problem.locations.as_ref().map(|table| table.len()) else {
        return Ok(());
    };

    let mismatched = ctx
        .matrices
        .iter()
        .flat_map(|matrices| matrices.iter())
        .map(get_matrix_size)
        .filter(|&matrix_size| matrix_size != table_size)
        .collect::<HashSet<_>>();

    if mismatched.is_empty() {
        Ok(())
    } else {
        let mut sizes = mismatched.into_iter().collect::<Vec<_>>();
        sizes.sort_unstable();
        let sizes = sizes.iter().map(|size| size.to_string()).collect::<Vec<_>>().join(", ");
        Err(FormatError::new(
            "E1507".to_string(),
            "routing matrix size does not match location table".to_string(),
            format!("give every matrix as many locations as the location table ('{table_size}'), not '{sizes}'"),
        ))
    }
}

/// Checks that every routing matrix names its profile when there is a location table: a profile
/// with no matrix is approximated, so matrices cannot be paired with profiles by position.
fn check_e1508_unnamed_matrix_with_location_table(ctx: &ValidationContext) -> Result<(), FormatError> {
    let has_unnamed_matrix = ctx.matrices.is_some_and(|matrices| matrices.iter().any(|m| m.profile.is_none()));

    if ctx.problem.locations.is_some() && has_unnamed_matrix {
        Err(FormatError::new(
            "E1508".to_string(),
            "routing matrix without a profile alongside a location table".to_string(),
            "set the profile of every routing matrix".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn get_matrix_size(matrix: &Matrix) -> usize {
    (matrix.distances.len() as Float).sqrt().round() as usize
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
        check_e1506_location_table_with_coordinates(ctx, location_types),
        check_e1507_matrix_size_differs_from_location_table(ctx),
        check_e1508_unnamed_matrix_with_location_table(ctx),
    ])
    .map_err(From::from)
}
