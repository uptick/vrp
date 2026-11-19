//! Applies a routing profile's travel adjustments to the durations of its legs.

#[cfg(test)]
#[path = "../../tests/unit/format/travel_adjustments_test.rs"]
mod travel_adjustments_test;

use crate::format::problem::{Problem, TravelAdjustments};
use crate::format::{CoordIndex, CustomLocationType, Location};
use std::collections::HashMap;
use vrp_core::prelude::Float;

/// A profile's travel adjustments, with each arrival rule's location resolved to its matrix index.
pub(crate) struct ProfileAdjustments {
    minimum: Option<Float>,
    arrivals: HashMap<usize, Vec<ArrivalRule>>,
}

#[derive(Clone, Copy)]
enum ArrivalRule {
    Buffer(Float),
    RoundUpTo(Float),
}

impl ProfileAdjustments {
    /// Resolves `adjustments` against the problem's locations. A rule whose location is not in the
    /// problem, or is an unknown location, is left out: validation rejects both.
    pub(crate) fn new(adjustments: &TravelAdjustments, coord_index: &CoordIndex) -> Self {
        let mut arrivals: HashMap<usize, Vec<ArrivalRule>> = HashMap::new();

        adjustments.arrivals.iter().flatten().for_each(|arrival| {
            if matches!(arrival.location, Location::Custom { r#type: CustomLocationType::Unknown }) {
                return;
            }

            let rule = match (arrival.buffer, arrival.round_up_to) {
                (Some(buffer), None) => ArrivalRule::Buffer(buffer),
                (None, Some(step)) => ArrivalRule::RoundUpTo(step),
                _ => return,
            };

            if let Some(index) = coord_index.get_by_loc(&arrival.location) {
                arrivals.entry(index).or_default().push(rule);
            }
        });

        Self { minimum: adjustments.minimum, arrivals }
    }

    /// The adjusted duration of the leg from `from` to `to`: the minimum first, then the rules for
    /// arrivals at `to` in the order given. A location's leg to itself, and a leg marked unreachable
    /// with a negative duration, are left as they are.
    pub(crate) fn apply(&self, from: usize, to: usize, duration: Float) -> Float {
        self.apply_with_rules(from, to, duration, self.arrivals.get(&to).map(Vec::as_slice))
    }

    /// Adjusts every leg of a row-major `size` x `size` matrix of durations, in place.
    pub(crate) fn apply_to_matrix(&self, durations: &mut [Float], size: usize) {
        // Look each location's arrival rules up once, not once per leg.
        let rules_by_column = (0..size).map(|to| self.arrivals.get(&to).map(Vec::as_slice)).collect::<Vec<_>>();

        durations.iter_mut().enumerate().for_each(|(cell, duration)| {
            let (from, to) = (cell / size, cell % size);
            *duration = self.apply_with_rules(from, to, *duration, rules_by_column[to]);
        });
    }

    fn apply_with_rules(&self, from: usize, to: usize, duration: Float, rules: Option<&[ArrivalRule]>) -> Float {
        if from == to || duration < 0. {
            return duration;
        }

        let duration = self.minimum.map_or(duration, |minimum| duration.max(minimum));

        rules.into_iter().flatten().fold(duration, |duration, rule| match *rule {
            ArrivalRule::Buffer(buffer) => duration + buffer,
            ArrivalRule::RoundUpTo(step) => (duration / step).ceil() * step,
        })
    }
}

/// Each profile's travel adjustments by profile index: `None` for a profile without any.
pub(crate) fn create_profile_adjustments(
    problem: &Problem,
    coord_index: &CoordIndex,
    profile_indices: &HashMap<String, usize>,
) -> Vec<Option<ProfileAdjustments>> {
    let mut adjustments = (0..profile_indices.len()).map(|_| None).collect::<Vec<_>>();

    problem.fleet.profiles.iter().rev().for_each(|profile| {
        if let (Some(&index), Some(profile_adjustments)) = (profile_indices.get(&profile.name), &profile.adjustments) {
            adjustments[index] = Some(ProfileAdjustments::new(profile_adjustments, coord_index));
        }
    });

    adjustments
}
