#[cfg(test)]
#[path = "../../tests/unit/format/location_fallback_test.rs"]
mod location_fallback_test;

use crate::format::{CoordIndex, CustomLocationType, Location as ApiLocation};
use std::sync::Arc;
use vrp_core::models::common::{Distance, Duration, Location, Profile};
use vrp_core::models::problem::TransportFallback;
use vrp_core::prelude::Float;

/// A transport fallback for only a custom unknown location type.
/// Returns, for a leg to, from or between unknown locations, the profile's unknown location travel:
/// zero distance and duration unless set.
pub struct UnknownLocationFallback {
    coord_index: Arc<CoordIndex>,
    // (duration, distance) by profile index.
    unknown_location_travel: Vec<(Duration, Distance)>,
}

impl UnknownLocationFallback {
    /// Creates a new instance of [`UnknownLocationFallback`] which returns zero for every profile.
    pub fn new(coord_index: Arc<CoordIndex>) -> Self {
        Self::with_unknown_location_travel(coord_index, vec![])
    }

    /// Creates a new instance of [`UnknownLocationFallback`] with the travel of a leg with an
    /// unknown end, as `(duration, distance)` by profile index. A profile without one gets zero.
    pub fn with_unknown_location_travel(
        coord_index: Arc<CoordIndex>,
        unknown_location_travel: Vec<(Duration, Distance)>,
    ) -> Self {
        Self { coord_index, unknown_location_travel }
    }

    fn get_default_value(&self, profile: &Profile, from: Location, to: Location) -> (Duration, Distance) {
        let (from, to) = (self.coord_index.get_by_idx(from), self.coord_index.get_by_idx(to));

        match (from, to) {
            (Some(ApiLocation::Custom { r#type: CustomLocationType::Unknown }), _)
            | (_, Some(ApiLocation::Custom { r#type: CustomLocationType::Unknown })) => {
                self.unknown_location_travel.get(profile.index).copied().unwrap_or((Float::default(), Float::default()))
            }
            _ => panic!("fallback is only for locations of custom unknown type"),
        }
    }
}

impl TransportFallback for UnknownLocationFallback {
    fn duration(&self, profile: &Profile, from: Location, to: Location) -> Duration {
        self.get_default_value(profile, from, to).0
    }

    fn distance(&self, profile: &Profile, from: Location, to: Location) -> Distance {
        self.get_default_value(profile, from, to).1
    }
}
