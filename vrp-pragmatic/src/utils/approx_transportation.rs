#[cfg(test)]
#[path = "../../tests/unit/utils/approx_transportation_test.rs"]
mod approx_transportation_test;

use crate::format::{CustomLocationType, Location};
use vrp_core::models::common::Distance;
use vrp_core::utils::{Float, ParallelismPolicy, parallel_collect};

/// How travel is worked out from the straight-line distance between two locations.
#[derive(Clone, Debug, PartialEq)]
pub enum Approximation {
    /// The straight line covered at a speed (meters per second).
    Speed(Float),
    /// Tables of `[straight-line meters, value]` knots, read as `interpolate` does.
    Tables {
        /// Duration in seconds.
        durations: Vec<[Float; 2]>,
        /// Road distance in meters. The straight line itself when absent.
        distances: Option<Vec<[Float; 2]>>,
    },
}

impl Approximation {
    /// A key equal for two approximations exactly when they work travel out the same way.
    pub(crate) fn key(&self) -> Vec<u64> {
        match self {
            Approximation::Speed(speed) => vec![0, speed.to_bits()],
            Approximation::Tables { durations, distances } => {
                let mut key = vec![1];
                std::iter::once(durations).chain(distances).for_each(|table| {
                    key.push(table.len() as u64);
                    key.extend(table.iter().flat_map(|knot| knot.map(Float::to_bits)));
                });
                key
            }
        }
    }

    fn duration(&self, distance: Float) -> Float {
        match self {
            Approximation::Speed(speed) => distance / speed,
            Approximation::Tables { durations, .. } => interpolate(durations, distance),
        }
    }

    fn distance(&self, distance: Float) -> Float {
        match self {
            Approximation::Tables { distances: Some(distances), .. } => interpolate(distances, distance),
            _ => distance,
        }
    }
}

/// Gets approximated durations and distances rounded to nearest integer, one pair of
/// matrices per approximation.
pub fn get_approx_transportation(
    locations: &[Location],
    approximations: &[Approximation],
) -> Vec<(Vec<i64>, Vec<i64>)> {
    assert!(!approximations.is_empty());
    assert!(approximations.iter().all(|approximation| match approximation {
        Approximation::Speed(speed) => *speed > 0.,
        Approximation::Tables { .. } => true,
    }));

    let distances = locations
        .iter()
        .flat_map(|l1| locations.iter().map(move |l2| get_haversine_distance(l1, l2)))
        .collect::<Vec<_>>();

    let distances_rounded = distances.iter().map(|distance| distance.round() as i64).collect::<Vec<_>>();

    parallel_collect(approximations, ParallelismPolicy::Default, |approximation| {
        let durations =
            distances.iter().map(|&distance| approximation.duration(distance).round() as i64).collect::<Vec<_>>();

        let road_distances = match approximation {
            Approximation::Tables { distances: Some(_), .. } => {
                distances.iter().map(|&distance| approximation.distance(distance).round() as i64).collect()
            }
            _ => distances_rounded.clone(),
        };

        (durations, road_distances)
    })
}

/// Reads a value off a table of `[x, y]` knots: along the straight line between the knots either
/// side of `x`, from an implicit `[0, 0]`, and beyond the last knot along the last segment's slope.
/// Knots are expected strictly increasing in `x`; an empty table gives `x` back.
pub(crate) fn interpolate(knots: &[[Float; 2]], x: Float) -> Float {
    let segment = match knots.partition_point(|knot| knot[0] < x) {
        _ if knots.is_empty() => return x,
        0 => ([0., 0.], knots[0]),
        index if index < knots.len() => (knots[index - 1], knots[index]),
        _ if knots.len() == 1 => ([0., 0.], knots[0]),
        _ => (knots[knots.len() - 2], knots[knots.len() - 1]),
    };
    let ([x0, y0], [x1, y1]) = segment;

    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

/// Gets distance between two points using haversine formula.
pub(crate) fn get_haversine_distance(p1: &Location, p2: &Location) -> Float {
    if matches!(p1, Location::Custom { r#type: CustomLocationType::Unknown })
        || matches!(p2, Location::Custom { r#type: CustomLocationType::Unknown })
    {
        return Distance::default();
    }

    let (p1_lat, p1_lng) = as_lat_lon(p1.clone());
    let (p2_lat, p2_lng) = as_lat_lon(p2.clone());

    let d_lat = degree_rad(p1_lat - p2_lat);
    let d_lng = degree_rad(p1_lng - p2_lng);

    let lat1 = degree_rad(p1_lat);
    let lat2 = degree_rad(p2_lat);

    let a =
        (d_lat / 2.).sin() * (d_lat / 2.).sin() + (d_lng / 2.).sin() * (d_lng / 2.).sin() * (lat1).cos() * (lat2).cos();
    let c = 2. * a.sqrt().atan2((1. - a).sqrt());

    let radius = wgs84_earth_radius(d_lat);

    (radius * c) as Float
}

/// Converts degrees to radians.
#[inline(always)]
fn degree_rad(degrees: f64) -> f64 {
    std::f64::consts::PI * degrees / 180.
}

#[inline(always)]
fn wgs84_earth_radius(lat: f64) -> f64 {
    // semi-axes of WGS-84 geoidal reference
    const WGS84_A: f64 = 6_378_137.0; // major semiaxis [m]
    const WGS84_B: f64 = 6_356_752.3; // minor semiaxis [m]

    // http://en.wikipedia.org/wiki/Earth_radius
    let an = WGS84_A * WGS84_A * lat.cos();
    let bn = WGS84_B * WGS84_B * lat.sin();
    let ad = WGS84_A * lat.cos();
    let bd = WGS84_B * lat.sin();

    ((an * an + bn * bn) / (ad * ad + bd * bd)).sqrt()
}

fn as_lat_lon(location: Location) -> (f64, f64) {
    match location {
        Location::Coordinate { lat, lng } => (lat, lng),
        _ => panic!("approximation requires coordinates"),
    }
}
