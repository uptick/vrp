#[cfg(test)]
#[path = "../../tests/unit/utils/approx_transportation_test.rs"]
mod approx_transportation_test;

use crate::format::{CustomLocationType, Location};
use vrp_core::models::common::Distance;
use vrp_core::utils::{Float, ParallelismPolicy, parallel_collect};

/// Gets approximated durations and distances rounded to nearest integer, one pair for each
/// `(speed, distance_multiplier)`. A leg's distance is its straight line times the multiplier, and
/// its duration that distance over the speed.
pub fn get_approx_transportation(locations: &[Location], settings: &[(Float, Float)]) -> Vec<(Vec<i64>, Vec<i64>)> {
    approximate(locations, settings, |value| value as i64)
}

/// The same as [`get_approx_transportation`], with the rounded values held as floats, the way
/// routing costs hold them, so they need no conversion and no second copy.
pub fn get_approx_travel(locations: &[Location], settings: &[(Float, Float)]) -> Vec<(Vec<Float>, Vec<Float>)> {
    approximate(locations, settings, |value| value)
}

fn approximate<T: Send>(
    locations: &[Location],
    settings: &[(Float, Float)],
    to_value: fn(Float) -> T,
) -> Vec<(Vec<T>, Vec<T>)> {
    assert!(!settings.is_empty());
    assert!(settings.iter().all(|&(speed, distance_multiplier)| speed > 0. && distance_multiplier > 0.));

    let distances = locations
        .iter()
        .flat_map(|l1| locations.iter().map(move |l2| get_haversine_distance(l1, l2)))
        .collect::<Vec<_>>();

    parallel_collect(settings, ParallelismPolicy::Default, |&(speed, distance_multiplier)| {
        let distances = distances.iter().map(|distance| distance * distance_multiplier);

        distances.map(|distance| (to_value((distance / speed).round()), to_value(distance.round()))).unzip()
    })
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
