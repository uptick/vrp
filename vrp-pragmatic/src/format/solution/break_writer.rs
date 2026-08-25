use super::*;
use vrp_core::construction::enablers::{
    ReservedTimePlacement, ReservedTimeWindow, ReservedTimesIndex, place_reserved_time,
};
use vrp_core::models::common::{Cost, Duration, TimeWindow};
use vrp_core::models::solution::Route;

/// Specifies how a reserved time is attached to the activity it is taken at.
enum BreakKind {
    /// Taken on the road, while driving to the activity.
    InTransit,
    /// Taken at the activity's stop, before its service starts. Keeps the time window of the service,
    /// which is delayed by the break, and its overlap with the originally reported waiting time.
    BeforeService { service_time: TimeWindow, waiting_overlap: Duration },
    /// Taken at the activity's stop, once its service is finished.
    AfterService,
}

/// Specifies where a reserved time is taken within a route.
struct BreakPlacement {
    /// An index of the activity the break is attached to.
    activity_idx: usize,
    /// A time window of the break itself.
    time: TimeWindow,
    /// Specifies how the break is attached to the activity.
    kind: BreakKind,
}

/// Converts reserved time duration applied to activity or travel time to break activity.
pub(super) fn insert_reserved_times_as_breaks(
    route: &Route,
    tour: &mut Tour,
    reserved_times_index: &ReservedTimesIndex,
) {
    // NOTE apply the placements from the last one, so that the activity indices of the earlier ones
    //      are not invalidated by the stops inserted for a break taken on the road
    let costs = &route.actor.vehicle.costs;
    let costs = (costs.per_service_time, costs.per_waiting_time);

    get_break_placements(route, reserved_times_index)
        .into_iter()
        .rev()
        .for_each(|placement| insert_break(tour, &placement, costs));
}

/// Finds where each reserved time of the route's actor is taken, in chronological order.
fn get_break_placements(route: &Route, reserved_times_index: &ReservedTimesIndex) -> Vec<BreakPlacement> {
    let offset = route.tour.start().map(|activity| activity.schedule.departure).unwrap_or(0.);

    reserved_times_index
        .get(&route.actor)
        .iter()
        .flat_map(|reserved_times| reserved_times.iter())
        .map(|reserved_time| reserved_time.to_reserved_time_window(offset))
        .filter_map(|reserved_time| get_break_placement(route, &reserved_time))
        .collect()
}

fn get_break_placement(route: &Route, reserved_time: &ReservedTimeWindow) -> Option<BreakPlacement> {
    let due = reserved_time.time.start;
    let occupied = TimeWindow::new(due, due + reserved_time.duration);

    // NOTE an activity owns the reserved time when it is in progress at the moment the break becomes
    //      due, which is checked against the schedule the activity would have without the break
    let at_activity = route.tour.all_activities().enumerate().find_map(|(activity_idx, activity)| {
        let (arrival, place) = (activity.schedule.arrival, &activity.place);
        let service_start = arrival.max(place.time.start);
        let service_end = service_start + place.duration;

        if !TimeWindow::new(arrival, service_end).intersects_exclusive(&occupied) {
            return None;
        }

        // NOTE when the break cannot be taken without interrupting the service, the route is not
        //      feasible and the solver reports it as such; still show the break after the service,
        //      which is where the schedule of the activity accounts for it
        let placement = place_reserved_time(arrival, &place.time, place.duration, reserved_time)
            .unwrap_or(ReservedTimePlacement::AfterService { start: service_end });

        let (start, kind) = match placement {
            ReservedTimePlacement::BeforeService { start, service_start } => {
                let waiting = TimeWindow::new(arrival, arrival.max(place.time.start));
                let break_time = TimeWindow::new(start, start + reserved_time.duration);
                let waiting_overlap = waiting.overlapping(&break_time).map(|tw| tw.duration()).unwrap_or(0.);
                let service_time = TimeWindow::new(service_start, service_start + place.duration);

                (start, BreakKind::BeforeService { service_time, waiting_overlap })
            }
            ReservedTimePlacement::AfterService { start } => (start, BreakKind::AfterService),
        };

        Some(BreakPlacement { activity_idx, time: TimeWindow::new(start, start + reserved_time.duration), kind })
    });

    at_activity.or_else(|| {
        // NOTE otherwise, the break becomes due while the vehicle is on the road
        route.tour.legs().find_map(|(leg, idx)| match leg {
            [from, to] => {
                let travel = TimeWindow::new(from.schedule.departure, to.schedule.arrival);

                (travel.start <= occupied.start && occupied.end <= travel.end).then(|| BreakPlacement {
                    activity_idx: idx + 1,
                    time: occupied.clone(),
                    kind: BreakKind::InTransit,
                })
            }
            _ => None,
        })
    })
}

/// Finds a stop holding the activity with a given index, along with the index within that stop.
fn find_stop_of_activity(tour: &Tour, activity_idx: usize) -> Option<(usize, usize)> {
    tour.stops
        .iter()
        .enumerate()
        .scan(0, |offset, (stop_idx, stop)| {
            let start = *offset;
            *offset += stop.activities().len();

            Some((stop_idx, start, *offset))
        })
        .find(|(_, start, end)| (*start..*end).contains(&activity_idx))
        .map(|(stop_idx, start, _)| (stop_idx, activity_idx - start))
}

fn insert_break(tour: &mut Tour, placement: &BreakPlacement, costs: (Cost, Cost)) {
    let Some((stop_idx, activity_idx)) = find_stop_of_activity(tour, placement.activity_idx) else { return };

    let break_time = placement.time.duration() as i64;
    tour.statistic.times.break_time += break_time;

    match &placement.kind {
        BreakKind::InTransit => {
            // NOTE the driving time of the leg includes the break, so account for it separately
            tour.statistic.times.driving -= break_time;

            let load = tour.stops.get(stop_idx.max(1) - 1).map(|stop| stop.load().clone()).unwrap_or_default();
            tour.stops.insert(
                stop_idx,
                Stop::Transit(TransitStop {
                    time: ApiSchedule {
                        arrival: format_time(placement.time.start),
                        departure: format_time(placement.time.end),
                    },
                    load,
                    activities: vec![create_break_activity(&placement.time)],
                }),
            );
        }
        BreakKind::BeforeService { service_time, waiting_overlap } => {
            // NOTE the break replaces a part of the time the vehicle would have spent waiting
            tour.statistic.times.waiting -= *waiting_overlap as i64;
            add_break_cost(tour, stop_idx, placement.time.duration(), *waiting_overlap, costs);

            let Some(stop) = tour.stops.get_mut(stop_idx) else { return };
            if let Some(activity) = stop.activities_mut().get_mut(activity_idx) {
                activity.time =
                    Some(Interval { start: format_time(service_time.start), end: format_time(service_time.end) });
            }
            stop.activities_mut().insert(activity_idx, create_break_activity(&placement.time));
        }
        BreakKind::AfterService => {
            add_break_cost(tour, stop_idx, placement.time.duration(), 0., costs);

            let Some(stop) = tour.stops.get_mut(stop_idx) else { return };
            stop.activities_mut().insert(activity_idx + 1, create_break_activity(&placement.time));
        }
    }
}

/// Adds a cost of a break taken at a point stop: the time spent there is not a service time, so it is
/// not a part of the cost calculated for the stop's activities. The part of the break which replaces
/// the waiting time is already accounted for as such, so its cost is given back.
fn add_break_cost(
    tour: &mut Tour,
    stop_idx: usize,
    break_time: Duration,
    waiting_overlap: Duration,
    costs: (Cost, Cost),
) {
    let (per_service_time, per_waiting_time) = costs;

    if let Some(Stop::Point(_)) = tour.stops.get(stop_idx) {
        tour.statistic.cost += break_time * per_service_time - waiting_overlap * per_waiting_time;
    }
}

fn create_break_activity(time: &TimeWindow) -> ApiActivity {
    ApiActivity {
        job_id: "break".to_string(),
        activity_type: "break".to_string(),
        location: None,
        time: Some(Interval { start: format_time(time.start), end: format_time(time.end) }),
        job_tag: None,
        commute: None,
    }
}
