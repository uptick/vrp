#[cfg(test)]
#[path = "../../../tests/unit/construction/enablers/reserved_time_test.rs"]
mod reserved_time_test;

use crate::models::common::*;
use crate::models::problem::{ActivityCost, Actor, TransportCost, TravelTime};
use crate::models::solution::{Activity, Route};
use rosomaxa::prelude::GenericError;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;

/// Represent a reserved time span entity.
#[derive(Clone, Debug)]
pub struct ReservedTimeSpan {
    /// A specific time span when an extra reserved duration should be applied.
    pub time: TimeSpan,
    /// An extra duration to be applied at given time.
    pub duration: Duration,
}

impl ReservedTimeSpan {
    /// Converts `ReservedTimeSpan` to `ReservedTimeWindow`.
    pub fn to_reserved_time_window(&self, offset: Timestamp) -> ReservedTimeWindow {
        ReservedTimeWindow { time: self.time.to_time_window(offset), duration: self.duration }
    }
}

/// Represent a reserved time window entity.
#[derive(Clone, Debug)]
pub struct ReservedTimeWindow {
    /// A specific time window when an extra reserved duration should be applied.
    pub time: TimeWindow,
    /// An extra duration to be applied at given time.
    pub duration: Duration,
}

/// Specifies where a reserved time is taken relative to an activity's service.
///
/// A reserved time never interrupts the service: it is taken either before the work starts or once
/// it is finished.
#[derive(Clone, Debug, PartialEq)]
pub enum ReservedTimePlacement {
    /// The reserved time is taken before the service starts, delaying the work.
    BeforeService {
        /// A time when the reserved time starts.
        start: Timestamp,
        /// A time when the service starts, once the reserved time is over.
        service_start: Timestamp,
    },
    /// The reserved time is taken once the service is finished, delaying the departure.
    AfterService {
        /// A time when the reserved time starts, which is the moment the service ends.
        start: Timestamp,
    },
}

impl ReservedTimePlacement {
    /// Returns a time when the reserved time starts.
    pub fn start(&self) -> Timestamp {
        match self {
            ReservedTimePlacement::BeforeService { start, .. } | ReservedTimePlacement::AfterService { start } => {
                *start
            }
        }
    }
}

/// Places a reserved time around an activity's service, so that the work is never interrupted by it.
///
/// The reserved time is taken as soon as it is due: while the vehicle is still idle at the activity,
/// if the work has not started yet, or right after the work is finished otherwise. Returns `None`
/// when it cannot be taken within its time window without interrupting the service, which makes the
/// activity infeasible for the actor.
pub fn place_reserved_time(
    arrival: Timestamp,
    service_time: &TimeWindow,
    service_duration: Duration,
    reserved_time: &ReservedTimeWindow,
) -> Option<ReservedTimePlacement> {
    let service_start = arrival.max(service_time.start);
    let (due, deadline) = (reserved_time.time.start, reserved_time.time.end);

    if due <= service_start {
        // the work has not started yet: take the reserved time now, delaying the service if needed
        let start = arrival.max(due);
        let service_start = service_start.max(start + reserved_time.duration);

        // NOTE: do not allow to start work after the activity's time window is over
        if start > deadline || service_start > service_time.end {
            None
        } else {
            Some(ReservedTimePlacement::BeforeService { start, service_start })
        }
    } else {
        // the work is already in progress: defer the reserved time until it is finished
        let start = service_start + service_duration;

        if start > deadline { None } else { Some(ReservedTimePlacement::AfterService { start }) }
    }
}

/// Specifies reserved time index type.
pub type ReservedTimesIndex = HashMap<Arc<Actor>, Vec<ReservedTimeSpan>>;

/// Specifies a function which returns an extra reserved time window for given actor. This reserved
/// time should be considered for planning.
pub(crate) type ReservedTimesFn = Arc<dyn Fn(&Route, &TimeWindow) -> Option<ReservedTimeWindow> + Send + Sync>;

/// Provides way to calculate activity costs which might contain reserved time.
pub struct DynamicActivityCost {
    reserved_times_fn: ReservedTimesFn,
}

impl DynamicActivityCost {
    /// Creates a new instance of `DynamicActivityCost` with given reserved time function.
    pub fn new(reserved_times_index: ReservedTimesIndex) -> Result<Self, GenericError> {
        Ok(Self { reserved_times_fn: create_reserved_times_fn(reserved_times_index)? })
    }
}

impl ActivityCost for DynamicActivityCost {
    fn estimate_departure(
        &self,
        route: &Route,
        activity: &Activity,
        arrival: Timestamp,
    ) -> ControlFlow<Timestamp, Timestamp> {
        let service_start = arrival.max(activity.place.time.start);
        let departure = service_start + activity.place.duration;
        let schedule = TimeWindow::new(arrival, departure);

        (self.reserved_times_fn)(route, &schedule).map_or(ControlFlow::Continue(departure), |reserved_time| {
            let place = &activity.place;

            match place_reserved_time(arrival, &place.time, place.duration, &reserved_time) {
                Some(ReservedTimePlacement::BeforeService { service_start, .. }) => {
                    ControlFlow::Continue(service_start + place.duration)
                }
                Some(ReservedTimePlacement::AfterService { start }) => {
                    ControlFlow::Continue(start + reserved_time.duration)
                }
                // NOTE: the reserved time cannot be taken within its time window without interrupting
                //       the service, so the activity is not feasible here. Still report a schedule with
                //       the same total duration for the callers which ignore the violation.
                // TODO this branch is the reason why departure rescheduling is disabled.
                //      theoretically, rescheduling should be aware somehow about dynamic costs
                None => ControlFlow::Break(departure + reserved_time.duration),
            }
        })
    }

    fn estimate_arrival(
        &self,
        route: &Route,
        activity: &Activity,
        departure: Timestamp,
    ) -> ControlFlow<Timestamp, Timestamp> {
        let arrival = activity.place.time.end.min(departure - activity.place.duration);
        let schedule = TimeWindow::new(arrival, departure);

        let value = (self.reserved_times_fn)(route, &schedule)
            .map_or(arrival, |reserved_time| (arrival - reserved_time.duration).max(activity.place.time.start));

        ControlFlow::Continue(value)
    }
}

/// Provides way to calculate transport costs which might contain reserved time.
pub struct DynamicTransportCost {
    reserved_times_fn: ReservedTimesFn,
    inner: Arc<dyn TransportCost>,
}

impl DynamicTransportCost {
    /// Creates a new instance of `DynamicTransportCost`.
    pub fn new(reserved_times_index: ReservedTimesIndex, inner: Arc<dyn TransportCost>) -> Result<Self, GenericError> {
        Ok(Self { reserved_times_fn: create_reserved_times_fn(reserved_times_index)?, inner })
    }
}

impl TransportCost for DynamicTransportCost {
    fn duration_approx(&self, profile: &Profile, from: Location, to: Location) -> Duration {
        self.inner.duration_approx(profile, from, to)
    }

    fn distance_approx(&self, profile: &Profile, from: Location, to: Location) -> Distance {
        self.inner.distance_approx(profile, from, to)
    }

    fn duration(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Duration {
        let duration = self.inner.duration(route, from, to, travel_time);

        let time_window = match travel_time {
            TravelTime::Arrival(arrival) => TimeWindow::new(arrival - duration, arrival),
            TravelTime::Departure(departure) => TimeWindow::new(departure, departure + duration),
        };

        (self.reserved_times_fn)(route, &time_window)
            .map_or(duration, |reserved_time| duration + reserved_time.duration)
    }

    fn distance(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Distance {
        self.inner.distance(route, from, to, travel_time)
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

/// Optimizes reserved time schedules by rescheduling it to earlier time (e.g. to avoid transit stops,
/// reduce waiting time).
pub(crate) fn optimize_reserved_times_schedule(route: &mut Route, reserved_times_fn: &ReservedTimesFn) {
    // NOTE run in this order as reducing waiting time can be also applied on top of avoiding travel time
    avoid_reserved_time_when_driving(route, reserved_times_fn);
    reduce_waiting_by_reserved_time(route, reserved_times_fn);
}

fn avoid_reserved_time_when_driving(route: &mut Route, reserved_times_fn: &ReservedTimesFn) {
    // NOTE assume reserved times has no intersection
    let schedule_shifts = route
        .tour
        .legs()
        .filter_map(|(leg, idx)| match &leg {
            &[from, to] => Some((from, to, idx)),
            _ => None,
        })
        .filter_map(|(from, to, idx)| {
            let travel_tw = TimeWindow::new(from.schedule.departure, to.schedule.arrival);
            reserved_times_fn(route, &travel_tw).map(|reserved_time| (idx, from, reserved_time))
        })
        .filter(|(_, from, reserved_time)| from.schedule.departure > reserved_time.time.start)
        .map(|(idx, _, reserved_time)| (idx, reserved_time.duration))
        .collect::<Vec<_>>();

    schedule_shifts.into_iter().for_each(|(idx, duration)| {
        route.tour.get_mut(idx).unwrap().schedule.departure += duration;
    });
}

fn reduce_waiting_by_reserved_time(_route: &mut Route, _reserved_times_fn: &ReservedTimesFn) {
    // TODO: could be added if necessary, but it should be thought carefully to keep solution feasibility
}

/// Creates a reserved time function from reserved time index.
pub(crate) fn create_reserved_times_fn(
    reserved_times_index: ReservedTimesIndex,
) -> Result<ReservedTimesFn, GenericError> {
    if reserved_times_index.is_empty() {
        return Ok(Arc::new(|_, _| None));
    }

    let reserved_times = reserved_times_index.into_iter().try_fold(
        HashMap::<_, (Vec<_>, Vec<_>)>::new(),
        |mut acc, (actor, mut times)| {
            // NOTE do not allow different types to simplify interval searching
            let are_same_types = times.windows(2).all(|pair| {
                if let [ReservedTimeSpan { time: a, .. }, ReservedTimeSpan { time: b, .. }] = pair {
                    matches!(
                        (a, b),
                        (TimeSpan::Window(_), TimeSpan::Window(_)) | (TimeSpan::Offset(_), TimeSpan::Offset(_))
                    )
                } else {
                    false
                }
            });

            if !are_same_types {
                return Err("has reserved types of different time span types".to_string());
            }

            times.sort_by(|ReservedTimeSpan { time: a, .. }, ReservedTimeSpan { time: b, .. }| {
                let (a, b) = match (a, b) {
                    (TimeSpan::Window(a), TimeSpan::Window(b)) => (a.start, b.start),
                    (TimeSpan::Offset(a), TimeSpan::Offset(b)) => (a.start, b.start),
                    _ => unreachable!(),
                };
                a.total_cmp(&b)
            });
            let has_no_intersections = times.windows(2).all(|pair| {
                if let [ReservedTimeSpan { time: a, .. }, ReservedTimeSpan { time: b, .. }] = pair {
                    !a.intersects(0., &b.to_time_window(0.))
                } else {
                    false
                }
            });

            if has_no_intersections {
                let (indices, intervals): (Vec<_>, Vec<_>) = times
                    .into_iter()
                    .map(|span| {
                        let start = match &span.time {
                            TimeSpan::Window(time) => time.start,
                            TimeSpan::Offset(time) => time.start,
                        };

                        (start as u64, span)
                    })
                    .unzip();
                acc.insert(actor, (indices, intervals));

                Ok(acc)
            } else {
                Err("reserved times have intersections".to_string())
            }
        },
    )?;

    // NOTE: a reserved time is owned by the activity or travel leg which is in progress at the moment
    //       it becomes due, so the search is driven by reserved_time.time.start. As the reserved time
    //       is inserted into the timeline, everything after the owner is shifted by its duration, which
    //       keeps the exclusive intersection below matching exactly one interval.
    Ok(Arc::new(move |route: &Route, time_window: &TimeWindow| {
        reserved_times.get(&route.actor).and_then(|(indices, intervals)| {
            let offset = route.tour.start().map(|a| a.schedule.departure).unwrap_or(0.);

            // NOTE map external absolute time window to time span's start/end
            let (interval_start, interval_end) = match intervals.first().map(|rt| &rt.time) {
                Some(TimeSpan::Offset(_)) => (time_window.start - offset, time_window.end - offset),
                Some(TimeSpan::Window(_)) => (time_window.start, time_window.end),
                _ => unreachable!(),
            };

            match indices.binary_search(&(interval_start as u64)) {
                Ok(idx) => intervals.get(idx),
                Err(idx) => (idx.max(1) - 1..=idx) // NOTE left (earliest) wins
                    .map(|idx| intervals.get(idx))
                    .find(|reserved_time| {
                        reserved_time.is_some_and(|reserved_time| {
                            let (reserved_start, reserved_end) = match &reserved_time.time {
                                TimeSpan::Offset(to) => (to.start, to.start + reserved_time.duration),
                                TimeSpan::Window(tw) => (tw.start, tw.start + reserved_time.duration),
                            };

                            // NOTE use exclusive intersection
                            interval_start < reserved_end && reserved_start < interval_end
                        })
                    })
                    .flatten(),
            }
            .map(|reserved_time| reserved_time.to_reserved_time_window(offset))
        })
    }))
}
