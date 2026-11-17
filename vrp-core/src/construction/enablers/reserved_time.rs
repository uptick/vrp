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
    /// An optional id which can be used to trace the reserved time back to its origin.
    pub id: Option<String>,
}

impl ReservedTimeSpan {
    /// Converts `ReservedTimeSpan` to `ReservedTimeWindow`.
    pub fn to_reserved_time_window(&self, offset: Timestamp) -> ReservedTimeWindow {
        ReservedTimeWindow { time: self.time.to_time_window(offset), duration: self.duration, id: self.id.clone() }
    }
}

/// Represent a reserved time window entity.
#[derive(Clone, Debug)]
pub struct ReservedTimeWindow {
    /// A specific time window when an extra reserved duration should be applied.
    pub time: TimeWindow,
    /// An extra duration to be applied at given time.
    pub duration: Duration,
    /// An optional id which can be used to trace the reserved time back to its origin.
    pub id: Option<String>,
}

/// Specifies reserved time index type.
pub type ReservedTimesIndex = HashMap<Arc<Actor>, Vec<ReservedTimeSpan>>;

/// Specifies a function which returns an extra reserved time window for given actor. This reserved
/// time should be considered for planning.
pub(crate) type ReservedTimesFn = Arc<dyn Fn(&Route, &TimeWindow) -> Option<ReservedTimeWindow> + Send + Sync>;

/// Provides way to calculate activity costs which might contain reserved time.
///
/// Reserved time never interrupts a service: it is taken either before the service starts or after
/// it is finished. See [`ReservedSlot`] for the rules.
pub struct DynamicActivityCost {
    reserved_times: HashMap<Arc<Actor>, ActorReservedTimes>,
}

impl DynamicActivityCost {
    /// Creates a new instance of `DynamicActivityCost` with given reserved time function.
    pub fn new(reserved_times_index: ReservedTimesIndex) -> Result<Self, GenericError> {
        // NOTE reserved times have to conform the same rules as the ones used by reserved time lookup
        create_reserved_times_fn(reserved_times_index.clone())?;

        let reserved_times = reserved_times_index
            .into_iter()
            .filter(|(_, times)| !times.is_empty())
            .map(|(actor, times)| (actor, ActorReservedTimes::new(&times)))
            .collect();

        Ok(Self { reserved_times })
    }
}

impl ActivityCost for DynamicActivityCost {
    fn estimate_departure(
        &self,
        route: &Route,
        activity: &Activity,
        arrival: Timestamp,
    ) -> ControlFlow<Timestamp, Timestamp> {
        let place = &activity.place;
        let departure = arrival.max(place.time.start) + place.duration;

        let Some(reserved_times) = self.reserved_times.get(&route.actor) else {
            return ControlFlow::Continue(departure);
        };

        let slots = reserved_times.slots_since(arrival, reserved_times.offset(route));

        estimate_service_start(slots, arrival, &place.time, place.duration).map_or(
            ControlFlow::Continue(departure),
            |(service_start, _)| {
                let departure = service_start + place.duration;

                // NOTE: do not allow to start work after break finished
                if service_start > place.time.end {
                    // TODO this branch is the reason why departure rescheduling is disabled.
                    //      theoretically, rescheduling should be aware somehow about dynamic costs
                    ControlFlow::Break(departure)
                } else {
                    ControlFlow::Continue(departure)
                }
            },
        )
    }

    fn estimate_arrival(
        &self,
        route: &Route,
        activity: &Activity,
        departure: Timestamp,
    ) -> ControlFlow<Timestamp, Timestamp> {
        let place = &activity.place;
        let latest_service_start = place.time.end.min(departure - place.duration);

        let Some(reserved_times) = self.reserved_times.get(&route.actor) else {
            return ControlFlow::Continue(latest_service_start);
        };

        let offset = reserved_times.offset(route);
        // NOTE reserved times which start later cannot be reached by the service
        let slots = reserved_times.slots_until(latest_service_start.max(place.time.start) + place.duration, offset);

        estimate_latest_arrival(slots, latest_service_start, &place.time, place.duration)
    }
}

/// A reserved time which starts within `[earliest, latest]` and lasts for `duration`.
///
/// The rest of the solver (travel, lookup) considers reserved time at its latest start only. An
/// activity is different, as its service cannot be interrupted. So, when the service would run
/// over the latest start, reserved time is taken first and the service is postponed until it is
/// over. In that case reserved time starts as early as it is allowed, but not so early that the
/// service is finished before the latest end: otherwise, it would be applied again afterward.
#[derive(Clone, Copy, Debug)]
struct ReservedSlot {
    earliest: Timestamp,
    latest: Timestamp,
    duration: Duration,
}

impl ReservedSlot {
    /// Returns the time when reserved time is over if it starts at the latest time.
    fn latest_end(&self) -> Timestamp {
        self.latest + self.duration
    }

    fn shift(self, offset: Timestamp) -> Self {
        Self { earliest: self.earliest + offset, latest: self.latest + offset, ..self }
    }

    /// Applies reserved time to an activity which can be served since `activity_start` for
    /// `service_duration` by a vehicle which has nothing else to do since `free` time.
    /// Returns the time since the vehicle has nothing else to do once reserved time is over, or
    /// `None` if reserved time has no effect: it is either over already or starts after the service.
    fn apply(&self, free: Timestamp, activity_start: Timestamp, service_duration: Duration) -> Option<Timestamp> {
        let service_start = free.max(activity_start);

        if free >= self.latest_end() || self.latest >= service_start + service_duration {
            return None;
        }

        Some(if self.latest_end() <= service_start {
            // NOTE reserved time fits into waiting time at its latest start
            self.latest_end()
        } else {
            free.max(self.earliest).max(self.latest - service_duration) + self.duration
        })
    }

    /// Does the opposite of `apply`: returns the latest `free` time which leaves the vehicle with
    /// nothing else to do by `latest_free` time once reserved time is applied. Returns `None` if
    /// there is no such time.
    fn revert(
        &self,
        latest_free: Timestamp,
        activity_start: Timestamp,
        service_duration: Duration,
    ) -> Option<Timestamp> {
        let latest_before = self.latest - service_duration;

        if latest_free >= self.latest_end() {
            // NOTE reserved time is over
            Some(latest_free)
        } else if activity_start >= self.latest_end() {
            // NOTE reserved time is taken while waiting and cannot be over in time
            None
        } else if latest_free - self.duration >= self.earliest.max(latest_before) {
            // NOTE reserved time is taken first and the service is postponed
            Some(latest_free - self.duration)
        } else if activity_start <= latest_before {
            // NOTE service has to be finished before reserved time starts
            Some(latest_free.min(latest_before))
        } else {
            None
        }
    }
}

impl From<&ReservedTimeWindow> for ReservedSlot {
    fn from(reserved_time: &ReservedTimeWindow) -> Self {
        Self { earliest: reserved_time.time.start, latest: reserved_time.time.end, duration: reserved_time.duration }
    }
}

/// Keeps reserved times of a single actor.
struct ActorReservedTimes {
    /// Reserved times sorted by their latest start. Relative to tour's start when `is_offset` is set.
    slots: Vec<ReservedSlot>,
    /// For each slot, the greatest latest end among it and the slots before it.
    latest_ends: Vec<Timestamp>,
    is_offset: bool,
}

impl ActorReservedTimes {
    fn new(times: &[ReservedTimeSpan]) -> Self {
        let mut slots = times
            .iter()
            .map(|span| {
                let (earliest, latest) = match &span.time {
                    TimeSpan::Window(time) => (time.start, time.end),
                    TimeSpan::Offset(time) => (time.start, time.end),
                };
                ReservedSlot { earliest, latest, duration: span.duration }
            })
            .collect::<Vec<_>>();
        slots.sort_by(|a, b| a.latest.total_cmp(&b.latest));

        let latest_ends = slots
            .iter()
            .scan(Timestamp::MIN, |max_end, slot| {
                *max_end = max_end.max(slot.latest_end());
                Some(*max_end)
            })
            .collect();

        Self {
            slots,
            latest_ends,
            is_offset: matches!(times.first().map(|span| &span.time), Some(TimeSpan::Offset(_))),
        }
    }

    fn offset(&self, route: &Route) -> Timestamp {
        if self.is_offset { route.tour.start().map_or(0., |start| start.schedule.departure) } else { 0. }
    }

    /// Returns reserved times which might not be over by given time, the earliest first.
    fn slots_since(&self, time: Timestamp, offset: Timestamp) -> impl Iterator<Item = ReservedSlot> + '_ {
        let idx = self.latest_ends.partition_point(|latest_end| latest_end + offset <= time);

        self.slots[idx..].iter().map(move |slot| slot.shift(offset))
    }

    /// Returns reserved times with the latest start before given time, the latest first. Each one
    /// comes with the greatest latest end among it and the reserved times before it.
    fn slots_until(&self, time: Timestamp, offset: Timestamp) -> impl Iterator<Item = (ReservedSlot, Timestamp)> + '_ {
        let idx = self.slots.partition_point(|slot| slot.latest + offset < time);

        self.slots[..idx]
            .iter()
            .zip(self.latest_ends[..idx].iter())
            .rev()
            .map(move |(slot, latest_end)| (slot.shift(offset), latest_end + offset))
    }
}

/// Estimates when service at an activity starts when reserved times apply to it, and how much reserved
/// time postpones it. Returns `None` when no reserved time applies.
///
/// Reserved time never interrupts a service: it is either taken while waiting for the activity's
/// time window to open, or the service is postponed until the reserved time is finished. Postponing
/// the service past one reserved time can move it into the next one, so each of them is applied in turn.
pub fn estimate_service_start_with_reserved_times(
    reserved_times: &[ReservedTimeWindow],
    arrival: Timestamp,
    activity_tw: &TimeWindow,
    service_duration: Duration,
) -> Option<(Timestamp, Duration)> {
    let mut slots = reserved_times.iter().map(ReservedSlot::from).collect::<Vec<_>>();
    slots.sort_by(|a, b| a.latest.total_cmp(&b.latest));

    estimate_service_start(slots.into_iter(), arrival, activity_tw, service_duration)
}

/// Estimates service start and duration of reserved times which postpone it. Expects reserved times
/// to be sorted by their latest start, the earliest first.
fn estimate_service_start(
    slots: impl Iterator<Item = ReservedSlot>,
    arrival: Timestamp,
    activity_tw: &TimeWindow,
    service_duration: Duration,
) -> Option<(Timestamp, Duration)> {
    let mut free = arrival;
    let mut postponed_by = None;

    for slot in slots {
        let service_start = free.max(activity_tw.start);

        // NOTE service is finished before this and any later reserved time starts
        if slot.latest >= service_start + service_duration {
            break;
        }

        if let Some(free_after) = slot.apply(free, activity_tw.start, service_duration) {
            let duration = if free_after > service_start { slot.duration } else { 0. };

            postponed_by = Some(postponed_by.unwrap_or(0.) + duration);
            free = free_after;
        }
    }

    postponed_by.map(|postponed_by| (free.max(activity_tw.start), postponed_by))
}

/// Estimates the latest arrival which allows service to start not later than `latest_service_start`.
/// This is the opposite of `estimate_service_start`, hence reserved times are expected in the
/// opposite order: the latest first.
fn estimate_latest_arrival(
    slots: impl Iterator<Item = (ReservedSlot, Timestamp)>,
    latest_service_start: Timestamp,
    activity_tw: &TimeWindow,
    service_duration: Duration,
) -> ControlFlow<Timestamp, Timestamp> {
    let mut latest_free = latest_service_start;

    for (slot, latest_end) in slots {
        // NOTE this and all earlier reserved times are over
        if latest_free >= latest_end {
            break;
        }

        match slot.revert(latest_free, activity_tw.start, service_duration) {
            Some(value) => latest_free = value,
            // NOTE service cannot start in time whenever the vehicle arrives. Return the arrival
            // which at least does not postpone the service more than necessary
            None => return ControlFlow::Break(latest_free.min(slot.latest - service_duration)),
        }
    }

    ControlFlow::Continue(latest_free)
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
                            TimeSpan::Window(time) => time.end,
                            TimeSpan::Offset(time) => time.end,
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

    // NOTE: this function considers only latest time from reserved time
    //       reserved_time.time.start is ignored and should be handled by post processing
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
                                TimeSpan::Offset(to) => (to.end, to.end + reserved_time.duration),
                                TimeSpan::Window(tw) => (tw.end, tw.end + reserved_time.duration),
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
