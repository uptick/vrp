//! Objectives which trade what the tours achieve against the travel it takes. Each tour in use
//! scores `gain - travel_penalty * travel`, where `travel` is the travel time of every leg,
//! including those out of and back to the depot. The fitness is the sum negated. The gain is:
//!
//! * for net utilization, the service time of productive jobs, so the penalty is seconds of
//!   service per second of travel;
//! * for net value, the value of the served jobs, so the penalty is value per second of travel.
//!
//! The score is not divided by shift length: a ratio over the shifts in use improves by leaving a
//! quiet shift empty, rather than by doing more work.
//!
//! Ranked above an objective which rewards assignment, they prefer to drop a job whose gain is less
//! than `travel_penalty` times the travel it adds. That is best-effort: insertion places every job
//! it can, so a job is only left out when the search tries a solution without it.
//!
//! Travel is read from the routing matrix rather than the schedule, which folds reserved time (such
//! as a required break taken on the road) into a leg's travel. With a time-dependent matrix, the
//! time-independent duration is used.
//!
//! A vicinity cluster is served from one stop: the vehicle parks and the jobs are reached on foot.
//! Only the jobs' service times count as service; the parking and walking count as travel. The
//! parking is not recorded on the cluster, so it is passed in.
//!
//! Objective only - no constraint or state.

#[cfg(test)]
#[path = "../../../tests/unit/construction/features/utilization_test.rs"]
mod utilization_test;

use super::*;
use crate::construction::clustering::vicinity::ClusterInfoDimension;
use crate::models::solution::{Activity, Route};

/// Specifies a function which tells whether time spent serving a job counts as productive: true for
/// real work, false for activities such as breaks or reloads.
pub type ProductiveJobFn = Arc<dyn Fn(&Single) -> bool + Send + Sync>;

/// Creates a feature which maximizes net utilization: the service time of productive jobs less
/// `travel_penalty` times the travel time of each tour in use.
///
/// `cluster_parking` is the time spent parking at a cluster of jobs, which counts as travel; zero
/// when jobs are not clustered.
pub fn create_maximize_utilization_feature(
    name: &str,
    travel_penalty: Float,
    transport: Arc<dyn TransportCost>,
    is_productive_fn: ProductiveJobFn,
    cluster_parking: Duration,
) -> GenericResult<Feature> {
    create_feature(name, travel_penalty, transport, Gain::ServiceTime(is_productive_fn), cluster_parking)
}

/// Creates a feature which maximizes net value: the value of the served jobs, as `job_value_fn`
/// reads it, less `travel_penalty` times the travel time of each tour in use.
///
/// `cluster_parking` is the time spent parking at a cluster of jobs, which counts as travel; zero
/// when jobs are not clustered.
pub fn create_maximize_net_value_feature(
    name: &str,
    travel_penalty: Float,
    transport: Arc<dyn TransportCost>,
    job_value_fn: SimpleValueFn,
    cluster_parking: Duration,
) -> GenericResult<Feature> {
    create_feature(name, travel_penalty, transport, Gain::JobValue(job_value_fn), cluster_parking)
}

fn create_feature(
    name: &str,
    travel_penalty: Float,
    transport: Arc<dyn TransportCost>,
    gain: Gain,
    cluster_parking: Duration,
) -> GenericResult<Feature> {
    FeatureBuilder::default()
        .with_name(name)
        .with_objective(TravelPenalisedObjective { travel_penalty, transport, gain, cluster_parking })
        .build()
}

/// What a tour achieves, which the objective trades against its travel.
enum Gain {
    /// The service time of productive activities. Depends on the place an activity is served at.
    ServiceTime(ProductiveJobFn),
    /// The value of the jobs served. Counted once per job, however many activities it has.
    JobValue(SimpleValueFn),
}

struct TravelPenalisedObjective {
    travel_penalty: Float,
    transport: Arc<dyn TransportCost>,
    gain: Gain,
    cluster_parking: Duration,
}

impl TravelPenalisedObjective {
    fn travel(&self, route: &Route, from: &Activity, to: &Activity) -> Duration {
        self.transport.duration_approx(&route.actor.vehicle.profile, from.place.location, to.place.location)
    }

    /// Splits the time spent at an activity into its service time and the time spent moving on foot
    /// and parking, which is only ever non-zero for a cluster of jobs.
    fn split_time_at(&self, activity: &Activity) -> (Duration, Duration) {
        let duration = activity.place.duration;
        let Some(clustered) = activity.job.as_ref().and_then(|single| single.dimens.get_cluster_info()) else {
            return (duration, Duration::default());
        };

        // the first job's service time includes the parking
        let service = clustered.iter().map(|info| info.service_time).sum::<Duration>() - self.cluster_parking;
        let service = service.clamp(Duration::default(), duration);

        (service, duration - service)
    }

    fn service_time(&self, activity: &Activity) -> Duration {
        match (&self.gain, &activity.job) {
            (Gain::ServiceTime(is_productive_fn), Some(single)) if is_productive_fn(single) => {
                self.split_time_at(activity).0
            }
            _ => Duration::default(),
        }
    }

    fn movement(&self, activity: &Activity) -> Duration {
        self.split_time_at(activity).1
    }

    fn job_value(&self, job: &Job) -> Float {
        match &self.gain {
            Gain::JobValue(job_value_fn) => job_value_fn(job),
            Gain::ServiceTime(_) => Float::default(),
        }
    }

    fn score(&self, gain: Float, travel: Duration) -> Cost {
        self.travel_penalty * travel - gain
    }
}

impl FeatureObjective for TravelPenalisedObjective {
    fn fitness(&self, solution: &InsertionContext) -> Cost {
        solution
            .solution
            .routes
            .iter()
            .map(|route_ctx| route_ctx.route())
            // a tour with no jobs is not driven, even when its start and end differ
            .filter(|route| route.tour.has_jobs())
            .map(|route| {
                let activities = route.tour.all_activities();
                let travel = activities
                    .clone()
                    .zip(activities.clone().skip(1))
                    .map(|(from, to)| self.travel(route, from, to))
                    .chain(activities.clone().map(|activity| self.movement(activity)))
                    .sum::<Duration>();
                let gain = match &self.gain {
                    Gain::ServiceTime(_) => activities.map(|activity| self.service_time(activity)).sum::<Float>(),
                    Gain::JobValue(_) => route.tour.jobs().map(|job| self.job_value(job)).sum::<Float>(),
                };

                self.score(gain, travel)
            })
            .sum()
    }

    fn estimate(&self, move_ctx: &MoveContext<'_>) -> Cost {
        match move_ctx {
            // a job's value does not depend on where it goes, so it is scored once per job
            MoveContext::Route { job, .. } => self.score(self.job_value(job), Duration::default()),
            // the detour, and the service time, depend on the place chosen for the activity
            MoveContext::Activity { route_ctx, activity_ctx, .. } => {
                let route = route_ctx.route();
                let (prev, target) = (activity_ctx.prev, activity_ctx.target);

                let detour = match activity_ctx.next {
                    Some(next) => {
                        self.travel(route, prev, target) + self.travel(route, target, next)
                            - self.travel(route, prev, next)
                    }
                    None => self.travel(route, prev, target),
                };

                self.score(self.service_time(target), detour + self.movement(target))
            }
        }
    }
}
