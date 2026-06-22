//! Prefers assigning jobs whose location is close (by travel time) to the vehicle's depot.
//!
//! The depot is the vehicle shift's start location (`route.actor.detail.start`). This objective
//! scores each served job by the travel duration from its route's depot to the job, preferring
//! assignments that keep a vehicle working near its own depot even when that is not the globally
//! cheapest route.
//!
//! It has two modes, selected by the optional `cap`:
//!
//! * **Soft (`cap` = `None`)** — purely a penalty over *assigned* jobs (unassigned jobs cost
//!   nothing). Intended as a member of a `weighted-sum` tier where it trades off against other
//!   objectives. On its own as a dominant tier it is degenerate: the cheapest solution assigns
//!   nothing (zero travel), so it must not be used standalone in this mode.
//!
//! * **Capped / "depot radius" (`cap` = `Some(c)`)** — leaving a job unassigned costs `c`, while
//!   assigning it costs its (penalised) depot→job travel. Minimising this assigns every job whose
//!   depot travel is below `c` (nearest first) and drops the rest, so it drives *proximity-gated
//!   assignment* with a tunable radius and is safe to use as a dominant lexicographic tier (the
//!   empty solution costs `N * c`, never wins).
//!
//! Objective only — no constraint or state.

#[cfg(test)]
#[path = "../../../tests/unit/construction/features/depot_proximity_test.rs"]
mod depot_proximity_test;

use super::*;
use crate::models::problem::TransportCost;
use crate::models::solution::Route;
use crate::utils::Either;
use std::iter::empty;

/// Maps a raw depot→job travel duration to a penalty cost. Linear (identity) today, but this
/// signature is the single extension point for a future non-linear penalty shape (e.g. quadratic
/// or thresholded) without touching the objective plumbing.
pub type DepotPenaltyFn = Arc<dyn Fn(Duration) -> Cost + Send + Sync>;

/// Creates a feature which prefers assigning jobs close (by travel time) to the vehicle's depot.
///
/// `penalty_fn` converts the raw depot→job travel duration into a penalty cost; the default caller
/// passes the identity closure (`Arc::new(|duration| duration)`) for a linear penalty.
///
/// `cap` selects the mode: `None` for a soft weighted-sum penalty (assigned jobs only), or
/// `Some(c)` for a "depot radius" where leaving a job unassigned costs `c` — driving proximity-gated
/// assignment (assign jobs within `c` of the depot, nearest first; drop the rest). `c` is expressed
/// in the same (penalised) units the `penalty_fn` returns.
pub fn create_minimize_depot_travel_time_feature(
    name: &str,
    transport: Arc<dyn TransportCost>,
    penalty_fn: DepotPenaltyFn,
    cap: Option<Cost>,
) -> Result<Feature, GenericError> {
    FeatureBuilder::default()
        .with_name(name)
        .with_objective(MinimizeDepotTravelTimeObjective { transport, penalty_fn, cap })
        .build()
}

struct MinimizeDepotTravelTimeObjective {
    transport: Arc<dyn TransportCost>,
    penalty_fn: DepotPenaltyFn,
    cap: Option<Cost>,
}

impl MinimizeDepotTravelTimeObjective {
    /// Returns the penalised travel duration from the route's depot (shift start) to the nearest
    /// of the job's candidate locations. Returns `0` when the route has no depot or the job has no
    /// located place.
    fn depot_to_job_duration(&self, route: &Route, job: &Job) -> Cost {
        let Some(depot) = route.actor.detail.start.as_ref().map(|place| place.location) else {
            return 0.;
        };
        let profile = &route.actor.vehicle.profile;

        // Take the closest candidate location across the job's singles (handles Single and Multi).
        let nearest = job
            .places()
            .filter_map(|place| place.location)
            .map(|location| self.transport.duration_approx(profile, depot, location))
            .min_by(|left, right| left.total_cmp(right));

        match nearest {
            Some(duration) => (self.penalty_fn)(duration),
            None => 0.,
        }
    }
}

impl FeatureObjective for MinimizeDepotTravelTimeObjective {
    fn fitness(&self, solution: &InsertionContext) -> Cost {
        let assigned = solution.solution.routes.iter().fold(0., |acc, route_ctx| {
            route_ctx.route().tour.jobs().fold(acc, |acc, job| acc + self.depot_to_job_duration(route_ctx.route(), job))
        });

        match self.cap {
            // Capped mode: each unassigned job costs `cap`, so dropping near jobs is worse than
            // assigning them and the empty solution does not win. Mirrors `minimize_unassigned`'s
            // handling of `ignored` jobs when no routes exist yet.
            Some(cap) => {
                let unassigned = if solution.solution.routes.is_empty() {
                    Either::Left(solution.solution.ignored.iter())
                } else {
                    Either::Right(empty())
                }
                .chain(solution.solution.unassigned.keys())
                .count();

                assigned + unassigned as Cost * cap
            }
            None => assigned,
        }
    }

    fn estimate(&self, move_ctx: &MoveContext<'_>) -> Cost {
        match move_ctx {
            // Inserting a job removes its unassigned `cap` penalty and adds its depot travel, so the
            // marginal cost is `travel - cap` (negative — rewarded — for jobs within the radius).
            MoveContext::Route { route_ctx, job, .. } => {
                let travel = self.depot_to_job_duration(route_ctx.route(), job);
                match self.cap {
                    Some(cap) => travel - cap,
                    None => travel,
                }
            }
            // Proximity is per-job, not per-activity; scoring here would double-count.
            MoveContext::Activity { .. } => Cost::default(),
        }
    }
}
