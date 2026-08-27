//! Prefers scheduling work into the earliest shifts of the planning period.
//!
//! In a multi-day plan every vehicle day is a separate shift, and every shift is a separate actor
//! (route). This objective scores each *shift in use* by how far it starts after the earliest shift
//! in the fleet, so the cheapest solution is the one that draws its shifts from the front of the
//! planning window: given five days of shifts and two days of work, it prefers Mon/Tue over Thu/Fri.
//!
//! It is deliberately indifferent to how the work is split across shifts that are already in use -
//! only opening a shift costs anything. Weighting per job instead would make it pay unbounded
//! travel to drag one more job into an earlier day, since it outranks the cost objective: in a
//! two-cluster test that cost 42% extra distance to move a single job one day earlier. Where the
//! work goes among the chosen days is left to the cost objective below.
//!
//! It deliberately uses the shift's *static* start (the shift window's earliest, not the route's
//! optimised departure or its arrival back at the depot), so it says nothing at all about the order
//! or timing of jobs *within* a day. That is the difference from `minimize_arrival_time`, which
//! scores the tour's end arrival and therefore also pushes every route to finish as early as
//! possible - overriding cost-based intra-day decisions such as waiting for a property's access
//! window.
//!
//! Degenerate on its own: an empty solution costs nothing, so it needs a tier above it that
//! rewards assignment (the same caveat as `depot_proximity` in soft mode). `minimize_unassigned`
//! guards outright. A `total_value` tier guards only the jobs that carry a value, since it sums
//! value over assigned jobs - one with no value of its own is invisible to it, and this objective
//! will happily leave it unassigned. The pragmatic format rejects a goal with neither ranked above
//! (E1608); it does not check which jobs carry a value, so the `total_value` gap stays open.
//!
//! Neither this nor `fleet_usage` (minimize tours) can veto the other during insertion - both
//! charge only when a shift is *opened* - but their relative order still decides the outcome
//! whenever a later set of shifts would use fewer tours. That needs non-uniform per-day capacity:
//! with a uniform fleet, k days of work always means the earliest k shifts and the two agree. With
//! per-day capacities of 4/4/4/4/9 and nine jobs, days 1-3 sum to a delay of 0 + 1 + 2 against day
//! 5's 4, so this objective placed above minimize tours takes three tours on days 1-3, and placed
//! below it takes one tour on day 5. Pick the order that matches what you are trading: earliest
//! work, or fewest vehicle days.
//!
//! Summing delays also means several early shifts can tie exactly with one late shift, since
//! `0 + 1 + 2` equals `3`; minimize tours then breaks the tie from either position.
//!
//! Objective only - no constraint or state.

use super::*;
use crate::construction::enablers::FirstJobArrivalFloorDimension;
use crate::models::solution::Route;

/// Creates a feature which prefers to serve jobs in the earliest shifts of the planning period.
///
/// `origin` is the timestamp that counts as "no delay" - normally the earliest shift start in the
/// fleet, see [`get_earliest_shift_start`]. Fitness is then the summed delay of the shifts in use,
/// and a solution which does all its work in the earliest shift scores zero.
pub fn create_prefer_early_tours_feature(name: &str, origin: Timestamp) -> GenericResult<Feature> {
    FeatureBuilder::default().with_name(name).with_objective(PreferEarlyToursObjective { origin }).build()
}

/// Returns the static start of an actor's shift.
///
/// Normally this is the shift window's earliest bound. When the actor allows out-of-hours depot
/// travel that bound is relaxed to `None` (so `detail.time.start` collapses to zero) and the real
/// shift start is kept as the first-job arrival floor instead, so prefer that when it is set.
pub fn get_shift_start(actor: &Actor) -> Timestamp {
    actor.vehicle.dimens.get_first_job_arrival_floor().copied().unwrap_or(actor.detail.time.start)
}

/// Returns the earliest shift start across the whole fleet, to be used as the `origin` of
/// [`create_prefer_early_tours_feature`].
pub fn get_earliest_shift_start(fleet: &Fleet) -> Timestamp {
    fleet.actors.iter().map(|actor| get_shift_start(actor)).min_by(|left, right| left.total_cmp(right)).unwrap_or(0.)
}

struct PreferEarlyToursObjective {
    origin: Timestamp,
}

impl PreferEarlyToursObjective {
    /// Returns how long after `origin` the route's shift starts: the cost of opening this shift
    /// rather than one at the front of the planning window.
    fn delay(&self, route: &Route) -> Cost {
        (get_shift_start(route.actor.as_ref()) - self.origin).max(Cost::default())
    }
}

impl FeatureObjective for PreferEarlyToursObjective {
    fn fitness(&self, solution: &InsertionContext) -> Cost {
        solution
            .solution
            .routes
            .iter()
            .filter(|route_ctx| route_ctx.route().tour.has_jobs())
            .map(|route_ctx| self.delay(route_ctx.route()))
            .sum()
    }

    fn estimate(&self, move_ctx: &MoveContext<'_>) -> Cost {
        match move_ctx {
            // only opening a shift costs its delay; once a shift is in use, moving work into it is
            // free here, leaving the split between open shifts to the cost objective below
            MoveContext::Route { route_ctx, .. } => {
                if route_ctx.route().tour.has_jobs() {
                    Cost::default()
                } else {
                    self.delay(route_ctx.route())
                }
            }
            // the objective says nothing about where in the tour the job goes
            MoveContext::Activity { .. } => Cost::default(),
        }
    }
}
