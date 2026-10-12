//! Objectives which score jobs against their due dates: how many are late, how late, and how
//! early in their window they are served. Rank them in that order: the count decides how many
//! jobs are late, the sum how late the rest are, and window use only orders jobs which are on
//! time. The sum alone would make five jobs a day late to save one five days late; the count
//! alone gives the search no gradient between improvements.
//!
//! A due date is soft, so it never makes a job unassignable. A job left unassigned is scored as
//! arriving when the plan ends, which no served job arrives after, so dropping a job never scores
//! better than serving it. The plan ends at the last shift end, or at a later horizon end when
//! one is given: a job due after the last shift on the plan's final day still cannot be served
//! on time once the plan is over.
//!
//! Lateness is measured from arrival, so a due date means "be there by": a caller who means
//! "any time that day" sends the end of the day.

#[cfg(test)]
#[path = "../../../tests/unit/construction/features/due_date_test.rs"]
mod due_date_test;

use super::*;
use crate::construction::enablers::calculate_travel;

/// Specifies a function which returns a job's due date, if it has one.
pub type JobDueDateFn = Arc<dyn Fn(&Job) -> Option<Timestamp> + Send + Sync>;

/// Specifies what a due-date objective scores each job with a due date by.
#[derive(Clone, Copy)]
pub enum DueDateMeasure {
    /// One per job arriving after its due date.
    LateJobs,
    /// How long after its due date a job arrives.
    Tardiness,
    /// How much of its window has passed when a job arrives, from 0 to 1. The window runs from
    /// when the job may first be served (the plan start, or its earliest time window if later) to
    /// its due date. A share rather than a duration, so a day counts for more against a weekly's
    /// week than an annual's year, and dropping work due soon after the plan costs more than
    /// dropping work due much later.
    WindowUse,
}

/// Creates a feature which minimizes the given due-date measure. `horizon_end` moves the plan's
/// end past its last shift end; an earlier one is ignored.
pub fn create_due_date_feature(
    name: &str,
    measure: DueDateMeasure,
    job_due_date_fn: JobDueDateFn,
    transport: Arc<dyn TransportCost>,
    fleet: &Fleet,
    horizon_end: Option<Timestamp>,
) -> GenericResult<Feature> {
    // open-ended shifts end at `Float::MAX`
    let shift_ends = fleet.actors.iter().map(|actor| actor.detail.time.end).filter(|&end| end < Float::MAX);

    FeatureBuilder::default()
        .with_name(name)
        .with_objective(DueDateObjective {
            measure,
            job_due_date_fn,
            transport,
            plan_start: get_earliest_shift_start(fleet),
            plan_end: shift_ends.chain(horizon_end).max_by(Float::total_cmp),
        })
        .build()
}

struct DueDateObjective {
    measure: DueDateMeasure,
    job_due_date_fn: JobDueDateFn,
    transport: Arc<dyn TransportCost>,
    plan_start: Timestamp,
    plan_end: Option<Timestamp>,
}

impl FeatureObjective for DueDateObjective {
    fn fitness(&self, solution: &InsertionContext) -> Cost {
        let routes = &solution.solution.routes;
        let served = routes.iter().flat_map(|route_ctx| {
            let tour = &route_ctx.route().tour;
            tour.jobs().filter_map(move |job| {
                let due_date = (self.job_due_date_fn)(job)?;
                // a multi-job is done when its last activity is
                let arrival =
                    tour.job_activities(job).map(|activity| activity.schedule.arrival).max_by(Float::total_cmp)?;

                Some(self.score(job, arrival, due_date))
            })
        });

        // with open-ended shifts the latest arrival is the plan's end, so that no served job is
        // ever scored as arriving after a dropped one
        let latest_arrivals = routes.iter().filter_map(|route_ctx| route_ctx.route().tour.end());
        let plan_end =
            self.plan_end.into_iter().chain(latest_arrivals.map(|end| end.schedule.arrival)).max_by(Float::total_cmp);
        let dropped = solution
            .solution
            .unassigned
            .keys()
            .filter_map(|job| Some(self.score(job, plan_end?, (self.job_due_date_fn)(job)?)));

        served.chain(dropped).sum()
    }

    fn estimate(&self, move_ctx: &MoveContext<'_>) -> Cost {
        // arrival is only known once a position within the route is proposed
        let MoveContext::Activity { route_ctx, activity_ctx, .. } = move_ctx else {
            return Cost::default();
        };
        let Some(job) = activity_ctx.target.retrieve_job() else {
            return Cost::default();
        };
        let Some(due_date) = (self.job_due_date_fn)(&job) else {
            return Cost::default();
        };

        // NOTE: scores only the inserted job, not how the insertion delays the jobs after it
        let (_, (prev_to_target_duration, _)) = calculate_travel(route_ctx, activity_ctx, self.transport.as_ref());

        self.score(&job, activity_ctx.prev.schedule.departure + prev_to_target_duration, due_date)
    }
}

impl DueDateObjective {
    fn score(&self, job: &Job, arrival: Timestamp, due_date: Timestamp) -> Cost {
        match self.measure {
            DueDateMeasure::LateJobs => Cost::from(arrival > due_date),
            DueDateMeasure::Tardiness => (arrival - due_date).max(0.),
            DueDateMeasure::WindowUse => {
                // offset time windows are relative to a tour, so only absolute ones count
                let release = job
                    .places()
                    .flat_map(|place| place.times.iter())
                    .filter_map(|time| time.as_time_window())
                    .map(|window| window.start)
                    .min_by(Float::total_cmp)
                    .map_or(self.plan_start, |start| start.max(self.plan_start));
                let window = due_date - release;

                // a job due before it may be served is overdue: tardiness orders those
                if window > 0. { ((arrival - release) / window).clamp(0., 1.) } else { 1. }
            }
        }
    }
}
