//! Objectives which score jobs against their due dates: how many are late, and how late. Rank
//! them in that order: the count decides how many jobs are late, the sum how late the rest are.
//! The sum alone would make five jobs a day late to save one five days late; the count alone
//! gives the search no gradient between improvements.
//!
//! A due date is soft, so it never makes a job unassignable. A job left unassigned is scored as
//! arriving when the plan ends, which no served job arrives after, so dropping a job never scores
//! better than serving it. The plan ends at the last shift end.
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
}

/// Creates a feature which minimizes the given due-date measure.
pub fn create_due_date_feature(
    name: &str,
    measure: DueDateMeasure,
    job_due_date_fn: JobDueDateFn,
    transport: Arc<dyn TransportCost>,
    fleet: &Fleet,
) -> GenericResult<Feature> {
    // open-ended shifts end at `Float::MAX`
    let shift_ends = fleet.actors.iter().map(|actor| actor.detail.time.end).filter(|&end| end < Float::MAX);

    FeatureBuilder::default()
        .with_name(name)
        .with_objective(DueDateObjective {
            measure,
            job_due_date_fn,
            transport,
            plan_end: shift_ends.max_by(Float::total_cmp),
        })
        .build()
}

struct DueDateObjective {
    measure: DueDateMeasure,
    job_due_date_fn: JobDueDateFn,
    transport: Arc<dyn TransportCost>,
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

                Some(self.score(arrival, due_date))
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
            .filter_map(|job| Some(self.score(plan_end?, (self.job_due_date_fn)(job)?)));

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

        self.score(activity_ctx.prev.schedule.departure + prev_to_target_duration, due_date)
    }
}

impl DueDateObjective {
    fn score(&self, arrival: Timestamp, due_date: Timestamp) -> Cost {
        match self.measure {
            DueDateMeasure::LateJobs => Cost::from(arrival > due_date),
            DueDateMeasure::Tardiness => (arrival - due_date).max(0.),
        }
    }
}
