//! A feature to model jobs which need several vehicles (a crew) to serve them together.
//!
//! A crew job is read as several copies of the same single job, one copy per crew member. Copies
//! share a [`JobCrew`] dimension and must:
//! - be served by different vehicles;
//! - start at the same time;
//! - be all assigned or all unassigned.
//!
//! Start times are kept in sync by locking. The first copy placed starts wherever it fits. The next copy is
//! evaluated with its time window narrowed to the times the first copy could start at without breaking its
//! own route: from its current start to its latest arrival (see [`get_crew_window`]). Once it is inserted,
//! all placed copies get their activity time window locked to its start, which can push the first copy
//! later. A locked window is an ordinary hard time window, so each route keeps its own schedule feasible
//! without looking at the others, and any further copy has to start at exactly that time.
//!
//! When a solution is accepted, a crew with some copies unassigned is unassigned entirely. Ruin removes
//! a crew entirely, so that it is re-planned together.

use super::*;
use crate::construction::enablers::{ScheduleActivityState, update_route_schedule};
use crate::models::solution::Activity;
use std::collections::{HashMap, HashSet};

/// Identifies a copy of a crew job.
#[derive(Clone, Debug)]
pub struct JobCrew {
    /// An id shared by all copies of the same crew job.
    pub id: String,
    /// The amount of copies, one per crew member.
    pub size: usize,
}

custom_dimension!(pub JobCrew typeof JobCrew);

/// Creates a feature which serves each crew job by several vehicles at the same time.
pub fn create_crew_feature(
    name: &str,
    transport: Arc<dyn TransportCost>,
    activity: Arc<dyn ActivityCost>,
    code: ViolationCode,
) -> Result<Feature, GenericError> {
    FeatureBuilder::default()
        .with_name(name)
        .with_constraint(CrewConstraint { code })
        .with_state(CrewState { transport, activity, code })
        .build()
}

/// Returns the times a crew job copy can start at when inserted into the given route: the times the
/// copies already placed on other routes can start at without breaking their own routes. Returns `None`
/// when the job is not a crew copy or no other copy is placed yet.
pub fn get_crew_window(
    solution_ctx: &SolutionContext,
    route_ctx: &RouteContext,
    single: &Single,
) -> Option<TimeWindow> {
    let crew = single.dimens.get_job_crew()?;
    let actor = &route_ctx.route().actor;

    solution_ctx
        .routes
        .iter()
        .filter(|other| other.route().actor != *actor)
        .flat_map(|other| {
            other
                .route()
                .tour
                .all_activities()
                .enumerate()
                .filter(|(_, activity)| is_other_copy(activity, single, crew))
                .map(move |(idx, activity)| {
                    let start = get_service_start(activity);
                    let latest = other.state().get_schedule_at(idx).map_or(start, |state| state.latest_arrival);

                    TimeWindow::new(start, latest.max(start))
                })
        })
        .reduce(|lhs, rhs| {
            let start = lhs.start.max(rhs.start);
            TimeWindow::new(start, lhs.end.min(rhs.end).max(start))
        })
}

/// Returns crew copies placed in the given routes whose other copies are not all there.
pub(crate) fn get_split_crew_jobs(routes: &[RouteContext]) -> HashSet<Job> {
    let mut copies: HashMap<&str, (usize, Vec<Job>)> = HashMap::default();

    routes.iter().flat_map(|route_ctx| route_ctx.route().tour.jobs()).for_each(|job| {
        if let Some(crew) = job.dimens().get_job_crew() {
            let entry = copies.entry(crew.id.as_str()).or_insert_with(|| (crew.size, Vec::default()));
            entry.1.push(job.clone());
        }
    });

    copies.into_values().filter(|(size, jobs)| jobs.len() < *size).flat_map(|(_, jobs)| jobs).collect()
}

fn is_other_copy(activity: &Activity, single: &Single, crew: &JobCrew) -> bool {
    activity.job.as_ref().is_some_and(|other| {
        !std::ptr::eq(other.as_ref(), single) && other.dimens.get_job_crew().is_some_and(|other| other.id == crew.id)
    })
}

fn get_service_start(activity: &Activity) -> Timestamp {
    activity.schedule.arrival.max(activity.place.time.start)
}

struct CrewConstraint {
    code: ViolationCode,
}

impl FeatureConstraint for CrewConstraint {
    fn evaluate(&self, move_ctx: &MoveContext<'_>) -> Option<ConstraintViolation> {
        match move_ctx {
            MoveContext::Route { solution_ctx, route_ctx, job } => {
                let single = job.as_single()?;
                let crew = single.dimens.get_job_crew()?;
                let has_other_copy = |route_ctx: &RouteContext| {
                    route_ctx.route().tour.all_activities().any(|activity| is_other_copy(activity, single, crew))
                };

                // Some search operators evaluate a copy of a route with jobs taken out, so also check the
                // route of the same vehicle as it is in the solution: swapping two copies of one crew between
                // their routes would otherwise escape the start time lock.
                let actor = &route_ctx.route().actor;
                let in_solution =
                    solution_ctx.routes.iter().find(|other| other.route().actor == *actor).is_some_and(has_other_copy);

                if in_solution || has_other_copy(route_ctx) { ConstraintViolation::fail(self.code) } else { None }
            }
            MoveContext::Activity { .. } => None,
        }
    }

    fn merge(&self, source: Job, candidate: Job) -> Result<Job, ViolationCode> {
        let has_crew = |job: &Job| job.dimens().get_job_crew().is_some();

        if has_crew(&source) || has_crew(&candidate) { Err(self.code) } else { Ok(source) }
    }
}

type PlacedCopies = Vec<(usize, Job, Timestamp)>;

struct CrewState {
    transport: Arc<dyn TransportCost>,
    activity: Arc<dyn ActivityCost>,
    code: ViolationCode,
}

impl CrewState {
    fn update_schedule(&self, route_ctx: &mut RouteContext) {
        update_route_schedule(route_ctx, self.activity.as_ref(), self.transport.as_ref());
    }
}

impl FeatureState for CrewState {
    fn accept_insertion(&self, solution_ctx: &mut SolutionContext, route_index: usize, job: &Job) {
        let Some(single) = job.as_single() else { return };
        let Some(crew) = single.dimens.get_job_crew() else { return };

        let has_other_copies = solution_ctx.routes.iter().enumerate().any(|(idx, route_ctx)| {
            idx != route_index && route_ctx.route().tour.all_activities().any(|a| is_other_copy(a, single, crew))
        });
        if !has_other_copies {
            return;
        }

        // the inserted copy starts within the times the other copies can start at, so they follow it
        let route_ctx = &mut solution_ctx.routes[route_index];
        self.update_schedule(route_ctx);
        let Some(start) = route_ctx
            .route()
            .tour
            .all_activities()
            .find(|activity| activity.job.as_ref().is_some_and(|other| Arc::ptr_eq(other, single)))
            .map(get_service_start)
        else {
            return;
        };

        // lock every placed copy, including the inserted one, to the shared start
        solution_ctx.routes.iter_mut().for_each(|route_ctx| {
            let indices = route_ctx
                .route()
                .tour
                .all_activities()
                .enumerate()
                .filter(|(_, activity)| {
                    let is_copy = activity
                        .job
                        .as_ref()
                        .and_then(|single| single.dimens.get_job_crew())
                        .is_some_and(|other| other.id == crew.id);
                    is_copy && (activity.place.time.start != start || activity.place.time.end != start)
                })
                .map(|(idx, _)| idx)
                .collect::<Vec<_>>();

            if indices.is_empty() {
                return;
            }

            let tour = &mut route_ctx.route_mut().tour;
            indices.into_iter().for_each(|idx| {
                tour.get_mut(idx).expect("activity index").place.time = TimeWindow::new(start, start);
            });

            self.update_schedule(route_ctx);
        });
    }

    fn accept_route_state(&self, _: &mut RouteContext) {}

    fn accept_solution_state(&self, solution_ctx: &mut SolutionContext) {
        // start times below are read from schedules, so they have to be up to date
        solution_ctx
            .routes
            .iter_mut()
            .filter(|route_ctx| route_ctx.is_stale())
            .for_each(|route_ctx| self.update_schedule(route_ctx));

        // crew id -> (crew size, placed copies as (route index, job, start))
        let mut placed: HashMap<String, (usize, PlacedCopies)> = HashMap::default();
        solution_ctx.routes.iter().enumerate().for_each(|(route_idx, route_ctx)| {
            route_ctx.route().tour.all_activities().for_each(|activity| {
                let Some(single) = activity.job.as_ref() else { return };
                let Some(crew) = single.dimens.get_job_crew() else { return };

                let entry = placed.entry(crew.id.clone()).or_insert_with(|| (crew.size, Vec::default()));
                entry.1.push((route_idx, Job::Single(single.clone()), get_service_start(activity)));
            });
        });

        if placed.is_empty() {
            return;
        }

        let crew_id = |job: &Job| job.dimens().get_job_crew().map(|crew| crew.id.clone());
        let failed = solution_ctx.unassigned.keys().filter_map(crew_id).collect::<HashSet<_>>();

        let mut to_unassigned = Vec::default();

        placed.into_iter().for_each(|(id, (size, copies))| {
            if failed.contains(&id) {
                // the rest of the crew could not be assigned: a crew is served all together or not at all
                to_unassigned.extend(copies);
            } else if copies.len() == size {
                // a safety net: an operator which bypasses the lock leaves the crew out of sync
                let start = copies[0].2;
                let routes = copies.iter().map(|(route_idx, _, _)| *route_idx).collect::<HashSet<_>>();
                let is_out_of_sync = copies.iter().any(|(_, _, other)| (other - start).abs() > 1E-6);

                if is_out_of_sync || routes.len() != size {
                    to_unassigned.extend(copies);
                }
            }
            // otherwise, the other copies wait for insertion, or are not part of this solution (a decomposed
            // part of it): keep as is, a copy inserted later starts together with the placed ones
        });

        if to_unassigned.is_empty() {
            return;
        }

        let mut touched = HashSet::new();
        to_unassigned.iter().for_each(|(route_idx, job, _)| {
            solution_ctx.routes[*route_idx].route_mut().tour.remove(job);
            touched.insert(*route_idx);
        });
        touched.into_iter().for_each(|route_idx| self.update_schedule(&mut solution_ctx.routes[route_idx]));

        solution_ctx
            .unassigned
            .extend(to_unassigned.into_iter().map(|(_, job, _)| (job, UnassignmentInfo::Simple(self.code))));
    }
}
