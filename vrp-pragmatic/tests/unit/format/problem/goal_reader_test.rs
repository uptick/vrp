use super::*;
use vrp_core::construction::clustering::vicinity::ClusterInfo;
use vrp_core::models::common::Dimensions;
use vrp_core::models::solution::Commute;

fn create_job(value: Option<Float>) -> CoreJob {
    let mut dimens = Dimensions::default();
    if let Some(value) = value {
        dimens.set_job_value(value);
    }

    CoreJob::Single(Arc::new(Single { places: vec![], dimens }))
}

fn create_cluster(own_value: Option<Float>, member_values: Vec<Option<Float>>) -> CoreJob {
    let CoreJob::Single(cluster) = create_job(own_value) else { unreachable!() };
    let mut dimens = cluster.dimens.clone();
    dimens.set_cluster_info(
        member_values
            .into_iter()
            .map(|value| ClusterInfo {
                job: create_job(value),
                service_time: 0.,
                place_idx: 0,
                commute: Commute::default(),
            })
            .collect(),
    );

    CoreJob::Single(Arc::new(Single { places: vec![], dimens }))
}

parameterized_test! {can_get_job_value_with_clusters, (job, expected), {
    assert_eq!(get_job_value_with_clusters(&job), expected);
}}

can_get_job_value_with_clusters! {
    case01_valued_job: (create_job(Some(10.)), 10.),
    case02_job_without_value: (create_job(None), 0.),
    // without `maximize-value`, a cluster keeps its first job's value as its own
    case03_cluster_without_merged_value: (create_cluster(Some(10.), vec![Some(10.), Some(5.), None]), 15.),
    // with it, the merged value is already summed, and must not be counted twice
    case04_cluster_with_merged_value: (create_cluster(Some(15.), vec![Some(10.), Some(5.)]), 15.),
}
