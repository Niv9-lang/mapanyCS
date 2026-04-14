import argparse
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_MULTISCALE_VOXEL_SCALES = [1.0, 0.5, 0.25]
DEFAULT_MULTISCALE_ITERATIONS = [60, 40, 30]


@dataclass
class RegistrationSummary:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    coarse_fitness: float
    coarse_rmse: float
    refinement_method: str
    used_colored_icp: bool
    used_overlap_crop: bool
    stage_metrics: list[dict[str, float | bool | str]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register and merge overlapping PLY point clouds into a single point cloud."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the chunk PLY files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.ply",
        help="Glob pattern used to select the chunk PLY files.",
    )
    parser.add_argument(
        "--output-ply",
        required=True,
        help="Path of the merged output PLY file.",
    )
    parser.add_argument(
        "--report-json",
        help="Optional JSON report path containing registration metrics and transforms.",
    )
    parser.add_argument(
        "--registration-voxel-size",
        type=float,
        default=0.08,
        help="Voxel size used for coarse registration and ICP preprocessing.",
    )
    parser.add_argument(
        "--final-voxel-size",
        type=float,
        default=0.02,
        help="Voxel size used for the final merged cloud. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--normal-radius-mult",
        type=float,
        default=2.5,
        help="Normal estimation radius multiplier relative to registration voxel size.",
    )
    parser.add_argument(
        "--feature-radius-mult",
        type=float,
        default=5.0,
        help="FPFH feature radius multiplier relative to registration voxel size.",
    )
    parser.add_argument(
        "--coarse-distance-mult",
        type=float,
        default=1.8,
        help="Maximum correspondence distance multiplier for coarse registration.",
    )
    parser.add_argument(
        "--coarse-method",
        choices=["auto", "fgr", "ransac"],
        default="auto",
        help="Global registration method used to initialize alignment.",
    )
    parser.add_argument(
        "--fine-distance-mult",
        type=float,
        default=0.8,
        help="Maximum correspondence distance multiplier for fine ICP.",
    )
    parser.add_argument(
        "--colored-distance-mult",
        type=float,
        default=0.5,
        help="Maximum correspondence distance multiplier for colored ICP.",
    )
    parser.add_argument(
        "--target-window",
        type=int,
        default=2,
        help="How many already-registered chunks to use as the registration target.",
    )
    parser.add_argument(
        "--icp-iterations",
        type=int,
        default=80,
        help="Maximum iterations for point-to-plane ICP.",
    )
    parser.add_argument(
        "--colored-icp-iterations",
        type=int,
        default=40,
        help="Maximum iterations for colored ICP refinement.",
    )
    parser.add_argument(
        "--multiscale-voxel-scales",
        type=float,
        nargs="+",
        default=DEFAULT_MULTISCALE_VOXEL_SCALES,
        help="Relative voxel sizes used by the multiscale ICP refinement stages.",
    )
    parser.add_argument(
        "--multiscale-max-iterations",
        type=int,
        nargs="+",
        default=DEFAULT_MULTISCALE_ITERATIONS,
        help="ICP max iterations for each multiscale refinement stage.",
    )
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.12,
        help="Minimum ICP fitness required to accept a registration.",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=0.25,
        help="Maximum ICP inlier RMSE allowed to accept a registration.",
    )
    parser.add_argument(
        "--statistical-nb-neighbors",
        type=int,
        default=24,
        help="Number of neighbors used for statistical outlier removal.",
    )
    parser.add_argument(
        "--statistical-std-ratio",
        type=float,
        default=1.5,
        help="Std ratio used for statistical outlier removal.",
    )
    parser.add_argument(
        "--skip-statistical-filter",
        action="store_true",
        help="Disable statistical outlier removal during preprocessing and final cleanup.",
    )
    parser.add_argument(
        "--radius-nb-points",
        type=int,
        default=0,
        help="If > 0, apply radius outlier removal with this minimum neighbor count.",
    )
    parser.add_argument(
        "--radius-search-mult",
        type=float,
        default=3.0,
        help="Radius search multiplier relative to voxel size for radius outlier removal.",
    )
    parser.add_argument(
        "--disable-colored-icp",
        action="store_true",
        help="Disable colored ICP and use only point-to-plane ICP refinement.",
    )
    generalized_icp_group = parser.add_mutually_exclusive_group()
    generalized_icp_group.add_argument(
        "--enable-generalized-icp",
        action="store_true",
        help="Enable generalized ICP as an additional refinement candidate.",
    )
    generalized_icp_group.add_argument(
        "--disable-generalized-icp",
        action="store_true",
        help="Compatibility flag. Generalized ICP is already disabled by default.",
    )
    parser.add_argument(
        "--ransac-max-iteration",
        type=int,
        default=50000,
        help="Maximum iterations for RANSAC-based coarse registration.",
    )
    parser.add_argument(
        "--ransac-confidence",
        type=float,
        default=0.999,
        help="Confidence used by RANSAC-based coarse registration.",
    )
    parser.add_argument(
        "--max-clouds",
        type=int,
        default=0,
        help="Optional limit on the number of PLY files to merge. 0 means all matched files.",
    )
    parser.add_argument(
        "--retry-scale",
        type=float,
        default=1.5,
        help="Scale factor used for the automatic fallback registration pass.",
    )
    parser.add_argument(
        "--retry-target-window",
        type=int,
        default=0,
        help="Target window used by the fallback pass. 0 means use all previously registered chunks.",
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable the automatic fallback registration pass.",
    )
    overlap_crop_group = parser.add_mutually_exclusive_group()
    overlap_crop_group.add_argument(
        "--enable-overlap-crop",
        action="store_true",
        help="Enable overlap-aware cropping before ICP refinement.",
    )
    overlap_crop_group.add_argument(
        "--disable-overlap-crop",
        action="store_true",
        help="Compatibility flag. Overlap-aware cropping is already disabled by default.",
    )
    parser.add_argument(
        "--overlap-margin-mult",
        type=float,
        default=4.0,
        help="Bounding-box expansion multiplier relative to the stage voxel size when cropping overlap regions.",
    )
    parser.add_argument(
        "--min-overlap-points",
        type=int,
        default=200,
        help="Minimum points required in both cropped clouds before overlap-only refinement is used.",
    )
    return parser.parse_args()


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        raise RuntimeError(f"Empty point cloud: {path}")
    return point_cloud


def apply_statistical_filter(
    point_cloud: o3d.geometry.PointCloud,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    filtered, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered


def apply_radius_filter(
    point_cloud: o3d.geometry.PointCloud,
    nb_points: int,
    radius: float,
) -> o3d.geometry.PointCloud:
    filtered, _ = point_cloud.remove_radius_outlier(
        nb_points=nb_points,
        radius=radius,
    )
    return filtered


def estimate_normals(
    point_cloud: o3d.geometry.PointCloud,
    radius: float,
) -> None:
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )


def preprocess_registration_cloud(
    point_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius_mult: float,
    feature_radius_mult: float,
    use_statistical_filter: bool,
    radius_nb_points: int,
    radius_search_mult: float,
    nb_neighbors: int,
    std_ratio: float,
):
    cloud = copy.deepcopy(point_cloud)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if use_statistical_filter:
        cloud = apply_statistical_filter(cloud, nb_neighbors, std_ratio)
    if radius_nb_points > 0:
        cloud = apply_radius_filter(
            cloud,
            radius_nb_points,
            max(voxel_size * radius_search_mult, 1e-3),
        )

    normal_radius = max(voxel_size * normal_radius_mult, 1e-3)
    feature_radius = max(voxel_size * feature_radius_mult, normal_radius)
    estimate_normals(cloud, normal_radius)
    features = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    return cloud, features


def build_target_cloud(
    transformed_clouds: list[o3d.geometry.PointCloud],
    target_window: int,
) -> o3d.geometry.PointCloud:
    start_index = max(0, len(transformed_clouds) - target_window)
    target_cloud = o3d.geometry.PointCloud()
    for point_cloud in transformed_clouds[start_index:]:
        target_cloud += point_cloud
    return target_cloud


def expand_bounding_box(
    bounding_box: o3d.geometry.AxisAlignedBoundingBox,
    margin: float,
) -> o3d.geometry.AxisAlignedBoundingBox:
    return o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bounding_box.get_min_bound() - margin,
        max_bound=bounding_box.get_max_bound() + margin,
    )


def count_points(point_cloud: o3d.geometry.PointCloud) -> int:
    return int(np.asarray(point_cloud.points).shape[0])


def build_overlap_refinement_clouds(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    margin: float,
    min_overlap_points: int,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, bool]:
    transformed_source = copy.deepcopy(source_cloud)
    transformed_source.transform(transformation)

    source_box = expand_bounding_box(
        transformed_source.get_axis_aligned_bounding_box(),
        margin,
    )
    target_box = expand_bounding_box(
        target_cloud.get_axis_aligned_bounding_box(),
        margin,
    )
    cropped_source = transformed_source.crop(target_box)
    cropped_target = target_cloud.crop(source_box)
    if (
        count_points(cropped_source) < min_overlap_points
        or count_points(cropped_target) < min_overlap_points
    ):
        return transformed_source, copy.deepcopy(target_cloud), False
    return cropped_source, cropped_target, True


def run_coarse_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh,
    target_fpfh,
    max_distance: float,
    coarse_method: str,
    ransac_max_iteration: int,
    ransac_confidence: float,
):
    if coarse_method in ("auto", "fgr"):
        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=max_distance,
        )
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            option,
        )
        if coarse_method == "fgr" or result.fitness >= 0.05:
            return result

    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_distance),
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        max_iteration=ransac_max_iteration,
        confidence=ransac_confidence,
    )
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        max_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        checkers,
        criteria,
    )


def cloud_has_usable_colors(point_cloud: o3d.geometry.PointCloud) -> bool:
    if not point_cloud.has_colors():
        return False
    colors = np.asarray(point_cloud.colors)
    if colors.size == 0:
        return False
    return float(np.var(colors)) > 1e-8


def registration_quality(fitness: float, rmse: float) -> float:
    return float(fitness) - 0.25 * float(rmse)


def run_multiscale_registration(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    base_voxel_size: float,
    voxel_scales: list[float],
    max_iterations: list[int],
    normal_radius_mult: float,
    fine_distance_mult: float,
    colored_distance_mult: float,
    colored_icp_iterations: int,
    use_colored_icp: bool,
    use_generalized_icp: bool,
    use_overlap_crop: bool,
    overlap_margin_mult: float,
    min_overlap_points: int,
) -> RegistrationSummary:
    if len(voxel_scales) != len(max_iterations):
        raise ValueError(
            "--multiscale-voxel-scales and --multiscale-max-iterations must have the same length."
        )

    transformation = np.asarray(init_transform, dtype=np.float64)
    latest_fitness = 0.0
    latest_rmse = float("inf")
    refinement_method = "point_to_plane"
    used_color_refinement = False
    used_overlap_crop_once = False
    stage_metrics: list[dict[str, float | bool | str]] = []
    use_color_refinement = (
        use_colored_icp
        and cloud_has_usable_colors(source_cloud)
        and cloud_has_usable_colors(target_cloud)
    )

    for stage_index, (scale, iterations) in enumerate(zip(voxel_scales, max_iterations)):
        curr_voxel = max(base_voxel_size * float(scale), 1e-3)
        if use_overlap_crop:
            stage_source, stage_target, used_overlap_crop = build_overlap_refinement_clouds(
                source_cloud,
                target_cloud,
                transformation,
                margin=max(curr_voxel * overlap_margin_mult, 1e-3),
                min_overlap_points=min_overlap_points,
            )
            used_overlap_crop_once = used_overlap_crop_once or used_overlap_crop
        else:
            stage_source = copy.deepcopy(source_cloud)
            stage_source.transform(transformation)
            stage_target = copy.deepcopy(target_cloud)
            used_overlap_crop = False

        source_down = stage_source.voxel_down_sample(curr_voxel)
        target_down = stage_target.voxel_down_sample(curr_voxel)
        if count_points(source_down) < 10 or count_points(target_down) < 10:
            stage_source = copy.deepcopy(source_cloud)
            stage_source.transform(transformation)
            stage_target = copy.deepcopy(target_cloud)
            source_down = stage_source.voxel_down_sample(curr_voxel)
            target_down = stage_target.voxel_down_sample(curr_voxel)
            used_overlap_crop = False
        estimate_normals(source_down, max(curr_voxel * normal_radius_mult, 1e-3))
        estimate_normals(target_down, max(curr_voxel * normal_radius_mult, 1e-3))

        fine_distance = max(curr_voxel * fine_distance_mult, 1e-3)
        point_to_plane_result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            fine_distance,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
        )
        best_delta = point_to_plane_result.transformation
        best_fitness = float(point_to_plane_result.fitness)
        best_rmse = float(point_to_plane_result.inlier_rmse)
        best_method = "point_to_plane"

        if use_generalized_icp:
            generalized_result = o3d.pipelines.registration.registration_generalized_icp(
                source_down,
                target_down,
                fine_distance,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
            )
            if registration_quality(
                generalized_result.fitness,
                generalized_result.inlier_rmse,
            ) > registration_quality(best_fitness, best_rmse):
                best_delta = generalized_result.transformation
                best_fitness = float(generalized_result.fitness)
                best_rmse = float(generalized_result.inlier_rmse)
                best_method = "generalized_icp"

        transformation = best_delta @ transformation
        latest_fitness = best_fitness
        latest_rmse = best_rmse
        refinement_method = best_method
        stage_metrics.append(
            {
                "stage": int(stage_index),
                "voxel_size": float(curr_voxel),
                "fitness": best_fitness,
                "rmse": best_rmse,
                "method": best_method,
                "used_overlap_crop": bool(used_overlap_crop),
            }
        )

        is_last_stage = stage_index == len(voxel_scales) - 1
        if use_color_refinement and is_last_stage:
            colored_distance = max(curr_voxel * colored_distance_mult, 1e-3)
            try:
                transformed_source = copy.deepcopy(source_down)
                transformed_source.transform(best_delta)
                colored_result = o3d.pipelines.registration.registration_colored_icp(
                    transformed_source,
                    target_down,
                    colored_distance,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(
                        lambda_geometric=0.968
                    ),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=colored_icp_iterations
                    ),
                )
                colored_fitness = float(colored_result.fitness)
                colored_rmse = float(colored_result.inlier_rmse)
                stage_metrics.append(
                    {
                        "stage": int(stage_index),
                        "voxel_size": float(curr_voxel),
                        "fitness": colored_fitness,
                        "rmse": colored_rmse,
                        "method": "colored_icp_candidate",
                        "used_overlap_crop": bool(used_overlap_crop),
                    }
                )
                if registration_quality(colored_fitness, colored_rmse) > registration_quality(
                    latest_fitness,
                    latest_rmse,
                ):
                    transformation = colored_result.transformation @ transformation
                    latest_fitness = colored_fitness
                    latest_rmse = colored_rmse
                    refinement_method = "colored_icp"
                    used_color_refinement = True
            except RuntimeError as exc:
                print(f" colored_icp skipped: {exc}")

    if not stage_metrics:
        raise RuntimeError("Multiscale registration did not run any refinement stage.")
    return RegistrationSummary(
        transformation=transformation,
        fitness=latest_fitness,
        inlier_rmse=latest_rmse,
        coarse_fitness=0.0,
        coarse_rmse=0.0,
        refinement_method=refinement_method,
        used_colored_icp=used_color_refinement,
        used_overlap_crop=used_overlap_crop_once,
        stage_metrics=stage_metrics,
    )


def register_source_to_target(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius_mult: float,
    feature_radius_mult: float,
    coarse_distance_mult: float,
    coarse_method: str,
    fine_distance_mult: float,
    colored_distance_mult: float,
    colored_icp_iterations: int,
    multiscale_voxel_scales: list[float],
    multiscale_max_iterations: list[int],
    use_colored_icp: bool,
    use_generalized_icp: bool,
    use_statistical_filter: bool,
    radius_nb_points: int,
    radius_search_mult: float,
    nb_neighbors: int,
    std_ratio: float,
    ransac_max_iteration: int,
    ransac_confidence: float,
    use_overlap_crop: bool,
    overlap_margin_mult: float,
    min_overlap_points: int,
) -> RegistrationSummary:
    source_down, source_fpfh = preprocess_registration_cloud(
        source_cloud,
        voxel_size,
        normal_radius_mult,
        feature_radius_mult,
        use_statistical_filter,
        radius_nb_points,
        radius_search_mult,
        nb_neighbors,
        std_ratio,
    )
    target_down, target_fpfh = preprocess_registration_cloud(
        target_cloud,
        voxel_size,
        normal_radius_mult,
        feature_radius_mult,
        use_statistical_filter,
        radius_nb_points,
        radius_search_mult,
        nb_neighbors,
        std_ratio,
    )
    coarse_distance = max(voxel_size * coarse_distance_mult, 1e-3)
    coarse_result = run_coarse_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        coarse_distance,
        coarse_method,
        ransac_max_iteration,
        ransac_confidence,
    )
    fine_result = run_multiscale_registration(
        source_cloud,
        target_cloud,
        coarse_result.transformation,
        voxel_size,
        multiscale_voxel_scales,
        multiscale_max_iterations,
        normal_radius_mult,
        fine_distance_mult,
        colored_distance_mult,
        colored_icp_iterations,
        use_colored_icp,
        use_generalized_icp,
        use_overlap_crop,
        overlap_margin_mult,
        min_overlap_points,
    )
    fine_result.coarse_fitness = float(coarse_result.fitness)
    fine_result.coarse_rmse = float(coarse_result.inlier_rmse)
    return fine_result


def merge_point_clouds(
    clouds: list[o3d.geometry.PointCloud],
    final_voxel_size: float,
    use_statistical_filter: bool,
    radius_nb_points: int,
    radius_search_mult: float,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for point_cloud in clouds:
        merged += point_cloud

    if final_voxel_size > 0:
        merged = merged.voxel_down_sample(final_voxel_size)
    if use_statistical_filter:
        merged = apply_statistical_filter(merged, nb_neighbors, std_ratio)
    if radius_nb_points > 0:
        merged = apply_radius_filter(
            merged,
            radius_nb_points,
            max(final_voxel_size * radius_search_mult, 1e-3),
        )
    return merged


def transformation_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_ply = Path(args.output_ply)
    report_json = Path(args.report_json) if args.report_json else None
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ply_paths = sorted(input_dir.glob(args.pattern))
    if args.max_clouds > 0:
        ply_paths = ply_paths[: args.max_clouds]
    if len(ply_paths) < 2:
        raise ValueError(f"Need at least 2 PLY files to merge, found {len(ply_paths)}")

    use_statistical_filter = not args.skip_statistical_filter
    use_colored_icp = not args.disable_colored_icp
    use_generalized_icp = bool(args.enable_generalized_icp) and not bool(
        args.disable_generalized_icp
    )
    use_overlap_crop = bool(args.enable_overlap_crop) and not bool(
        args.disable_overlap_crop
    )
    if len(args.multiscale_voxel_scales) != len(args.multiscale_max_iterations):
        raise ValueError(
            "--multiscale-voxel-scales and --multiscale-max-iterations must have the same length."
        )
    if args.radius_nb_points < 0:
        raise ValueError("--radius-nb-points must be >= 0")
    if args.min_overlap_points < 0:
        raise ValueError("--min-overlap-points must be >= 0")

    print("Merging the following chunk files:")
    for path in ply_paths:
        print(f" - {path.name}")

    original_clouds = [load_point_cloud(path) for path in ply_paths]
    transformed_clouds = [copy.deepcopy(original_clouds[0])]
    metrics = [
        {
            "name": ply_paths[0].name,
            "transformation": transformation_to_list(np.eye(4)),
            "fitness": 1.0,
            "rmse": 0.0,
            "accepted": True,
            "notes": "anchor cloud",
        }
    ]

    for index in range(1, len(original_clouds)):
        source_name = ply_paths[index].name
        print(f"\nRegistering {source_name} ({index + 1}/{len(original_clouds)})")
        target_cloud = build_target_cloud(transformed_clouds, args.target_window)

        fine_result = register_source_to_target(
            original_clouds[index],
            target_cloud,
            args.registration_voxel_size,
            args.normal_radius_mult,
            args.feature_radius_mult,
            args.coarse_distance_mult,
            args.coarse_method,
            args.fine_distance_mult,
            args.colored_distance_mult,
            args.colored_icp_iterations,
            args.multiscale_voxel_scales,
            args.multiscale_max_iterations,
            use_colored_icp,
            use_generalized_icp,
            use_statistical_filter,
            args.radius_nb_points,
            args.radius_search_mult,
            args.statistical_nb_neighbors,
            args.statistical_std_ratio,
            args.ransac_max_iteration,
            args.ransac_confidence,
            use_overlap_crop,
            args.overlap_margin_mult,
            args.min_overlap_points,
        )

        accepted = (
            fine_result.fitness >= args.min_fitness
            and fine_result.inlier_rmse <= args.max_rmse
        )
        used_fallback = False
        print(
            f" coarse_fitness={fine_result.coarse_fitness:.4f}"
            f" coarse_rmse={fine_result.coarse_rmse:.4f}"
            f" refine={fine_result.refinement_method}"
            f" overlap_crop={fine_result.used_overlap_crop}"
        )
        print(
            f" fitness={fine_result.fitness:.4f}"
            f" rmse={fine_result.inlier_rmse:.4f}"
            f" accepted={accepted}"
        )
        if not accepted and not args.disable_fallback:
            fallback_target_window = (
                args.retry_target_window
                if args.retry_target_window > 0
                else len(transformed_clouds)
            )
            fallback_voxel = args.registration_voxel_size * args.retry_scale
            print(
                " primary registration rejected; retrying with "
                f"voxel={fallback_voxel:.4f} target_window={fallback_target_window}"
            )
            fallback_target_cloud = build_target_cloud(
                transformed_clouds,
                fallback_target_window,
            )
            fallback_result = register_source_to_target(
                original_clouds[index],
                fallback_target_cloud,
                fallback_voxel,
                args.normal_radius_mult,
                args.feature_radius_mult,
                args.coarse_distance_mult,
                args.coarse_method,
                args.fine_distance_mult * args.retry_scale,
                args.colored_distance_mult * args.retry_scale,
                args.colored_icp_iterations,
                args.multiscale_voxel_scales,
                args.multiscale_max_iterations,
                use_colored_icp,
                use_generalized_icp,
                use_statistical_filter,
                args.radius_nb_points,
                args.radius_search_mult,
                args.statistical_nb_neighbors,
                args.statistical_std_ratio,
                args.ransac_max_iteration,
                args.ransac_confidence,
                use_overlap_crop,
                args.overlap_margin_mult,
                args.min_overlap_points,
            )
            fallback_accepted = (
                fallback_result.fitness >= args.min_fitness
                and fallback_result.inlier_rmse <= args.max_rmse
            )
            print(
                f" fallback_coarse_fitness={fallback_result.coarse_fitness:.4f}"
                f" fallback_coarse_rmse={fallback_result.coarse_rmse:.4f}"
                f" refine={fallback_result.refinement_method}"
                f" overlap_crop={fallback_result.used_overlap_crop}"
            )
            print(
                f" fallback_fitness={fallback_result.fitness:.4f}"
                f" fallback_rmse={fallback_result.inlier_rmse:.4f}"
                f" accepted={fallback_accepted}"
            )
            if fallback_accepted:
                fine_result = fallback_result
                accepted = True
                used_fallback = True

        if not accepted:
            raise RuntimeError(
                f"Registration rejected for {source_name}: "
                f"fitness={fine_result.fitness:.4f}, rmse={fine_result.inlier_rmse:.4f}. "
                f"Adjust --registration-voxel-size, --target-window, --min-fitness, or --max-rmse."
            )

        transformed_source = copy.deepcopy(original_clouds[index])
        transformed_source.transform(fine_result.transformation)
        transformed_clouds.append(transformed_source)
        metrics.append(
            {
                "name": source_name,
                "transformation": transformation_to_list(fine_result.transformation),
                "fitness": float(fine_result.fitness),
                "rmse": float(fine_result.inlier_rmse),
                "coarse_fitness": float(fine_result.coarse_fitness),
                "coarse_rmse": float(fine_result.coarse_rmse),
                "accepted": True,
                "target_window": int(args.target_window),
                "used_fallback": bool(used_fallback),
                "refinement_method": fine_result.refinement_method,
                "used_colored_icp": bool(fine_result.used_colored_icp),
                "used_overlap_crop": bool(fine_result.used_overlap_crop),
                "stage_metrics": fine_result.stage_metrics,
            }
        )

    print("\nMerging registered point clouds...")
    merged_cloud = merge_point_clouds(
        transformed_clouds,
        args.final_voxel_size,
        use_statistical_filter,
        args.radius_nb_points,
        args.radius_search_mult,
        args.statistical_nb_neighbors,
        args.statistical_std_ratio,
    )
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_point_cloud(str(output_ply), merged_cloud)
    if not success:
        raise RuntimeError(f"Failed to write merged point cloud to {output_ply}")
    print(f"Saved merged point cloud: {output_ply}")

    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "input_dir": str(input_dir),
            "pattern": args.pattern,
            "output_ply": str(output_ply),
            "registration_voxel_size": args.registration_voxel_size,
            "final_voxel_size": args.final_voxel_size,
            "target_window": args.target_window,
            "min_fitness": args.min_fitness,
            "max_rmse": args.max_rmse,
            "coarse_method": args.coarse_method,
            "use_colored_icp": use_colored_icp,
            "use_generalized_icp": use_generalized_icp,
            "use_overlap_crop": use_overlap_crop,
            "overlap_margin_mult": args.overlap_margin_mult,
            "min_overlap_points": args.min_overlap_points,
            "use_statistical_filter": use_statistical_filter,
            "radius_nb_points": args.radius_nb_points,
            "radius_search_mult": args.radius_search_mult,
            "multiscale_voxel_scales": args.multiscale_voxel_scales,
            "multiscale_max_iterations": args.multiscale_max_iterations,
            "files": metrics,
            "merged_points": int(np.asarray(merged_cloud.points).shape[0]),
        }
        report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved merge report: {report_json}")


if __name__ == "__main__":
    main()
