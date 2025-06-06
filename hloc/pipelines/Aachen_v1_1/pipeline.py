import argparse
from pathlib import Path
from pprint import pformat

from ... import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)


def run(args):
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images_upright/"
    sift_sfm = dataset / "3D-models/aachen_v_1_1"

    outputs = args.outputs  # where everything will be saved
    reference_sfm = outputs / f"sfm_{args.extractor}+{args.matcher}"  # the SfM model we will build
    sfm_pairs = (
        outputs / f"pairs-db-covis{args.num_covis}.txt"
    )  # top-k most covisible in SIFT model
    loc_pairs = (
        outputs / f"pairs-query-{args.global_extractor}{args.num_loc}.txt"
    )  # top-k retrieved by {args.global_extractor}
    results = (
        outputs / f"Aachen-v1.1_hloc_{args.extractor}+{args.matcher}_{args.global_extractor}{args.num_loc}.txt"
    )

    # list the standard configurations available
    logger.info("Configs for feature extractors:\n%s", pformat(extract_features.confs))
    logger.info("Configs for feature matchers:\n%s", pformat(match_features.confs))

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs[args.global_extractor]
    feature_conf = extract_features.confs[args.extractor]
    matcher_conf = match_features.confs[args.matcher]

    features = extract_features.main(feature_conf, images, outputs)

    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    localize_sfm.main(
        reference_sfm,
        # dataset / "queries/*_time_queries_with_intrinsics.txt",
        dataset / "queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )  # not required with SuperPoint+SuperGlue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/aachen_v1.1",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/aachen_v1.1",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=50,
        help="Number of image pairs for loc, default: %(default)s",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="alike",
        help="Local extractor config, default: %(default)s",
    )
    parser.add_argument(
        "--global_extractor",
        type=str,
        default="netvlad",
        help="Global extractor config, default: %(default)s",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="NN",
        help="Matcher  config, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
