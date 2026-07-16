"""Query the persisted content recommender without refitting it."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommender_consumer import (  # noqa: E402
    load_recommender_artifacts,
    recommend_from_artifacts,
    resolve_catalog_track,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query persisted Spotify recommender artifacts read-only."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root")
    parser.add_argument("--track-id", help="Exact catalog track ID")
    parser.add_argument("--model-index", type=int, help="Exact catalog model index")
    parser.add_argument("--name", help="Exact catalog track name")
    parser.add_argument("--artists", help="Exact raw catalog artists value")
    parser.add_argument("--top-n", type=int, default=10, help="Recommendations to return")
    parser.add_argument("--output", type=Path, help="Optional output CSV path")
    parser.add_argument(
        "--no-validate-alignment",
        action="store_true",
        help="Skip full catalog-to-fitted-matrix alignment validation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        artifact_dir = root / "week7_outputs" / "model_artifacts"
        artifacts = load_recommender_artifacts(
            artifact_dir,
            validate_alignment=not args.no_validate_alignment,
        )
        query = resolve_catalog_track(
            artifacts.catalog,
            track_id=args.track_id,
            model_index=args.model_index,
            name=args.name,
            artists=args.artists,
        )
        recommendations = recommend_from_artifacts(
            artifacts,
            track_id=args.track_id,
            model_index=args.model_index,
            name=args.name,
            artists=args.artists,
            top_n=args.top_n,
        )
        print(
            f"Query: {query['name']} | {query['artists']} "
            f"(model_index={int(query['_model_index'])})"
        )
        print(recommendations.to_string(index=False))

        if args.output is not None:
            output_path = args.output.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            recommendations.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Output: {output_path}")
        return 0
    except (FileNotFoundError, LookupError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
