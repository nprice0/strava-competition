from models import SegmentResult
from strava_api import get_segment_efforts

def process_segments(segments, runners):
    results = {}
    for segment in segments:
        results[segment.name] = {}
        for runner in runners:
            efforts = get_segment_efforts(runner, segment.id, segment.start_date, segment.end_date)
            if not efforts:
                continue
            attempts = len(efforts)
            fastest = min(efforts, key=lambda e: e["elapsed_time"])
            seg_result = SegmentResult(
                runner=runner.name,
                team=runner.segment_team,
                segment=segment.name,
                attempts=attempts,
                fastest_time=fastest["elapsed_time"],
                fastest_date=fastest["start_date_local"]
            )
            if runner.team not in results[segment.name]:
                results[segment.name][runner.team] = []
            results[segment.name][runner.team].append(seg_result)

        # Sort each team's results
        for team in results[segment.name]:
            results[segment.name][team].sort(key=lambda r: r.fastest_time)
    return results
