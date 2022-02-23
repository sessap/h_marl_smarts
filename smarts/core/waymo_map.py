# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import logging
import math
import time
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import numpy as np
import rtree
from cached_property import cached_property
from shapely.geometry import Polygon
from waymo_open_dataset.protos import scenario_pb2

from smarts.sstudio.types import MapSpec

from .coordinates import (
    BoundingBox,
    Heading,
    Point,
    Pose,
    RefLinePoint,
    distance_point_to_polygon,
    offset_along_shape,
    position_at_shape_offset,
)
from .lanepoints import LanePoints, LinkedLanePoint
from .road_map import RoadMap, Waypoint
from .utils.file import read_tfrecord_file
from .utils.math import ray_boundary_intersect, inplace_unwrap, radians_to_vec, vec_2d


class WaymoMap(RoadMap):
    """A map associated with a Waymo dataset"""

    DEFAULT_LANE_SPEED = 16.67  # in m/s
    DEFAULT_LANE_WIDTH = 3.5

    def __init__(self, map_spec: MapSpec, waymo_scenario):
        self._log = logging.getLogger(self.__class__.__name__)
        self._map_spec = map_spec
        self._surfaces: Dict[str, WaymoMap.Surface] = dict()
        self._lanes: Dict[str, WaymoMap.Lane] = dict()
        self._roads: Dict[str, WaymoMap.Road] = dict()
        self._waymo_features: Dict[int, Any] = dict()
        self._default_lane_width = WaymoMap.DEFAULT_LANE_WIDTH
        self._lane_rtree = None
        self._no_roads = False  # for debugging purposes
        self._load_from_scenario(waymo_scenario)

        self._waypoints_cache = WaymoMap._WaypointsCache()
        self._lanepoints = None
        if map_spec.lanepoint_spacing is not None:
            assert map_spec.lanepoint_spacing > 0
            # XXX: this should be last here since LanePoints() calls road_network methods immediately
            self._lanepoints = LanePoints.from_waymo(
                self, spacing=map_spec.lanepoint_spacing
            )

    @staticmethod
    def _find_lane_segments(waymo_lane_feats, side: str) -> Dict[int, bool]:
        split_pts = {}
        boundaries = getattr(waymo_lane_feats, f"{side}_boundaries")
        for bd in range(1, len(boundaries)):
            bdry = boundaries[bd]
            if bdry.boundary_type != boundaries[bd - 1].boundary_type:
                split_pts[bdry.lane_start_index] = False
        # overwrite any "soft" boundary splits with "hard" neighbor splits
        for nb in getattr(waymo_lane_feats, f"{side}_neighbors"):
            split_pts[nb.self_start_index] = True
            split_pts[nb.self_end_index + 1] = True
        return split_pts

    def _waymo_pb_to_dict(self, waymo_lane_feats) -> Dict[str, Any]:
        # we can't mutate the waymo protobuf objects, nor do they have a __dict__,
        # so we just keep the fields we're going to use...
        attribs = [
            "type",
            "interpolating",
            "entry_lanes",
            "exit_lanes",
            "polyline",
            "speed_limit_mph",
            "left_boundaries",
            "right_boundaries",
            "left_neighbors",
            "right_neighbors",
        ]
        return {attr: getattr(waymo_lane_feats, attr) for attr in attribs}

    class _LaneSegment:
        def __init__(
            self,
            feat_id: str,
            lane_dict: Dict[str, Any] = {},
            start_pt: int = 0,
            end_pt: int = -1,
            sub_segs: Sequence["WaymoMap._LaneSegment"] = [],
        ):
            self.feat_id = feat_id
            self.start_pt = start_pt
            if end_pt < 0:
                self.end_pt = len(lane_dict.get("polyline", []))
            else:
                self.end_pt = end_pt
            self.lane_dict = lane_dict
            self.sub_segs = sub_segs
            if sub_segs:
                self.lane_dict["sublanes"] = [ss.seg_id for ss in sub_segs]
                for ss in sub_segs:
                    ss["composite"] = self.seg_id
            self._shift_and_clip("left")
            self._shift_and_clip("right")

        @cached_property
        def seg_id(self) -> str:
            seg_id = f"{self.feat_id}"
            if self.start_pt > 0:
                # try to keep seg_ids the same as lane ids when not doing segmentation
                seg_id += f"_{self.start_pt}"
            for ss in range(1, len(self.sub_segs)):
                seg_id += f"_{self.sub_segs[ss].end_pt}"
            return seg_id

        def _shift_and_clip(self, side: str):
            offset = self.start_pt
            max_ind = len(self.lane_dict.get("polyline", []))
            if offset == 0 and max_ind == self.end_pt:
                return

            new_neighbors = []
            neighbors = self.lane_dict.get(f"{side}_neighbors", [])
            for nb in neighbors:
                if (
                    nb.self_start_index < offset
                    or nb.self_start_index >= max_ind + offset - 1
                ):
                    continue
                nb.self_start_index -= offset
                nb.self_end_index -= offset
                for nbbd in nb.boundaries:
                    nbbd.lane_start_index -= offset
                    nbbd.lane_end_index -= offset
                # TAI:  could try to get nb's lane_dict here?  what if it's a composite?
                nb_seg = WaymoMap._LaneSegment(
                    str(nb.feature_id),
                    {},
                    nb.neighbor_start_index,
                    nb.neighbor_end_index,
                )
                self.lane_dict.setdefault(f"_{side}_nb_segs", {})[
                    nb.feature_id
                ] = nb_seg.seg_id
                nb.neighbor_end_index -= nb.neighbor_start_index
                nb.neighbor_start_index = 0
                new_neighbors.append(nb)
            # assert len(new_neighbors) <= 1, f"{new_neighbors}"
            self.lane_dict[f"{side}_neighbors"] = new_neighbors

            new_boundaries = []
            boundaries = self.lane_dict.get(f"{side}_boundaries", [])
            for bd in boundaries:
                if (
                    bd.lane_start_index < offset
                    or bd.lane_start_index >= max_ind + offset - 1
                ):
                    continue
                bd.lane_start_index -= offset
                bd.lane_end_index -= offset
                new_boundaries.append(bd)
            self.lane_dict[f"{side}_boundaries"] = new_boundaries

        def create_new_segment(
            self,
            split_pt: int,
            prev_seg: "WaymoMap._LaneSegment",
            sub_segs: Optional[Sequence] = None,
        ) -> "WaymoMap._LaneSegment":
            new_lane_dict = deepcopy(self.lane_dict)
            # XXX: Here we'd like to do the following, but python protobuf seem to have a bug with slicing:
            #    new_lane_dict["polyline"] = self.lane_dict["polyline"][prev_seg.end_pt:split_pt]
            # XXX: So instead we use a list comprehension:
            seg_start = prev_seg.end_pt if prev_seg else 0
            new_lane_dict["polyline"] = [
                Point(p.x, p.y)
                for i, p in enumerate(self.lane_dict["polyline"])
                if seg_start <= i <= split_pt
            ]
            if prev_seg and prev_seg.seg_id:
                new_lane_dict["entry_lanes"] = [prev_seg.seg_id]
            new_seg = WaymoMap._LaneSegment(
                self.feat_id, new_lane_dict, seg_start, split_pt, sub_segs
            )
            if prev_seg and prev_seg.lane_dict:
                prev_seg.lane_dict["exit_lanes"] = [new_seg.seg_id]
            return new_seg

    def _create_lanes_and_roads(self, waymo_lanes: List[Tuple[str, Any]]):
        # first segment lanes based on their boundaries and neighbors
        waymo_lanedicts = {}
        for lane_id, lane_feat in waymo_lanes:
            split_pts = None
            if not self._no_roads:
                rt_split_pts = WaymoMap._find_lane_segments(lane_feat, "right")
                lft_split_pts = WaymoMap._find_lane_segments(lane_feat, "left")
                split_pts = {
                    k: rt_split_pts.get(k, False) or lft_split_pts.get(k, False)
                    for k in rt_split_pts.keys() | lft_split_pts.keys()
                }
            waymo_lane_dict = self._waymo_pb_to_dict(lane_feat)
            waymo_lane_dict["_feature_id"] = lane_id
            orig_seg = WaymoMap._LaneSegment(lane_id, waymo_lane_dict)
            if not split_pts:
                waymo_lanedicts[orig_seg.seg_id] = waymo_lane_dict
                continue
            prev_split_pt = 0
            prev_soft_seg = None
            prev_hard_seg = None
            softs_since_hard = []
            maxind = orig_seg.end_pt
            split_pts[maxind] = True
            for split_pt in sorted(split_pts.keys()):
                if split_pt >= maxind and prev_split_pt >= maxind - 1:
                    # avoid singleton-point polyline segments
                    break
                prev_split_pt = split_pt
                if split_pts[split_pt]:
                    if softs_since_hard:
                        prev_soft_seg = orig_seg.create_new_segment(
                            split_pt, prev_soft_seg
                        )
                        waymo_lanedicts[prev_soft_seg.seg_id] = prev_soft_seg.lane_dict
                        softs_since_hard.append(prev_soft_seg)
                    prev_hard_seg = orig_seg.create_new_segment(
                        split_pt, prev_hard_seg, softs_since_hard
                    )
                    waymo_lanedicts[prev_hard_seg.seg_id] = prev_hard_seg.lane_dict
                    softs_since_hard = []
                else:
                    prev_soft_seg = orig_seg._create_lane_segment(split, prev_soft_seg)
                    waymo_lanedicts[prev_soft_seg.seg_id] = prev_soft_seg.lane_dict
                    softs_since_hard.append(prev_soft_seg)

        # then create a road segments out of adjacent lane segments
        for lane_id, lane_dict in waymo_lanedicts.items():
            # TODO: consider case of soft segments in right lane but not left
            if not self._no_roads and lane_dict["right_neighbors"]:
                continue
            assert lane_id not in self._lanes
            lane = WaymoMap.Lane(self, lane_id, lane_dict)
            self._lanes[lane_id] = lane
            self._surfaces[lane_id] = lane
            if self._no_roads:
                continue
            road_lanes = [lane]
            self._add_adj_lanes_to_road(lane_dict, road_lanes, waymo_lanedicts)
            road = WaymoMap.Road(self, road_lanes)
            self._roads[road.road_id] = road
            self._surfaces[road.road_id] = road

    def _adj_seg_id(
        self,
        rn_feature_id: str,
        ln_feature_id: int,
        lanedicts: Dict[str, Dict[str, Any]],
    ) -> str:
        ln_dict = lanedicts[str(ln_feature_id)]
        for rn in ln_dict["right_neighbors"]:
            if rn.feature_id == rn_feature_id:
                assert rn.neighbor_start_index == 0, f"{rn}"
                assert len(rn.boundaries) <= 1, f"{rn}"
                nb_seg = WaymoMap._LaneSegment(
                    str(ln_feature_id), {}, rn.self_start_index, rn.self_end_index
                )
                return nb_seg.seg_id
        return None

    def _add_adj_lanes_to_road(
        self,
        lane_dict: Dict[str, Any],
        road_lane: List[RoadMap.Lane],
        lanedicts: Dict[str, Dict[str, Any]],
    ):
        # XXX: there _should_ only be one left-neighbor at a time now,
        # xXX: but we hit cases where there's more than one overlapping on a segment,
        # XXX: so we loop here just in case...
        lns_to_do = []
        for ln in lane_dict["left_neighbors"]:
            ln_seg_id = lane_dict.get("_left_nb_segs", {}).get(ln.feature_id)
            if not ln_seg_id:
                ln_seg_id = self._adj_seg_id(
                    lane_dict["_feature_id"], ln.feature_id, lanedicts
                )
            assert ln_seg_id, f"{ln.feature_id}"
            ln_lane_dict = lanedicts[ln_seg_id]
            lns_to_do.append(ln_lane_dict)
            lane = self._lanes.setdefault(
                ln_seg_id, WaymoMap.Lane(self, ln_seg_id, ln_lane_dict)
            )
            self._surfaces[ln_seg_id] = lane
            road_lane.append(lane)
        for adj_lane_dict in lns_to_do:
            self._add_adj_lanes_to_road(adj_lane_dict, road_lane, lanedicts)

    def _load_from_scenario(self, waymo_scenario):
        start = time.time()
        waymo_lanes: List[Tuple[str, Any]] = []
        for i in range(len(waymo_scenario.map_features)):
            map_feature = waymo_scenario.map_features[i]
            key = map_feature.WhichOneof("feature_data")
            if key is not None:
                self._waymo_features[map_feature.id] = getattr(map_feature, key)
                if key == "lane":
                    waymo_lanes.append((str(map_feature.id), getattr(map_feature, key)))

        # First pass -- create lane objects, initial geometry computations, infer roads
        self._create_lanes_and_roads(waymo_lanes)

        # Second pass -- try to fill in missing points from connected lanes
        should_run = True
        i = 0
        while should_run and i < 10:
            should_run = False
            i += 1
            for lane_id, lane in self._lanes.items():
                if lane._fill_in_connected_points():
                    should_run = True

        # Third pass -- create polygons
        for lane_id, lane in self._lanes.items():
            lane._create_polygon()

        self._waymo_scenario_id = waymo_scenario.scenario_id

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Loading Waymo map took: {elapsed} ms")

    @staticmethod
    def _parse_source_to_scenario(source: str):
        # Read the dataset file and get the specified scenario
        dataset_path = source.split("#")[0]
        scenario_id = source.split("#")[1]
        dataset_records = read_tfrecord_file(dataset_path)
        for record in dataset_records:
            parsed_scenario = scenario_pb2.Scenario()
            parsed_scenario.ParseFromString(bytearray(record))
            if parsed_scenario.scenario_id == scenario_id:
                return parsed_scenario
        errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
        raise ValueError(errmsg)

    @classmethod
    def from_spec(cls, map_spec: MapSpec):
        waymo_scenario = cls._parse_source_to_scenario(map_spec.source)
        assert waymo_scenario
        return cls(map_spec, waymo_scenario)

    @property
    def source(self) -> str:
        return self._map_spec.source

    @staticmethod
    def _spec_lane_width(map_spec: MapSpec) -> float:
        return (
            map_spec.default_lane_width
            if map_spec.default_lane_width is not None
            else WaymoMap.DEFAULT_LANE_WIDTH
        )

    def is_same_map(self, map_spec: MapSpec) -> bool:
        waymo_scenario = WaymoMap._parse_source_to_scenario(map_spec)
        return (
            waymo_scenario.scenario_id == self._waymo_scenario_id
            and map_spec.lanepoint_spacing == self._map_spec.lanepoint_spacing
            and (
                map_spec.default_lane_width == self._map_spec.default_lane_width
                or WaymoMap._spec_lane_width(map_spec)
                == WaymoMap._spec_lane_width(self._map_spec)
            )
            and map_spec.shift_to_origin == self._map_spec.shift_to_origin
        )

    @cached_property
    def bounding_box(self) -> BoundingBox:
        """Get the minimal axis aligned bounding box that contains all map geometry."""
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for road_id in self._roads:
            road = self._roads[road_id]
            x_mins.append(road.bounding_box.min_pt.x)
            y_mins.append(road.bounding_box.min_pt.y)
            x_maxs.append(road.bounding_box.max_pt.x)
            y_maxs.append(road.bounding_box.max_pt.y)

        return BoundingBox(
            min_pt=Point(x=min(x_mins), y=min(y_mins)),
            max_pt=Point(x=max(x_maxs), y=max(y_maxs)),
        )

    @property
    def scale_factor(self) -> float:
        return 1.0  # TODO

    def to_glb(self, at_path: str):
        pass  # TODO (or not!)

    class Surface(RoadMap.Surface):
        def __init__(self, surface_id: str, road_map):
            self._surface_id = surface_id
            self._map = road_map

        @property
        def surface_id(self) -> str:
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            # XXX: this may be over-riden below
            return True

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        return self._surfaces.get(surface_id)

    class Lane(RoadMap.Lane, Surface):
        def __init__(self, road_map, lane_id: str, lane_dict: Dict[str, Any]):
            super().__init__(lane_id, road_map)
            self._map = road_map
            self._lane_id = lane_id
            self._road = None  # set when lane is added to a Road
            self._index = None  # set when lane is added to a Road
            self._lane_dict = lane_dict
            self._lane_pts = [np.array([p.x, p.y]) for p in lane_dict["polyline"]]
            self._centerline_pts = [Point(p.x, p.y) for p in lane_dict["polyline"]]
            self._lane_width = None
            self._bounding_box = None
            self._speed_limit = (
                lane_dict.get("speed_limit_mph", WaymoMap.DEFAULT_LANE_SPEED / 0.44704)
                * 0.44704
            )
            self._is_composite = bool(lane_dict.get("sublanes", None))

            # Geometry
            self._n_pts = len(self._lane_pts)
            self._left_widths = [0] * self._n_pts
            self._right_widths = [0] * self._n_pts
            self._lane_polygon = None
            self._raycast_boundaries()

        def _calculate_normals(self) -> Sequence[np.ndarray]:
            normals = [None] * self._n_pts
            for i in range(self._n_pts):
                p = self._lane_pts[i]
                if i < self._n_pts - 1:
                    dp = self._lane_pts[i + 1] - p
                else:
                    dp = p - self._lane_pts[i - 1]

                dp /= np.linalg.norm(dp)
                angle = math.pi / 2
                normal = np.array(
                    [
                        math.cos(angle) * dp[0] - math.sin(angle) * dp[1],
                        math.sin(angle) * dp[0] + math.cos(angle) * dp[1],
                    ]
                )
                normals[i] = normal
            return normals

        def _raycast_boundaries(self, ray_dist=20.0):
            self._normals = self._calculate_normals()
            for i in range(self._n_pts):
                ray_start = self._lane_pts[i]
                normal = self._normals[i]

                if self._lane_dict["left_neighbors"]:
                    sign = 1.0
                    ray_end = ray_start + sign * ray_dist * normal
                    for n in self._lane_dict["left_neighbors"]:
                        if not (n.self_start_index <= i <= n.self_end_index):
                            continue
                        feature = self._map._waymo_features[n.feature_id]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._left_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

                if self._lane_dict["right_neighbors"]:
                    sign = -1.0
                    ray_end = ray_start + sign * ray_dist * normal
                    for n in self._lane_dict["right_neighbors"]:
                        if not (n.self_start_index <= i <= n.self_end_index):
                            continue
                        feature = self._map._waymo_features[n.feature_id]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._right_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

            # Sometimes lanes that overlap are considered neighbors, so filter those out
            width_threshold = 0.5
            if (
                max(self._left_widths) > width_threshold
                or max(self._right_widths) > width_threshold
            ):
                return

            for i in [0, self._n_pts - 1]:
                ray_start = self._lane_pts[i]
                normal = self._normals[i]

                if self._lane_dict["left_boundaries"]:
                    sign = 1.0
                    ray_end = ray_start + sign * ray_dist * normal
                    for boundary in self._lane_dict["left_boundaries"]:
                        if not (
                            boundary.lane_start_index <= i <= boundary.lane_end_index
                        ):
                            continue
                        feature = self._map._waymo_features[
                            boundary.boundary_feature_id
                        ]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._left_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

                if self._lane_dict["right_boundaries"]:
                    sign = -1.0
                    ray_end = ray_start + sign * ray_dist * normal
                    for boundary in self._lane_dict["right_boundaries"]:
                        if not (
                            boundary.lane_start_index <= i <= boundary.lane_end_index
                        ):
                            continue
                        feature = self._map._waymo_features[
                            boundary.boundary_feature_id
                        ]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._right_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

        def _fill_in_connected_points(self) -> bool:
            filled_in_point = False
            if (
                self._left_widths[0] == 0
                and self._right_widths[0] == 0
                and self._left_widths[-1] == 0
                and self._right_widths[-1] == 0
            ):
                for n in self.incoming_lanes:
                    if n and n._left_widths[-1] != 0:
                        self._left_widths[0] = n._left_widths[-1]
                        filled_in_point = True
                        break
                for n in self.incoming_lanes:
                    if n and n._right_widths[-1] != 0:
                        self._right_widths[0] = n._right_widths[-1]
                        filled_in_point = True
                        break
                for n in self.outgoing_lanes:
                    if n and n._left_widths[0] != 0:
                        self._left_widths[-1] = n._left_widths[0]
                        filled_in_point = True
                        break
                for n in self.outgoing_lanes:
                    if n and n._right_widths[0] != 0:
                        self._right_widths[-1] = n._right_widths[0]
                        filled_in_point = True
                        break
            return filled_in_point

        def _create_polygon(self):
            max_width = max(
                self._left_widths[0],
                self._left_widths[-1],
                self._right_widths[0],
                self._right_widths[-1],
            )

            if max_width == 0:
                max_width = WaymoMap.DEFAULT_LANE_WIDTH / 2

            self._lane_width = max_width * 2

            new_left_pts = [None] * self._n_pts
            new_right_pts = [None] * self._n_pts
            for i in range(self._n_pts):
                p = self._lane_pts[i]
                new_left_pts[i] = p + (max_width * self._normals[i])
                new_right_pts[i] = p + (-1.0 * max_width * self._normals[i])

            xs, ys = [], []
            for p in new_left_pts + new_right_pts[::-1] + [new_left_pts[0]]:
                if p is not None:
                    xs.append(p[0])
                    ys.append(p[1])
            self._lane_polygon = list(zip(xs, ys))

            # No need to keep these around anymore...
            self._normals = None

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def index(self) -> int:
            return self._index

        @cached_property
        def length(self) -> float:
            length = 0.0
            for i in range(len(self._lane_pts) - 1):
                a = self._lane_pts[i]
                b = self._lane_pts[i + 1]
                length += np.linalg.norm(b - a)
            return length

        @cached_property
        def is_drivable(self) -> bool:
            # Waymo's LaneType.TYPE_BIKE_LANE = 3
            return self._lane_dict["type"] != 3

        @cached_property
        def composite_lane(self) -> RoadMap.Lane:
            composite_id = self._lane_dict.get("composite")
            if composite_id:
                return road_map.lane_by_id(composite_id)
            return self

        @property
        def is_composite(self) -> bool:
            return self._is_composite

        @cached_property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            return [
                self._map.lane_by_id(str(el)) for el in self._lane_dict["entry_lanes"]
            ]

        @cached_property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            return [
                self._map.lane_by_id(str(xl)) for xl in self._lane_dict["exit_lanes"]
            ]

        @cached_property
        def entry_surfaces(self) -> List[RoadMap.Surface]:
            # TODO?  can a non-lane connect into a lane?
            return self.incoming_lanes

        @cached_property
        def exit_surfaces(self) -> List[RoadMap.Surface]:
            # TODO?  can a lane exit to a non-lane?
            return self.outgoing_lanes

        @cached_property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            return [l for l in self.road.lanes if l != self]

        @cached_property
        def lane_to_left(self) -> Tuple[Optional[RoadMap.Lane], bool]:
            if not self._lane_dict["left_neighbors"]:
                return None, True
            left_lanes = []
            for l_neighbor in self._lane_dict["left_neighbors"]:
                left_lanes.append(self._map.lane_by_id(str(l_neighbor.feature_id)))
            assert len(left_lanes) == 1
            return left_lanes[0], True

        @cached_property
        def lane_to_right(self) -> Tuple[Optional[RoadMap.Lane], bool]:
            if not self._lane_dict["right_neighbors"]:
                return None, True
            right_lanes = []
            for r_neighbor in self._lane_dict["right_neighbors"]:
                right_lanes.append(self._map.lane_by_id(str(r_neighbor.feature_id)))
            assert len(right_lanes) == 1
            return right_lanes[0], True

        @property
        def speed_limit(self) -> float:
            return self._speed_limit

        @lru_cache(maxsize=8)
        def offset_along_lane(self, world_point: Point) -> float:
            return offset_along_shape(world_point[:2], self._centerline_pts)

        @lru_cache(maxsize=8)
        def width_at_offset(self, lane_point_s: float) -> Tuple[float, float]:
            return self._lane_width, 1.0

        @lru_cache(maxsize=8)
        def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
            return position_at_shape_offset(self._centerline_pts, lane_point.s)

        @lru_cache(maxsize=8)
        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            return super().to_lane_coord(world_point)

        @lru_cache(maxsize=8)
        def center_at_point(self, point: Point) -> Point:
            return super().center_at_point(point)

        @lru_cache(8)
        def vector_at_offset(self, start_offset: float) -> np.ndarray:
            return super().vector_at_offset(start_offset)

        @lru_cache(maxsize=8)
        def center_pose_at_point(self, point: Point) -> Pose:
            return super().center_pose_at_point(point)

        @lru_cache(maxsize=8)
        def curvature_radius_at_offset(
            self, offset: float, lookahead: int = 5
        ) -> float:
            return super().curvature_radius_at_offset(offset, lookahead)

        @lru_cache(maxsize=16)
        def oncoming_lanes_at_offset(self, offset: float) -> List[RoadMap.Lane]:
            result = []
            radius = 1.1 * self.width_at_offset(offset)[0]
            pt = self.from_lane_coord(RefLinePoint(offset))
            nearby_lanes = self._map.nearest_lanes(pt, radius=radius)
            if not nearby_lanes:
                return result
            my_vect = self.vector_at_offset(offset)
            my_norm = np.linalg.norm(my_vect)
            if my_norm == 0:
                return result
            threshold = -0.995562  # cos(175*pi/180)
            for lane, _ in nearby_lanes:
                if lane == self:
                    continue
                lane_refline_pt = lane.to_lane_coord(pt)
                lv = lane.vector_at_offset(lane_refline_pt.s)
                lv_norm = np.linalg.norm(lv)
                if lv_norm == 0:
                    continue
                lane_angle = np.dot(my_vect, lv) / (my_norm * lv_norm)
                if lane_angle < threshold:
                    result.append(lane)
            return result

        @cached_property
        def bounding_box(self) -> Optional[BoundingBox]:
            # XXX: this shouldn't be public.
            x_coordinates, y_coordinates = zip(*self._lane_polygon)
            self._bounding_box = BoundingBox(
                min_pt=Point(x=min(x_coordinates), y=min(y_coordinates)),
                max_pt=Point(x=max(x_coordinates), y=max(y_coordinates)),
            )
            return self._bounding_box

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            assert type(point) == Point
            if (
                self.bounding_box.min_pt.x <= point[0] <= self.bounding_box.max_pt.x
                and self.bounding_box.min_pt.y <= point[1] <= self.bounding_box.max_pt.y
            ):
                lane_point = self.to_lane_coord(point)
                return (
                    abs(lane_point.t) <= (self._lane_width / 2)
                    and 0 <= lane_point.s < self.length
                )
            return False

        @lru_cache(maxsize=8)
        def project_along(
            self, start_offset: float, distance: float
        ) -> Set[Tuple[RoadMap.Lane, float]]:
            return super().project_along(start_offset, distance)

        def waypoint_paths_for_pose(
            self, pose: Pose, lookahead: int, route: RoadMap.Route = None
        ) -> List[List[Waypoint]]:
            if not self.is_drivable:
                return []
            road_ids = [road.road_id for road in route.roads] if route else None
            return self._waypoint_paths_at(pose.position, lookahead, road_ids)

        def waypoint_paths_at_offset(
            self, offset: float, lookahead: int = 30, route: RoadMap.Route = None
        ) -> List[List[Waypoint]]:
            if not self.is_drivable:
                return []
            wp_start = self.from_lane_coord(RefLinePoint(offset))
            road_ids = [road.road_id for road in route.roads] if route else None
            return self._waypoint_paths_at(wp_start, lookahead, road_ids)

        def _waypoint_paths_at(
            self,
            point: Sequence,
            lookahead: int,
            filter_road_ids: Optional[Sequence[str]] = None,
        ) -> List[List[Waypoint]]:
            if not self.is_drivable:
                return []
            closest_linked_lp = (
                self._map._lanepoints.closest_linked_lanepoint_on_lane_to_point(
                    point, self._lane_id
                )
            )
            return self._map._waypoints_starting_at_lanepoint(
                closest_linked_lp,
                lookahead,
                tuple(filter_road_ids) if filter_road_ids else (),
                tuple(point),
            )

    class Road(RoadMap.Road, Surface):
        """This is akin to a 'road segment' in real life.
        Many of these might correspond to a single named road in reality."""

        def __init__(self, road_map, road_lanes: Sequence[RoadMap.Lane]):
            self._composite = None
            self._is_composite = False
            self._road_id = "waymo_road"
            for ind, lane in enumerate(road_lanes):
                self._road_id += f"-{lane.lane_id}"
                lane._road = self
                lane._index = ind
                if lane.is_composite:
                    # TAI: do we need to keep track of sub roads?
                    self._is_composite = True
                if lane.composite_lane != lane:
                    self._composite = lane.composite_lane.road
            super().__init__(self._road_id, road_map)
            self._lanes = road_lanes

            # Set road and index for its lanes
            idx_counter = 0
            for lane in self._lanes:
                lane._road = self
                lane._index = idx_counter
                idx_counter += 1

        @property
        def road_id(self) -> str:
            return self._road_id

        @cached_property
        def type(self) -> int:
            road_type = 0  # 0 == LaneType.TYPE_UNDEFINED
            for lane in self._lanes:
                if road_type != 0 and lane.type != road_type:
                    return 0
                road_type = lane.type
            return road_type

        @cached_property
        def type_as_str(self) -> str:
            road_type = self.type
            if road_type == 0:
                return "undefined"
            elif road_type == 1:
                return "freeway"
            elif road_type == 2:
                return "surface street"
            elif road_type == 3:
                return "bike lane"
            return "undefined"

        @cached_property
        def is_drivable(self) -> bool:
            for lane in self.lanes:
                if lane.is_drivable:
                    return True
            return False

        @cached_property
        def bounding_box(self) -> BoundingBox:
            """Get the minimal axis aligned bounding box that contains all map geometry."""
            x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            for lane in self._lanes:
                x_mins.append(lane.bounding_box.min_pt.x)
                y_mins.append(lane.bounding_box.min_pt.y)
                x_maxs.append(lane.bounding_box.max_pt.x)
                y_maxs.append(lane.bounding_box.max_pt.y)

            return BoundingBox(
                min_pt=Point(x=min(x_mins), y=min(y_mins)),
                max_pt=Point(x=max(x_maxs), y=max(y_maxs)),
            )

        @property
        def composite_road(self) -> RoadMap.Road:
            return self._composite or self

        @property
        def is_composite(self) -> bool:
            return self._is_composite

        @property
        def is_junction(self) -> bool:
            # TODO
            raise NotImplementedError()

        @cached_property
        def length(self) -> float:
            return max(lane.length for lane in self.lanes)

        @cached_property
        def incoming_roads(self) -> List[RoadMap.Road]:
            return list(
                {in_lane.road for lane in self.lanes for in_lane in lane.incoming_lanes}
            )

        @cached_property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            return list(
                {
                    out_lane.road
                    for lane in self.lanes
                    for out_lane in lane.outgoing_lanes
                }
            )

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            if (
                self._bounding_box.min_pt.x <= point[0] <= self._bounding_box.max_pt.x
                and self._bounding_box.min_pt.y
                <= point[1]
                <= self._bounding_box.max_pt.y
            ):
                for lane in self._lanes:
                    if lane.contains_point(point):
                        return True
            return False

        @lru_cache(maxsize=16)
        def oncoming_roads_at_point(self, point: Point) -> List[RoadMap.Road]:
            result = []
            for lane in self.lanes:
                offset = lane.to_lane_coord(point).s
                result += [
                    ol.road
                    for ol in lane.oncoming_lanes_at_offset(offset)
                    if ol.road != self
                ]
            return result

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            return []

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            return self._lanes

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            return self._lanes[index]

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if not road:
            self._log.warning(f"WaymoMap got request for unknown road_id '{road_id}'")
        return road

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        # note: all lanes were cached already by _load()
        lane = self._lanes.get(lane_id)
        if not lane:
            self._log.warning(f"WaymoMap got request for unknown lane_id '{lane_id}'")
        return lane

    def _build_lane_r_tree(self):
        result = rtree.index.Index()
        result.interleaved = True
        all_lanes = list(self._lanes.values())
        for idx, lane in enumerate(all_lanes):
            bounding_box = (
                lane.bounding_box.min_pt.x,
                lane.bounding_box.min_pt.y,
                lane.bounding_box.max_pt.x,
                lane.bounding_box.max_pt.y,
            )
            result.add(idx, bounding_box)
        return result

    def _get_neighboring_lanes(
        self, x: float, y: float, r: float = 0.1
    ) -> List[Tuple[RoadMap.Lane, float]]:
        neighboring_lanes = []
        all_lanes = list(self._lanes.values())
        if self._lane_rtree is None:
            self._lane_rtree = self._build_lane_r_tree()

        for i in self._lane_rtree.intersection((x - r, y - r, x + r, y + r)):
            lane = all_lanes[i]
            d = distance_point_to_polygon((x, y), lane._lane_polygon)
            if d < r:
                neighboring_lanes.append((lane, d))
        return neighboring_lanes

    @lru_cache(maxsize=16)
    def nearest_lanes(
        self, point: Point, radius: Optional[float] = None, include_junctions=False
    ) -> List[Tuple[RoadMap.Lane, float]]:
        if radius is None:
            radius = max(10, 2 * self._default_lane_width)
        candidate_lanes = self._get_neighboring_lanes(point[0], point[1], r=radius)
        candidate_lanes.sort(key=lambda lane_dist_tup: lane_dist_tup[1])
        return candidate_lanes

    def nearest_lane(
        self, point: Point, radius: float = None, include_junctions=False
    ) -> Optional[RoadMap.Lane]:
        nearest_lanes = self.nearest_lanes(point, radius, include_junctions)
        return nearest_lanes[0][0] if nearest_lanes else None

    @lru_cache(maxsize=16)
    def road_with_point(self, point: Point) -> RoadMap.Road:
        radius = max(5, 2 * self._default_lane_width)
        for nl, dist in self.nearest_lanes(point, radius):
            if nl.contains_point(point):
                return nl.road
        return None

    class _WaypointsCache:
        def __init__(self):
            self.lookahead = 0
            self.point = (0, 0, 0)
            self.filter_road_ids = ()
            self._starts = {}

        # XXX:  all vehicles share this cache now (as opposed to before
        # when it was in Plan.py and each vehicle had its own cache).
        # TODO: probably need to add vehicle_id to the key somehow (or just make it bigger)
        def _match(self, lookahead, point, filter_road_ids) -> bool:
            return (
                lookahead <= self.lookahead
                and point[0] == self.point[0]
                and point[1] == self.point[1]
                and filter_road_ids == self.filter_road_ids
            )

        def update(
            self,
            lookahead: int,
            point: Tuple[float, float, float],
            filter_road_ids: tuple,
            llp,
            paths: List[List[Waypoint]],
        ):
            """Update the current cache if not already cached."""
            if not self._match(lookahead, point, filter_road_ids):
                self.lookahead = lookahead
                self.point = point
                self.filter_road_ids = filter_road_ids
                self._starts = {}
            self._starts[llp.lp.lane.index] = paths

        def query(
            self,
            lookahead: int,
            point: Tuple[float, float, float],
            filter_road_ids: tuple,
            llp,
        ) -> Optional[List[List[Waypoint]]]:
            """Attempt to find previously cached waypoints"""
            if self._match(lookahead, point, filter_road_ids):
                hit = self._starts.get(llp.lp.lane.index, None)
                if hit:
                    # consider just returning all of them (not slicing)?
                    return [path[: (lookahead + 1)] for path in hit]
                return None

    def waypoint_paths(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: float = 5,
        route: RoadMap.Route = None,
    ) -> List[List[Waypoint]]:
        road_ids = []
        if route and route.roads:
            road_ids = [road.road_id for road in route.roads]
        if road_ids:
            return self._waypoint_paths_along_route(pose.position, lookahead, road_ids)
        closest_lps = self._lanepoints.closest_lanepoints(
            [pose], within_radius=within_radius
        )
        closest_lane = closest_lps[0].lane
        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += lane._waypoint_paths_at(pose.position, lookahead)
        return sorted(waypoint_paths, key=lambda p: p[0].lane_index)

    def _waypoint_paths_along_route(
        self, point, lookahead: int, route: Sequence[str]
    ) -> List[List[Waypoint]]:
        """finds the closest lane to vehicle's position that is on its route,
        then gets waypoint paths from all lanes in its road there."""
        assert len(route) > 0, f"Expected at least 1 road in the route, got: {route}"
        closest_llp_on_each_route_road = [
            self._lanepoints.closest_linked_lanepoint_on_road(point, road)
            for road in route
        ]
        closest_linked_lp = min(
            closest_llp_on_each_route_road,
            key=lambda l_lp: np.linalg.norm(
                vec_2d(l_lp.lp.pose.position) - vec_2d(point)
            ),
        )
        closest_lane = closest_linked_lp.lp.lane
        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += lane._waypoint_paths_at(point, lookahead, route)

        return sorted(waypoint_paths, key=len, reverse=True)

    @staticmethod
    def _equally_spaced_path(
        path: Sequence[LinkedLanePoint],
        point: Tuple[float, float, float],
        lp_spacing: float,
    ) -> List[Waypoint]:
        """given a list of LanePoints starting near point, return corresponding
        Waypoints that may not be evenly spaced (due to lane change) but start at point."""

        continuous_variables = [
            "positions_x",
            "positions_y",
            "headings",
            "lane_width",
            "speed_limit",
        ]
        discrete_variables = ["lane_id", "lane_index"]

        ref_lanepoints_coordinates = {
            parameter: [] for parameter in (continuous_variables + discrete_variables)
        }
        for idx, lanepoint in enumerate(path):

            if lanepoint.is_inferred and 0 < idx < len(path) - 1:
                continue

            ref_lanepoints_coordinates["positions_x"].append(
                lanepoint.lp.pose.position[0]
            )
            ref_lanepoints_coordinates["positions_y"].append(
                lanepoint.lp.pose.position[1]
            )
            ref_lanepoints_coordinates["headings"].append(
                lanepoint.lp.pose.heading.as_bullet
            )
            ref_lanepoints_coordinates["lane_id"].append(lanepoint.lp.lane.lane_id)
            ref_lanepoints_coordinates["lane_index"].append(lanepoint.lp.lane.index)

            ref_lanepoints_coordinates["lane_width"].append(lanepoint.lp.lane_width)

            ref_lanepoints_coordinates["speed_limit"].append(
                lanepoint.lp.lane.speed_limit
            )

        ref_lanepoints_coordinates["headings"] = inplace_unwrap(
            ref_lanepoints_coordinates["headings"]
        )
        first_lp_heading = ref_lanepoints_coordinates["headings"][0]
        lp_position = path[0].lp.pose.position[:2]
        vehicle_pos = np.array(point[:2])
        heading_vec = np.array(radians_to_vec(first_lp_heading))
        projected_distant_lp_vehicle = np.inner(
            (vehicle_pos - lp_position), heading_vec
        )

        ref_lanepoints_coordinates["positions_x"][0] = (
            lp_position[0] + projected_distant_lp_vehicle * heading_vec[0]
        )
        ref_lanepoints_coordinates["positions_y"][0] = (
            lp_position[1] + projected_distant_lp_vehicle * heading_vec[1]
        )

        cumulative_path_dist = np.cumsum(
            np.sqrt(
                np.ediff1d(ref_lanepoints_coordinates["positions_x"], to_begin=0) ** 2
                + np.ediff1d(ref_lanepoints_coordinates["positions_y"], to_begin=0) ** 2
            )
        )

        if len(cumulative_path_dist) <= lp_spacing:
            lp = path[0].lp

            return [
                Waypoint(
                    pos=lp.pose.position,
                    heading=lp.pose.heading,
                    lane_width=lp.lane.width_at_offset(0)[0],
                    speed_limit=lp.lane.speed_limit,
                    lane_id=lp.lane.lane_id,
                    lane_index=lp.lane.index,
                )
            ]

        evenly_spaced_cumulative_path_dist = np.linspace(
            0, cumulative_path_dist[-1], len(path)
        )

        evenly_spaced_coordinates = {}
        for variable in continuous_variables:
            evenly_spaced_coordinates[variable] = np.interp(
                evenly_spaced_cumulative_path_dist,
                cumulative_path_dist,
                ref_lanepoints_coordinates[variable],
            )

        for variable in discrete_variables:
            ref_coordinates = ref_lanepoints_coordinates[variable]
            evenly_spaced_coordinates[variable] = []
            jdx = 0
            for idx in range(len(path)):
                while (
                    jdx + 1 < len(cumulative_path_dist)
                    and evenly_spaced_cumulative_path_dist[idx]
                    > cumulative_path_dist[jdx + 1]
                ):
                    jdx += 1

                evenly_spaced_coordinates[variable].append(ref_coordinates[jdx])
            evenly_spaced_coordinates[variable].append(ref_coordinates[-1])

        waypoint_path = []
        for idx in range(len(path)):
            waypoint_path.append(
                Waypoint(
                    pos=np.array(
                        [
                            evenly_spaced_coordinates["positions_x"][idx],
                            evenly_spaced_coordinates["positions_y"][idx],
                        ]
                    ),
                    heading=Heading(evenly_spaced_coordinates["headings"][idx]),
                    lane_width=evenly_spaced_coordinates["lane_width"][idx],
                    speed_limit=evenly_spaced_coordinates["speed_limit"][idx],
                    lane_id=evenly_spaced_coordinates["lane_id"][idx],
                    lane_index=evenly_spaced_coordinates["lane_index"][idx],
                )
            )

        return waypoint_path

    def _waypoints_starting_at_lanepoint(
        self,
        lanepoint: LinkedLanePoint,
        lookahead: int,
        filter_road_ids: tuple,
        point: Tuple[float, float, float],
    ) -> List[List[Waypoint]]:
        """computes equally-spaced Waypoints for all lane paths starting at lanepoint
        up to lookahead waypoints ahead, constrained to filter_road_ids if specified."""

        # The following acts sort of like lru_cache(1), but it allows
        # for lookahead to be <= to the cached value...
        cache_paths = self._waypoints_cache.query(
            lookahead, point, filter_road_ids, lanepoint
        )
        if cache_paths:
            return cache_paths

        lanepoint_paths = self._lanepoints.paths_starting_at_lanepoint(
            lanepoint, lookahead, filter_road_ids
        )
        result = [
            WaymoMap._equally_spaced_path(
                path,
                point,
                self._map_spec.lanepoint_spacing,
            )
            for path in lanepoint_paths
        ]

        self._waypoints_cache.update(
            lookahead, point, filter_road_ids, lanepoint, result
        )

        return result