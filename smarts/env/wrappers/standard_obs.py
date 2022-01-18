# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import gym
import numpy as np

from smarts.core.events import Events
from smarts.core.road_map import Waypoint
from smarts.core.sensors import (
    DrivableAreaGridMap,
    EgoVehicleObservation,
    Observation,
    OccupancyGridMap,
    TopDownRGB,
    VehicleObservation,
)
from smarts.env.custom_observations import lane_ttc

_LIDAR_SHP = 300
_NEIGHBOR_SHP = 10
_WAYPOINT_SHP = (4, 20)


@dataclass(frozen=True)
class StdObs:
    distance_travelled: np.float32
    ego_vehicle_state: Dict[str, Union[np.float32, np.ndarray]]
    events: Dict[str, int]

    drivable_area_grid_map: Optional[np.ndarray] = None
    lidar_point_cloud: Optional[Dict[str, np.ndarray]] = None
    neighborhood_vehicle_states: Optional[Dict[str, np.ndarray]] = None
    occupancy_grid_map: Optional[np.ndarray] = None
    top_down_rgb: Optional[np.ndarray] = None
    ttc: Optional[Dict[str, np.ndarray]] = None
    waypoint_paths: Optional[Dict[str, np.ndarray]] = None


class StandardObs(gym.ObservationWrapper):
    """Converts SMARTS observations to standardized gym-compliant observations.
    The observation set returned depends on the features enabled via
    AgentInterface.

    Note: To use StandardObs wrapper, all agents must have the same
    AgentInterface attributes.

    The complete set of available observation per agent is as follows.

    1. "distance_travelled"
        Total distance travelled in meters. dtype=np.float32.
    2. "drivable_area_grid_map"
        Binary map. 255 if a cell contains a road, else 0. dtype=np.uint8.
    3. "ego_vehicle_state"
        a. "angular_acceleration"
            Angular acceleration vector. Requires `accelerometer` attribute
            enabled in AgentInterface. shape=(3,). dtype=np.float32.
        b. "angular_jerk"
            Angular jerk vector. Requires `accelerometer` attribute enabled in
            AgentInterface. shape=(3,). dtype=np.float32.
        c. "angular_velocity"
            Angular velocity vector. shape=(3,). dtype=np.float32).
        d. "bounding_box"
            Length, width, and height of the vehicle bounding box. shape=(3,).
            dtype=np.float32.
        e. "heading"
            Vehicle heading in radians [-pi, pi]. dtype=np.float32.
        f. "lane_index"
            Vehicle's lane number. Rightmost lane has index 0 and increases
            towards left. dtype=np.int8.
        g. "linear_acceleration"
            Vehicle acceleration in x, y, and z axes. Requires `accelerometer`
            attribute enabled in AgentInterface. shape=(3,). dtype=np.float32.
        h. "linear_jerk"
            Linear jerk vector. Requires `accelerometer` attribute enabled in
            AgentInterface. shape=(3,). dtype=np.float32.
        i. "linear_velocity"
            Vehicle velocity in x, y, and z axes. shape=(3,). dtype=np.float32.
        j. "position"
            Coordinate of the center of the vehicle bounding box's bottom
            plane. shape=(3,). dtype=np.float32.
        k. "speed"
            Vehicle speed in m/s. dtype=np.float32.
        l. "steering"
            Angle of front wheels in radians [-pi, pi]. dtype=np.float32.
        m. "yaw_rate"
            Rotation speed around vertical axis in rad/s [0, 2pi].
            dtype=np.float32.
    4. "lidar_point_cloud"
        a. "hit"
            Binary array. 1 if an object is hit, else 0. shape(300,).
        b. "point_cloud"
            Coordinates of point cloud. shape=(300,3). dtype=np.float32.
        c. "ray_origin"
            Ray origin coordinates. shape=(300,3). dtype=np.float32.
        d. "ray_vector"
            Ray vectors. shape=(300,3). dtype=np.float32.
    5. "events"
        a. "agents_alive_done"
            1 if `DoneCriteria.agents_alive` is triggered, else 0.
        b. "collisions"
            1 if any collisions occurred with ego vehicle, else 0.
        c. "not_moving"
            1 if `DoneCriteria.not_moving` is triggered, else 0.
        d. "off_road"
            1 if ego vehicle drives off road, else 0.
        e. "off_route"
            1 if ego vehicle drives off mission route, else 0.
        f. "on_shoulder"
            1 if ego vehicle drives on road shoulder, else 0.
        g. "reached_goal"
            1 if ego vehicle reaches its goal, else 0.
        h. "reached_max_episode_steps"
            1 if maximum episode steps reached, else 0.
        i. "wrong_way"
            1 if ego vehicle drives in the wrong traffic direction, else 0.
    6. "neighborhood_vehicle_states"
        Feature array of 10 nearest neighbor vehicles. If nearest neighbor
        vehicles are insufficient, default feature values are padded.
        a. "bounding_box"
            Bounding box of neighbor vehicles. Defaults to np.array([0,0,0])
            per vehicle. shape=(10,3). dtype=np.float32.
        b. "heading"
            Heading of neighbor vehicles in radians [-pi, pi]. Defaults to
            np.array([0]) per vehicle. shape=(10,). dtype=np.float32.
        c. "lane_index"
            Lane number of neighbor vehicles. Defaults to np.array([0]) per
            vehicle. shape=(10,). dtype=np.int8.
        d. "position"
            Coordinate of the center of neighbor vehicles' bounding box's
            bottom plane. Defaults to np.array([0,0,0]) per vehicle.
            shape=(10,3). dtype=np.float32.
        e. "speed"
            Speed of neighbor vehicles in m/s. Defaults to np.array([0]) per
            vehicle. shape=(10,). dtype=np.float32.
    7. "occupancy_grid_map"
        Binary map. 255 if a cell is occupied, else 0. dtype=np.uint8.
    8. "top_down_rgb"
        RGB image, from the top, with ego vehicle at the center.
        shape=(height, width, 3). dtype=np.uint8.
    9. "ttc"
        Time and distance to collision. Enabled only if both `waypoints` and
        `neighborhood_vehicles` attributes are enabled in AgentInterface.
        a. "angle_error"
            Angular error in radians [-pi, pi]. shape=(1,). dtype=np.float32.
        b. "distance_from_center"
            Distance of vehicle from lane center in meters. shape=(1,).
            dtype=np.float32.
        c. "ego_lane_dist"
            Distance to collision on the right lane (`ego_lane_dist[0]`),
            current lane (`ego_lane_dist[1]`), and left lane
            (`ego_lane_dist[2]`). If no lane is available, to the right or to
            the left, default value of 0 is padded. shape=(3,).
            dtype=np.float32.
        d. "ego_ttc"
            Time to collision on the right lane (`ego_ttc[0]`), current lane
            (`ego_ttc[1]`), and left lane (`ego_ttc[2]`). If no lane is
            available, to the right or to the left, default value of 0 is
            padded. shape=(3,). dtype=np.float32.
    10."waypoint_paths"
        Array of 20 waypoints ahead or in the mission route, from the nearest 4
        lanes. If lanes or waypoints ahead are insufficient, default values are
        padded.
        a. "heading"
            Lane heading angle at a waypoint in radians [-pi, pi]. Defaults to
            np.array([0]) per waypoint. shape=(4,20). dtype=np.float32.
        b. "lane_index"
            Lane number at a waypoint. Defaults to np.array([0]) per waypoint.
            shape=(4,20). dtype=np.int8.
        c. "lane_width"
            Lane width at a waypoint in meters. Defaults to np.array([0]) per
            waypoint. shape=(4,20). dtype=np.float32.
        d. "pos"
            Coordinate of a waypoint. Defaults to np.array([0,0,0]).
            shape=(4,20,3). dtype=np.float32.
        e. "speed_limit"
            Lane speed limit at a waypoint in m/s. shape=(4,20).
            dtype=np.float32.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): SMARTS environment to be wrapped.
        """
        super().__init__(env)

        agent_id = next(iter(self.agent_specs.keys()))
        intrfcs = {}
        for intrfc in {
            "accelerometer",
            "drivable_area_grid_map",
            "lidar",
            "neighborhood_vehicles",
            "ogm",
            "rgb",
            "waypoints",
        }:
            val = getattr(self.agent_specs[agent_id].interface, intrfc)
            if val:
                self._comp_intrfc(intrfc, val)
                intrfcs.update({intrfc: val})

        self._obs = {
            "distance_travelled",
            "drivable_area_grid_map",
            "ego_vehicle_state",
            "events",
            "lidar_point_cloud",
            "neighborhood_vehicle_states",
            "occupancy_grid_map",
            "top_down_rgb",
            "ttc",
            "waypoint_paths",
        }

        space = _make_space(intrfcs)
        self.observation_space = gym.spaces.Dict(
            {agent_id: gym.spaces.Dict(space) for agent_id in self.agent_specs.keys()}
        )

    def _comp_intrfc(self, intrfc: str, val: Any):
        assert all(
            getattr(self.agent_specs[agent_id].interface, intrfc) == val
            for agent_id in self.agent_specs.keys()
        ), f"To use StandardObs wrapper, all agents must have the same "
        f"AgentInterface.{intrfc} attribute."

    def observation(self, obs: Dict[str, Any]) -> Dict[str, StdObs]:
        """Converts SMARTS observations to standardized gym-compliant
        observations.

        Note: Users should not directly call this method.
        """
        wrapped_obs = {}
        for agent_id, agent_obs in obs.items():
            wrapped_ob = {}
            for ob in self._obs:
                func = globals()[f"_std_{ob}"]
                if ob == "ttc":
                    val = func(obs[agent_id])
                else:
                    val = func(getattr(agent_obs, ob))
                wrapped_ob.update({ob: val})
            wrapped_obs.update({agent_id: StdObs(**wrapped_ob)})

        return wrapped_obs


def _make_space(intrfcs: Dict[str, Any]) -> gym.spaces:
    intrfc_to_obs = {
        "drivable_area_grid_map": "drivable_area_grid_map",
        "lidar": "lidar_point_cloud",
        "neighborhood_vehicles": "neighborhood_vehicle_states",
        "ogm": "occupancy_grid_map",
        "rgb": "top_down_rgb",
        "waypoints": "waypoint_paths",
    }

    # fmt: off
    space = {
        "distance_travelled": gym.spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
        "ego_vehicle_state": gym.spaces.Dict({
            "angular_acceleration": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "angular_jerk": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "angular_velocity": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
            "bounding_box": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
            "heading": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
            "lane_index": gym.spaces.Box(low=0, high=127, shape=(1,), dtype=np.int8),
            "linear_acceleration": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "linear_velocity": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
            "linear_jerk": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "position": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "speed": gym.spaces.Box(low=0, high=1e10, shape=(1,), dtype=np.float32),
            "steering": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32),
            "yaw_rate": gym.spaces.Box(low=0, high=2*math.pi, shape=(1,), dtype=np.float32),
        }),
        "events": gym.spaces.Dict({
            "agents_alive_done": gym.spaces.MultiBinary(1),
            "collisions": gym.spaces.MultiBinary(1),
            "not_moving": gym.spaces.MultiBinary(1),
            "off_road": gym.spaces.MultiBinary(1),
            "off_route": gym.spaces.MultiBinary(1),
            "on_shoulder": gym.spaces.MultiBinary(1),
            "reached_goal": gym.spaces.MultiBinary(1),
            "reached_max_episode_steps": gym.spaces.MultiBinary(1),
            "wrong_way": gym.spaces.MultiBinary(1),
        }),
    }

    opt_space = {
        "drivable_area_grid_map": lambda val: gym.spaces.Box(low=0, high=255, shape=(val.height, val.width, 1), dtype=np.uint8),
        "lidar_point_cloud": lambda _: gym.spaces.Dict({
            "hit": gym.spaces.MultiBinary(_LIDAR_SHP),
            "point_cloud": gym.spaces.Box(low=-1e10, high=1e10, shape=(_LIDAR_SHP,3), dtype=np.float32),
            "ray_origin": gym.spaces.Box(low=-1e10, high=1e10, shape=(_LIDAR_SHP,3), dtype=np.float32),
            "ray_vector": gym.spaces.Box(low=-1e10, high=1e10, shape=(_LIDAR_SHP,3), dtype=np.float32),
        }),
        "neighborhood_vehicle_states": lambda _: gym.spaces.Dict({
            "bounding_box": gym.spaces.Box(low=0, high=1e10, shape=(_NEIGHBOR_SHP,3), dtype=np.float32),
            "heading": gym.spaces.Box(low=-math.pi, high=math.pi, shape=(_NEIGHBOR_SHP,), dtype=np.float32),
            "lane_index": gym.spaces.Box(low=0, high=127, shape=(_NEIGHBOR_SHP,), dtype=np.int8),
            "position": gym.spaces.Box(low=-1e10, high=1e10, shape=(_NEIGHBOR_SHP,3), dtype=np.float32),    
            "speed": gym.spaces.Box(low=0, high=1e10, shape=(_NEIGHBOR_SHP,), dtype=np.float32),
        }),
        "occupancy_grid_map": lambda val: gym.spaces.Box(low=0, high=255,shape=(val.height, val.width, 1), dtype=np.uint8),
        "top_down_rgb": lambda val: gym.spaces.Box(low=0, high=255, shape=(val.height, val.width, 3), dtype=np.uint8),
        "ttc": lambda _: gym.spaces.Dict({
            "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,), dtype=np.float32),
            "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,), dtype=np.float32),
            "ego_ttc": gym.spaces.Box(low=0, high=1e10, shape=(3,), dtype=np.float32),
        }),
        "waypoint_paths": lambda _: gym.spaces.Dict({
            "heading": gym.spaces.Box(low=-math.pi, high=math.pi, shape=_WAYPOINT_SHP, dtype=np.float32),
            "lane_index": gym.spaces.Box(low=0, high=127, shape=_WAYPOINT_SHP, dtype=np.int8),
            "lane_width": gym.spaces.Box(low=0, high=1e10, shape=_WAYPOINT_SHP, dtype=np.float32),
            "pos": gym.spaces.Box(low=-1e10, high=1e10, shape=_WAYPOINT_SHP + (3,), dtype=np.float32),
            "speed_limit": gym.spaces.Box(low=0, high=1e10, shape=_WAYPOINT_SHP, dtype=np.float32),
        }),
    }
    # fmt: on

    for intrfc, val in intrfcs.items():
        opt_ob = intrfc_to_obs.get(intrfc, None)
        if opt_ob:
            space.update({opt_ob: opt_space[opt_ob](val)})

    if "waypoints" in intrfcs.keys() and "neighborhood_vehicles" in intrfcs.keys():
        space.update({"ttc": opt_space["ttc"](None)})

    return space


def _std_distance_travelled(val: float) -> float:
    return np.float32(val)


def _std_drivable_area_grid_map(
    val: Optional[DrivableAreaGridMap],
) -> Optional[np.ndarray]:
    if not val:
        return None
    return val.data.astype(np.uint8)


def _std_ego_vehicle_state(
    val: EgoVehicleObservation,
) -> Dict[str, Union[np.float32, np.ndarray]]:
    return {
        "angular_acceleration": val.angular_acceleration.astype(np.float32),
        "angular_jerk": val.angular_jerk.astype(np.float32),
        "angular_velocity": val.angular_acceleration.astype(np.float32),
        "bounding_box": np.array(val.bounding_box.as_lwh).astype(np.float32),
        "heading": np.float32(val.heading),
        "lane_index": np.int8(val.lane_index),
        "linear_acceleration": val.linear_acceleration.astype(np.float32),
        "linear_jerk": val.linear_jerk.astype(np.float32),
        "linear_velocity": val.linear_velocity.astype(np.float32),
        "position": val.position.astype(np.float32),
        "speed": np.float32(val.speed),
        "steering": np.float32(val.steering),
        "yaw_rate": np.float32(val.yaw_rate),
    }


def _std_events(val: Events) -> Dict[str, int]:
    return {
        "agents_alive_done": int(val.agents_alive_done),
        "collisions": int(len(val.collisions) > 0),
        "not_moving": int(val.not_moving),
        "off_road": int(val.off_road),
        "off_route": int(val.off_route),
        "on_shoulder": int(val.on_shoulder),
        "reached_goal": int(val.reached_goal),
        "reached_max_episode_steps": int(val.reached_max_episode_steps),
        "wrong_way": int(val.wrong_way),
    }


def _std_lidar_point_cloud(val) -> Optional[Dict[str, np.ndarray]]:
    if not val:
        return None

    des_shp = _LIDAR_SHP
    hit = np.array(val[1], dtype=np.uint8)
    point_cloud = np.array(val[0], dtype=np.float32)
    point_cloud = np.nan_to_num(
        point_cloud,
        copy=False,
        nan=np.float32(0),
        posinf=np.float32(0),
        neginf=np.float32(0),
    )
    ray_origin, ray_vector = zip(*(val[2]))
    ray_origin = np.array(ray_origin, np.float32)
    ray_vector = np.array(ray_vector, np.float32)

    try:
        assert hit.shape == (des_shp,)
        assert point_cloud.shape == (des_shp, 3)
        assert ray_origin.shape == (des_shp, 3)
        assert ray_vector.shape == (des_shp, 3)
    except:
        raise Exception("Internal Error: Mismatched lidar point cloud shape.")

    return {
        "hit": hit,
        "point_cloud": point_cloud,
        "ray_origin": ray_origin,
        "ray_vector": ray_vector,
    }


def _std_neighborhood_vehicle_states(
    nghbs: Optional[List[VehicleObservation]],
) -> Optional[Dict[str, np.ndarray]]:
    if not nghbs:
        return None

    des_shp = _NEIGHBOR_SHP
    rcv_shp = len(nghbs)
    pad_shp = 0 if des_shp - rcv_shp < 0 else des_shp - rcv_shp

    nghbs = [
        (
            nghb.bounding_box.as_lwh,
            nghb.heading,
            nghb.lane_index,
            nghb.position,
            nghb.speed,
        )
        for nghb in nghbs[:des_shp]
    ]
    bounding_box, heading, lane_index, position, speed = zip(*nghbs)

    bounding_box = np.array(bounding_box, dtype=np.float32)
    heading = np.array(heading, dtype=np.float32)
    lane_index = np.array(lane_index, dtype=np.int8)
    position = np.array(position, dtype=np.float32)
    speed = np.array(speed, dtype=np.float32)

    # fmt: off
    bounding_box = np.pad(bounding_box, ((0,pad_shp),(0,0)), mode='constant', constant_values=0)
    heading = np.pad(heading, ((0,pad_shp)), mode='constant', constant_values=0)
    lane_index = np.pad(lane_index, ((0,pad_shp)), mode='constant', constant_values=0)
    position = np.pad(position, ((0,pad_shp),(0,0)), mode='constant', constant_values=0)
    speed = np.pad(speed, ((0,pad_shp)), mode='constant', constant_values=0)
    # fmt: on

    return {
        "bounding_box": bounding_box,
        "heading": heading,
        "lane_index": lane_index,
        "position": position,
        "speed": speed,
    }


def _std_occupancy_grid_map(val: Optional[OccupancyGridMap]) -> Optional[np.ndarray]:
    if not val:
        return None
    return val.data.astype(np.uint8)


def _std_top_down_rgb(val: Optional[TopDownRGB]) -> Optional[np.ndarray]:
    if not val:
        return None
    return val.data.astype(np.uint8)


def _std_ttc(obs: Observation) -> Optional[Dict[str, np.ndarray]]:
    if not obs.neighborhood_vehicle_states or not obs.waypoint_paths:
        return None

    val = lane_ttc(obs)
    return {
        "angle_error": np.array(val["angle_error"], dtype=np.float32),
        "distance_from_center": np.array(val["distance_from_center"], dtype=np.float32),
        "ego_lane_dist": np.array(val["ego_lane_dist"], dtype=np.float32),
        "ego_ttc": np.array(val["ego_ttc"], dtype=np.float32),
    }


def _std_waypoint_paths(
    paths: Optional[List[List[Waypoint]]],
) -> Optional[Dict[str, np.ndarray]]:
    if not paths:
        return None

    des_shp = _WAYPOINT_SHP
    rcv_shp = (len(paths), len(paths[0]))
    pad_shp = [0 if des - rcv < 0 else des - rcv for des, rcv in zip(des_shp, rcv_shp)]

    def extract_elem(waypoint):
        return (
            waypoint.heading,
            waypoint.lane_index,
            waypoint.lane_width,
            waypoint.pos,
            waypoint.speed_limit,
        )

    paths = [map(extract_elem, path[: des_shp[1]]) for path in paths[: des_shp[0]]]
    heading, lane_index, lane_width, pos, speed_limit = zip(
        *[zip(*path) for path in paths]
    )

    heading = np.array(heading, dtype=np.float32)
    lane_index = np.array(lane_index, dtype=np.int8)
    lane_width = np.array(lane_width, dtype=np.float32)
    pos = np.array(pos, dtype=np.float32)
    speed_limit = np.array(speed_limit, dtype=np.float32)

    # fmt: off
    heading = np.pad(heading, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_index = np.pad(lane_index, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    lane_width = np.pad(lane_width, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    pos = np.pad(pos, ((0,pad_shp[0]),(0,pad_shp[1]),(0,1)), mode='constant', constant_values=0)
    speed_limit = np.pad(speed_limit, ((0,pad_shp[0]),(0,pad_shp[1])), mode='constant', constant_values=0)
    # fmt: on

    return {
        "heading": heading,
        "lane_index": lane_index,
        "lane_width": lane_width,
        "pos": pos,
        "speed_limit": speed_limit,
    }