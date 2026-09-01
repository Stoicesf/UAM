"""VMAS Navigation + static obstacles (obs dim unchanged: 18 for 4 agents)."""

from __future__ import annotations

import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    """Navigation with optional static obstacles (``n_obstacles``).

    Observation layout matches standard navigation (18-dim @ 4 agents, 12 lidar rays):
    [pos(2), vel(2), goal_rel(2), lidar(12)].
    When ``n_obstacles > 0``, lidar detects agents + collidable obstacles.
    """

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 4)
        self.collisions = kwargs.pop("collisions", True)
        self.n_obstacles = kwargs.pop("n_obstacles", 0)

        self.world_spawning_x = kwargs.pop("world_spawning_x", 1.0)
        self.world_spawning_y = kwargs.pop("world_spawning_y", 1.0)
        self.enforce_bounds = kwargs.pop("enforce_bounds", False)

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 0.35)
        self.agent_radius = kwargs.pop("agent_radius", 0.1)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 12)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        self.obstacle_collision_penalty = kwargs.pop("obstacle_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
        else:
            self.x_semidim = None
            self.y_semidim = None

        assert 1 <= self.agents_with_same_goal <= self.n_agents

        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )

        def lidar_filter(e: Entity) -> bool:
            if isinstance(e, Agent):
                return True
            return isinstance(e, Landmark) and getattr(e, "collide", False)

        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)
        lidar_filter_fn = lidar_filter if self.n_obstacles > 0 else entity_filter_agents

        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=lidar_filter_fn,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            agent.obstacle_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        self.obstacles: List[Landmark] = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.agent_radius),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        return world

    def reset_world_at(self, env_index: int = None):
        spawn_x = max(self.world_spawning_x, 1.0 + 0.15 * self.n_obstacles)
        spawn_y = max(self.world_spawning_y, 1.0 + 0.15 * self.n_obstacles)

        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-spawn_x, spawn_x),
            (-spawn_y, spawn_y),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-spawn_x, spawn_x),
                y_bounds=(-spawn_y, spawn_y),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i
            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

        if self.obstacles:
            ScenarioUtils.spawn_entities_randomly(
                self.obstacles,
                self.world,
                env_index,
                self.min_distance_between_entities,
                (-spawn_x, spawn_x),
                (-spawn_y, spawn_y),
                occupied_positions=occupied_positions,
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0
                a.obstacle_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )
            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        penalty = self.agent_collision_penalty
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += penalty

                for obstacle in self.obstacles:
                    if self.world.collides(a, obstacle):
                        distance = self.world.get_distance(a, obstacle)
                        a.obstacle_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.obstacle_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return (
            pos_reward
            + self.final_rew
            + agent.agent_collision_rew
            + agent.obstacle_collision_rew
        )

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )

    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
            "obstacle_collisions": agent.obstacle_collision_rew,
        }

    def extra_render(self, env_index: int = 0):
        return []
