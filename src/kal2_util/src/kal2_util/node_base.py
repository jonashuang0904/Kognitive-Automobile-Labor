#!/usr/bin/env python3
from typing import Any

import rospy
import yaml
from cachetools import TTLCache, cached

from abc import ABC, abstractmethod


class ParamHelper:
    def __init__(self, *, ns: str, parameter_cache_time: float = 1):
        self._ns = ns
        self._cache = TTLCache(maxsize=128, ttl=parameter_cache_time)
        self._get_cached_param = cached(cache=self._cache)(rospy.get_param)

    def __getattr__(self, key: str) -> Any:
        return self._get_cached_param(f"{self._ns}{key}")

    def __setattr__(self, key: str, value: Any):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        
        self._cache.clear()
        rospy.set_param(f"{self._ns}{key}", value)

    def __repr__(self) -> str:
        return yaml.dump(self._get_param(self._ns), default_flow_style=False)


class NodeBase(ABC):
    def __init__(
        self, *, name: str, parameter_cache_time: float = 1, log_level: int = rospy.INFO
    ):
        rospy.init_node(name, log_level=log_level)

        self._params = ParamHelper(ns="~", parameter_cache_time=parameter_cache_time)
        self._active = False
        rospy.on_shutdown(self.__shutdown)

    def __shutdown(self):
        if self._active:
            self._active = False
            self.stop()

    def run(self):
        rospy.spin()

    @property
    def params(self) -> ParamHelper:
        return self._params
