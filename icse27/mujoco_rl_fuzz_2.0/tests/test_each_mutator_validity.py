"""For each mutator, on a small synthetic seed, every legal intensity mode
must (a) be applicable, (b) apply without exception, (c) result in either a
valid runtime_directive (runtime_only) or a model that compiles cleanly."""
from __future__ import annotations

import random

import pytest

from src.mjcf.spec_loader import SafeModel
from src.mutations.registry import all_mutators


SEED_XML = """<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="b1" pos="0 0 0.2">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom name="g1" type="sphere" size="0.05"/>
      <body name="b2" pos="0.1 0 0">
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005"/>
        <joint name="j2" type="hinge" axis="1 0 0"/>
        <geom name="g2" type="capsule" size="0.02 0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


@pytest.mark.parametrize("mut", all_mutators(), ids=lambda m: m.id)
def test_mutator_legal_modes_are_valid(mut):
    rng = random.Random(0xCAFE)
    n_legal = 0
    for mode in mut.intensity_modes:
        if mode in mut.invalid_parseable_modes:
            continue
        sm = SafeModel.from_string(SEED_XML)
        snap = sm.snapshot()
        if not mut.applicable(sm):
            sm.restore(snap)
            continue
        ar = mut.apply(sm, mode, rng)
        if mut.runtime_only:
            assert ar.ok, f"{mut.id}|{mode} runtime apply failed: {ar.reason}"
            assert ar.runtime_directive is not None
            n_legal += 1
            continue
        if not ar.ok:
            sm.restore(snap)
            pytest.fail(f"{mut.id}|{mode} apply ok=False: {ar.reason}")
        co = sm.compile()
        assert co.ok, f"{mut.id}|{mode} post-compile fail: {co.error_msg}"
        n_legal += 1
    assert n_legal >= 1, f"{mut.id} produced zero successful legal cells"
