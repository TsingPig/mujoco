"""GzFuzz-style high-level mutators (10).

Each mutator either edits the XML AST (lxml) and writes a new file, or leaves
the XML untouched and produces `runtime_directives` for the worker (state
perturbation, solver overrides, ctrl-clamp disable).

All mutators are **side-effect free with respect to the input tree**: they
deepcopy the tree before mutating, so the caller can reuse the original.
"""
from __future__ import annotations

import copy
from typing import Any

from lxml import etree

from .base import BaseMutator, MutationApplyResult
from .param_buckets import Buckets


# ============================================================
# helpers
# ============================================================

def _bodies(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.xpath("//worldbody//body")


def _all_bodies_incl_world(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.xpath("//worldbody | //worldbody//body")


def _geoms(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.xpath("//geom")


def _joints(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.xpath("//joint")


def _actuators(tree: etree._ElementTree) -> list[etree._Element]:
    return tree.xpath("//actuator/*")


def _option(tree: etree._ElementTree) -> etree._Element:
    root = tree.getroot()
    opt = root.find("option")
    if opt is None:
        opt = etree.SubElement(root, "option")
        # Move it to the front for readability (not required by MJCF)
        root.insert(0, opt)
    return opt


def _serialize(tree: etree._ElementTree, out_xml_path: str) -> None:
    tree.write(out_xml_path, pretty_print=True, xml_declaration=False, encoding="utf-8")


def _size_str(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return " ".join(str(float(x)) for x in v)
    return str(float(v))


# ============================================================
# 1. STRUCT_GROW
# ============================================================

class StructGrowMutator(BaseMutator):
    id = "STRUCT_GROW"

    def applicable(self, tree):
        return len(_all_bodies_incl_world(tree)) > 0

    def sample_params(self, tree, rng):
        parents = _all_bodies_incl_world(tree)
        return {
            "parent_idx": rng.randrange(len(parents)),
            "jtype": rng.choice(["hinge", "slide", "ball", "free"]),
            "shape": rng.choice(["sphere", "box", "capsule"]),
            "size_bucket": rng.choice(["small", "medium", "large"]),
            "pos_bucket": rng.randrange(7),  # index into pos bucket
        }

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        parents = _all_bodies_incl_world(tree)
        if not parents:
            return MutationApplyResult(False, reason="no_parent")
        parent = parents[params["parent_idx"] % len(parents)]
        # Generate unique name suffix
        new_name = f"grown_{abs(hash((id(parent), params['jtype']))) % 100000}"
        body = etree.SubElement(parent, "body", name=f"b_{new_name}", pos="0 0 0.1")
        # Free joint can only attach to worldbody children; degrade to hinge otherwise
        jtype = params["jtype"]
        if jtype == "free" and parent.tag != "worldbody":
            jtype = "hinge"
        etree.SubElement(body, "joint", name=f"j_{new_name}", type=jtype)
        size_map = {"small": "0.02", "medium": "0.1", "large": "0.5"}
        etree.SubElement(body, "geom",
                         name=f"g_{new_name}",
                         type=params["shape"],
                         size=size_map[params["size_bucket"]])
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 2. STRUCT_SHRINK
# ============================================================

class StructShrinkMutator(BaseMutator):
    id = "STRUCT_SHRINK"

    def applicable(self, tree):
        return len(_bodies(tree)) > 0

    def sample_params(self, tree, rng):
        bs = _bodies(tree)
        return {"target_idx": rng.randrange(len(bs))}

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        bs = _bodies(tree)
        if not bs:
            return MutationApplyResult(False, reason="no_body")
        b = bs[params["target_idx"] % len(bs)]
        parent = b.getparent()
        if parent is None:
            return MutationApplyResult(False, reason="no_parent")
        parent.remove(b)
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 3. STRUCT_REWIRE
# ============================================================

class StructRewireMutator(BaseMutator):
    id = "STRUCT_REWIRE"

    def applicable(self, tree):
        return len(_bodies(tree)) >= 2

    def sample_params(self, tree, rng):
        n = len(_bodies(tree))
        return {"src_idx": rng.randrange(n), "dst_idx": rng.randrange(n)}

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        bs = _bodies(tree)
        if len(bs) < 2:
            return MutationApplyResult(False, reason="not_enough_bodies")
        src = bs[params["src_idx"] % len(bs)]
        dst = bs[params["dst_idx"] % len(bs)]
        # Avoid src==dst, or dst being in src's subtree (would cycle)
        if src is dst or dst in src.iter():
            return MutationApplyResult(False, reason="invalid_rewire")
        parent = src.getparent()
        if parent is None:
            return MutationApplyResult(False, reason="no_parent")
        parent.remove(src)
        dst.append(src)
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 4. GEOM_PERTURB
# ============================================================

class GeomPerturbMutator(BaseMutator):
    id = "GEOM_PERTURB"

    def applicable(self, tree):
        return len(_geoms(tree)) > 0

    def sample_params(self, tree, rng):
        gs = _geoms(tree)
        return {
            "target_idx": rng.randrange(len(gs)),
            "shape": rng.choice(["sphere", "box", "capsule"]),
            "size_bucket": rng.choice(["small", "medium", "large"]),
            "pos_bucket": rng.randrange(7),
        }

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        gs = _geoms(tree)
        if not gs:
            return MutationApplyResult(False, reason="no_geom")
        g = gs[params["target_idx"] % len(gs)]
        # Don't touch plane geoms (changing their type is rarely meaningful)
        if g.get("type") == "plane":
            return MutationApplyResult(False, reason="plane_skipped")
        size_map = {"small": "0.005", "medium": "0.05", "large": "0.5"}
        g.set("type", params["shape"])
        g.set("size", size_map[params["size_bucket"]])
        # Drop incompatible attrs
        for a in ("fromto",):
            if a in g.attrib:
                del g.attrib[a]
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 5. JOINT_PERTURB
# ============================================================

class JointPerturbMutator(BaseMutator):
    id = "JOINT_PERTURB"

    def applicable(self, tree):
        return len(_joints(tree)) > 0

    def sample_params(self, tree, rng):
        js = _joints(tree)
        return {
            "target_idx": rng.randrange(len(js)),
            "jtype": rng.choice(["hinge", "slide"]),  # ball/free shouldn't be set on existing joints
            "range_bucket": rng.choice(["narrow", "medium", "wide"]),
            "damping_bucket": rng.choice(["low", "medium", "high"]),
        }

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        js = _joints(tree)
        if not js:
            return MutationApplyResult(False, reason="no_joint")
        j = js[params["target_idx"] % len(js)]
        rng_map = {"narrow": "-0.1 0.1", "medium": "-1.5 1.5", "wide": "-3.14 3.14"}
        damp_map = {"low": "0.001", "medium": "0.1", "high": "10.0"}
        j.set("type", params["jtype"])
        j.set("range", rng_map[params["range_bucket"]])
        j.set("limited", "true")
        j.set("damping", damp_map[params["damping_bucket"]])
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 6. INERTIAL_PERTURB
# ============================================================

class InertialPerturbMutator(BaseMutator):
    id = "INERTIAL_PERTURB"

    def applicable(self, tree):
        return len(_bodies(tree)) > 0

    def sample_params(self, tree, rng):
        bs = _bodies(tree)
        return {
            "target_idx": rng.randrange(len(bs)),
            "mass_bucket": rng.choice(["tiny", "normal", "heavy"]),
            "diag_bucket": rng.choice(["tiny", "normal", "heavy"]),
        }

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        bs = _bodies(tree)
        if not bs:
            return MutationApplyResult(False, reason="no_body")
        b = bs[params["target_idx"] % len(bs)]
        mass_map = {"tiny": 1e-3, "normal": 1.0, "heavy": 1e3}
        diag_map = {"tiny": 1e-6, "normal": 1.0, "heavy": 1e3}
        m = mass_map[params["mass_bucket"]]
        d = diag_map[params["diag_bucket"]]
        # Replace existing <inertial>
        for old in b.findall("inertial"):
            b.remove(old)
        etree.SubElement(b, "inertial", pos="0 0 0",
                         mass=str(m),
                         diaginertia=f"{d} {d} {d}")
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 7. CONTACT_PERTURB
# ============================================================

class ContactPerturbMutator(BaseMutator):
    id = "CONTACT_PERTURB"

    def applicable(self, tree):
        return len(_geoms(tree)) > 0

    def sample_params(self, tree, rng):
        gs = _geoms(tree)
        return {
            "target_idx": rng.randrange(len(gs)),
            "friction_bucket": rng.choice(["low", "medium", "high"]),
            "condim": rng.choice([1, 3, 4, 6]),
        }

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        gs = _geoms(tree)
        if not gs:
            return MutationApplyResult(False, reason="no_geom")
        g = gs[params["target_idx"] % len(gs)]
        fr_map = {"low":    "0.1 0.005 0.0001",
                  "medium": "1.0 0.005 0.0001",
                  "high":   "10.0 0.05 0.001"}
        g.set("friction", fr_map[params["friction_bucket"]])
        g.set("condim", str(params["condim"]))
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 8. ACTUATOR_EDIT
# ============================================================

class ActuatorEditMutator(BaseMutator):
    id = "ACTUATOR_EDIT"

    def applicable(self, tree):
        return len(_joints(tree)) > 0

    def sample_params(self, tree, rng):
        js = _joints(tree)
        existing = _actuators(tree)
        ops = ["mut"] if existing else []
        ops.append("add")
        if existing:
            ops.append("del")
        return {
            "op": rng.choice(ops),
            "joint_idx": rng.randrange(len(js)),
            "act_idx": rng.randrange(len(existing)) if existing else 0,
            "ctrlrange_bucket": rng.choice(["tight", "normal", "wide"]),
        }

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        root = tree.getroot()
        actuator_root = root.find("actuator")
        if actuator_root is None:
            actuator_root = etree.SubElement(root, "actuator")
        cr_map = {"tight": "-0.1 0.1", "normal": "-2 2", "wide": "-1000 1000"}
        op = params["op"]
        if op == "add":
            js = _joints(tree)
            if not js:
                return MutationApplyResult(False, reason="no_joint")
            j = js[params["joint_idx"] % len(js)]
            jname = j.get("name")
            if not jname:
                jname = f"j_auto_{params['joint_idx']}"
                j.set("name", jname)
            etree.SubElement(actuator_root, "motor",
                             name=f"act_{abs(hash(jname)) % 100000}",
                             joint=jname,
                             ctrlrange=cr_map[params["ctrlrange_bucket"]])
        elif op == "del":
            acts = list(actuator_root)
            if not acts:
                return MutationApplyResult(False, reason="no_actuator")
            actuator_root.remove(acts[params["act_idx"] % len(acts)])
        elif op == "mut":
            acts = list(actuator_root)
            if not acts:
                return MutationApplyResult(False, reason="no_actuator")
            a = acts[params["act_idx"] % len(acts)]
            a.set("ctrlrange", cr_map[params["ctrlrange_bucket"]])
        _serialize(tree, out_xml_path)
        return MutationApplyResult(True, new_xml_path=out_xml_path)


# ============================================================
# 9. SOLVER_TOGGLE
# ============================================================

# These map to mjtIntegrator / mjtSolver enum values in mujoco
_INTEGRATORS = {"Euler": 0, "RK4": 1, "implicit": 2, "implicitfast": 3}
_SOLVERS = {"PGS": 0, "CG": 1, "Newton": 2}


class SolverToggleMutator(BaseMutator):
    id = "SOLVER_TOGGLE"

    def applicable(self, tree):
        return True

    def sample_params(self, tree, rng):
        return {
            "integrator": rng.choice(list(_INTEGRATORS.keys())),
            "solver": rng.choice(list(_SOLVERS.keys())),
            "iterations": rng.choice([1, 5, 50, 500]),
        }

    def apply(self, tree, params, out_xml_path):
        tree = copy.deepcopy(tree)
        opt = _option(tree)
        opt.set("integrator", params["integrator"])
        opt.set("solver", params["solver"])
        opt.set("iterations", str(params["iterations"]))
        _serialize(tree, out_xml_path)
        return MutationApplyResult(
            True,
            new_xml_path=out_xml_path,
            runtime_directives={
                "solver_override": {
                    "integrator": _INTEGRATORS[params["integrator"]],
                    "solver": _SOLVERS[params["solver"]],
                    "iterations": int(params["iterations"]),
                }
            },
        )


# ============================================================
# 10. STATE_PERTURB (XML untouched, runtime directives only)
# ============================================================

class StatePerturbMutator(BaseMutator):
    id = "STATE_PERTURB"

    def applicable(self, tree):
        return True

    def sample_params(self, tree, rng):
        return {
            "target": rng.choice(["qpos", "qvel", "ctrl"]),
            "scale_bucket": rng.choice(["small", "medium", "large", "nan", "inf"]),
            "first_n": rng.choice([1, 3, 8]),
        }

    def apply(self, tree, params, out_xml_path):
        # XML unchanged; just persist a copy for reproducibility
        tree = copy.deepcopy(tree)
        _serialize(tree, out_xml_path)

        scale_map = {"small": 1e-2, "medium": 1.0, "large": 1e3,
                     "nan": "nan", "inf": "+inf"}
        v = scale_map[params["scale_bucket"]]
        n = int(params["first_n"])
        values = [v] * n  # worker truncates to actual size

        directives: dict[str, Any] = {"state_perturb": {params["target"]: values}}
        # If targeting ctrl with NaN/Inf, force-disable ctrl clamping
        if params["target"] == "ctrl" and params["scale_bucket"] in ("nan", "inf"):
            directives["disable_clamp_ctrl"] = True
        return MutationApplyResult(True, new_xml_path=out_xml_path,
                                   runtime_directives=directives)
