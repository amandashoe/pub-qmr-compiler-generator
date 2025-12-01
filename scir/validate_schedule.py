import rustworkx as rx
import json
from pprint import pprint
from collections import defaultdict
from copy import copy
from enum import Enum
from typing import Optional
from lassynth import LatticeSurgerySolution
from dataclasses import dataclass, replace
import re
from fractions import Fraction
import math
from structures import Operation, Schedule, Location, BoundaryType, Patch, PatchType, Architecture, Step
import examples


OPERATION_COSTS = {
    "S": Fraction(3, 2),
    "H": 0,
    "CX": 2,
    "T": 2, # CX to inject
    "CULTIVATE_T_X": 6,
    "CULTIVATE_T_Z": 6,
    "CULTIVATE_T": 6,
    "MOVE": 1,
    "MOVE_ROTATE": 1,
    "Y_INIT": Fraction(1, 2),
    "Y_MEAS": Fraction(1, 2),
    "RESET_TO_ANCILLA": 1,
    "LITINSKI_ROTATE": 3,
    "WALK": 2
}


def get_op_cost(operation_name: str) -> Fraction:
    if re.match(r"^(X+|Z+)(_T)?$", operation_name):
        return 1
    else:
        return OPERATION_COSTS[operation_name]


def get_volume(op: Operation) -> Fraction:
    if op.name == "S":
        return Fraction(5, 2)
    elif op.name == "H":
        return 0
    elif op.name == "CULTIVATE_T_Z":
        return 6
    elif op.name == "CX":
        return (2 * len(op.routing_qubits)) + 4
    elif op.name == "T":
        return (2 * len(op.routing_qubits)) + 4
    elif op.name == "MOVE" or op.name == "MOVE_ROTATE":
        return len(op.routing_qubits) + 2
    else:
        raise RuntimeError(f"unimplemented op for volume calculation: {op.name}")


class ValidationState:
    def __init__(self, schedule: Schedule):
        self.graph: rx.PyGraph = rx.generators.grid_graph(schedule.arch.width, schedule.arch.height)
        for i in self.graph.node_indexes():
            self.graph[i] = i
        self.loc_to_node_idx: dict[Location, int] = {Location(x, y): (x * schedule.arch.height) + y for x in range(schedule.arch.width) for y in range(schedule.arch.height)}
        self.node_idx_to_loc: dict[int, Location] = {v: k for k, v in self.loc_to_node_idx.items()}
        self.patches: dict[int, dict[str, Patch]] = defaultdict(dict)
        self.patches[0] = {p.name: replace(p) for p in schedule.arch.initial_patches}
        self.used_qubits: dict[int, set[str]] = defaultdict(set)
        self.lassynth = {}
        self.final_schedule = None


def get_patches(patches: list[Patch]) -> dict[str, Patch]:
    return {p.name: p for p in patches}


def validate_operation(operation: Operation, step: Step, validation_state: ValidationState):
    # check qubits, routing_qubits, magic_state_qubits all right types
    # check length of lists for each op
    # TODO
    if operation.name not in ["S", "Y_INIT"]:
        assert step.start_time.is_integer(), "operation not allowed to start at non-integer time"

    match operation.name:
        case "CX" | "T":
            # check boundary constraints (control X and target Z)
            control_patch = validation_state.patches[step.start_time][operation.qubits[0]]
            control_anc_patch = validation_state.patches[step.start_time][operation.routing_qubits[0]]

            if control_patch.top_bottom == BoundaryType.X:
                assert control_patch.location.x == control_anc_patch.location.x, "ancilla not adjacent to X boundary of control (vertical)"
            else:
                assert control_patch.left_right == BoundaryType.X, "patch should have alternating boundaries"
                assert control_patch.location.y == control_anc_patch.location.y, "ancilla not adjacent to X boundary of control (horizontal)"

            if operation.name == "CX":
                target_patch = validation_state.patches[step.start_time][operation.qubits[1]]
            else:
                target_patch = validation_state.patches[step.start_time][operation.magic_state_qubits[0]]
                assert target_patch.patch_type == PatchType.MAGIC_T, "must have magic state to inject"
            target_anc_patch = validation_state.patches[step.start_time][operation.routing_qubits[-1]]

            if target_patch.top_bottom == BoundaryType.Z:
                assert target_patch.location.x == target_anc_patch.location.x, "ancilla not adjacent to Z boundary of target (vertical)"
            else:
                assert target_patch.left_right == BoundaryType.Z, "patch should have alternating boundaries"
                assert target_patch.location.y == target_anc_patch.location.y, "ancilla not adjacent to Z boundary of target (horizontal)"

            # check path from control to target is connected
            has_bend = count_bends([control_patch] + [validation_state.patches[step.start_time][x] for x in operation.routing_qubits] + [target_patch]) > 0
            for i in range(1, len(operation.routing_qubits) - 1):
                prev = validation_state.patches[step.start_time][operation.routing_qubits[i - 1]]
                curr = validation_state.patches[step.start_time][operation.routing_qubits[i]]
                next = validation_state.patches[step.start_time][operation.routing_qubits[i + 1]]
                assert validation_state.graph.has_edge(
                    validation_state.loc_to_node_idx[prev.location],
                    validation_state.loc_to_node_idx[curr.location],
                ), "path not connected"
                assert validation_state.graph.has_edge(
                    validation_state.loc_to_node_idx[curr.location],
                    validation_state.loc_to_node_idx[next.location],
                ), "path not connected"

            assert validation_state.graph.has_edge(
                validation_state.loc_to_node_idx[control_patch.location],
                validation_state.loc_to_node_idx[control_anc_patch.location],
            ), "path not connected"
            assert validation_state.graph.has_edge(
                validation_state.loc_to_node_idx[target_anc_patch.location],
                validation_state.loc_to_node_idx[target_patch.location],
            ), "path not connected"

            # check path from control to target has at least one bend
            # only need to check when boundaries are not aligned and in same row/column
            if control_patch.top_bottom != target_patch.top_bottom and (control_patch.location.x == target_patch.location.x or control_patch.location.y == target_patch.location.y):
                assert has_bend, "routing path needs at least one bend (path from control to target needs at least one bend)"

            # mark qubits as occupied for duration of operation
            CX_COST = 2  # TODO: eventually comes from operation decl
            for t in [step.start_time + Fraction(1 * x, 2) for x in range(1, CX_COST * 2)]:
                qubits_involved = set(operation.qubits + operation.routing_qubits + operation.magic_state_qubits)
                assert validation_state.used_qubits[t].isdisjoint(qubits_involved), "patch used by multiple ops in same time step (update clash)"
                validation_state.used_qubits[t].update(qubits_involved)
            
            if operation.name == "T":
                # future changes to state after this operation
                T_COST = CX_COST
                assert target_patch.name not in validation_state.patches[step.start_time + T_COST].keys(), "state update clash"

                validation_state.patches[step.start_time + T_COST].update(
                    {
                        target_patch.name: Patch(target_patch.name, target_patch.location, None, None, PatchType.ANCILLA),
                    }
                )
        case "S":
            # check boundary adjacent with ancilla is X
            patch = validation_state.patches[step.start_time][operation.qubits[0]]
            anc_patch = validation_state.patches[step.start_time][operation.routing_qubits[0]]

            assert validation_state.graph.has_edge(
                validation_state.loc_to_node_idx[patch.location],
                validation_state.loc_to_node_idx[anc_patch.location],
            ), "ancilla not adjacent"

            if patch.top_bottom == BoundaryType.X:
                assert patch.location.x == anc_patch.location.x, "ancilla not adjacent to X boundary of control (vertical)"
            else:
                assert patch.left_right == BoundaryType.X, "patch should have alternating boundaries"
                assert patch.location.y == anc_patch.location.y, "ancilla not adjacent to X boundary of control (horizontal)"

            # mark qubits as occupied for duration of operation
            S_COST = Fraction(3, 2)
            for t in [step.start_time + Fraction(1 * x, 2) for x in range(1, int(S_COST * 2))]:
                qubits_involved = set(operation.qubits + operation.routing_qubits + operation.magic_state_qubits)
                assert validation_state.used_qubits[t].isdisjoint(qubits_involved), "patch used by multiple ops in same time step (update clash)"
                validation_state.used_qubits[t].update(qubits_involved)
        case "RESET_TO_ANCILLA":
            patch = validation_state.patches[step.start_time][operation.qubits[0]]
            # mark qubits as occupied for duration of operation
            RESET_TO_ANCILLA_COST = Fraction(2, 2)
            for t in [step.start_time + Fraction(1 * x, 2) for x in range(1, int(RESET_TO_ANCILLA_COST * 2))]:
                qubits_involved = set(operation.qubits + operation.routing_qubits + operation.magic_state_qubits)
                assert validation_state.used_qubits[t].isdisjoint(qubits_involved), "patch used by multiple ops in same time step (update clash)"
                validation_state.used_qubits[t].update(qubits_involved)

            # future changes to state after this operation
            assert patch.name not in validation_state.patches[step.start_time + RESET_TO_ANCILLA_COST].keys(), "state update clash"

            validation_state.patches[step.start_time + RESET_TO_ANCILLA_COST].update(
                {
                    patch.name: Patch(patch.name, patch.location, None, None, PatchType.ANCILLA),
                }
            )
        case "MOVE" | "MOVE_ROTATE":
            # target must be ancilla type
            orig_patch = validation_state.patches[step.start_time][operation.qubits[0]]
            dest_patch = validation_state.patches[step.start_time][operation.qubits[1]]

            assert dest_patch.patch_type == PatchType.ANCILLA, "move destination must be ancilla patch"

            routing_qubits = [validation_state.patches[step.start_time][q] for q in operation.routing_qubits]

            if operation.name == "MOVE":
                assert count_bends([orig_patch] + routing_qubits + [dest_patch]) % 2 == 0, "move requires even bends from source to dest"
            elif operation.name == "MOVE_ROTATE":
                assert count_bends([orig_patch] + routing_qubits + [dest_patch]) % 2 == 1, "move_rotate requires odd bends from source to dest"
            else:
                raise RuntimeError("should be unreachable")

            # path must be connected
            orig_anc = operation.routing_qubits[0] if len(routing_qubits) > 0 else dest_patch.name
            assert validation_state.graph.has_edge(
                validation_state.loc_to_node_idx[orig_patch.location],
                validation_state.loc_to_node_idx[validation_state.patches[step.start_time][orig_anc].location],
            ), "path not connected"
            if len(routing_qubits) > 0:
                for i in range(1, len(operation.routing_qubits) - 1):
                    prev = validation_state.patches[step.start_time][operation.routing_qubits[i - 1]]
                    curr = validation_state.patches[step.start_time][operation.routing_qubits[i]]
                    next = validation_state.patches[step.start_time][operation.routing_qubits[i + 1]]
                    assert validation_state.graph.has_edge(
                        validation_state.loc_to_node_idx[prev.location],
                        validation_state.loc_to_node_idx[curr.location],
                    ), "path not connected"
                    assert validation_state.graph.has_edge(
                        validation_state.loc_to_node_idx[curr.location],
                        validation_state.loc_to_node_idx[next.location],
                    ), "path not connected"

                assert validation_state.graph.has_edge(
                    validation_state.loc_to_node_idx[validation_state.patches[step.start_time][operation.routing_qubits[-1]].location],
                    validation_state.loc_to_node_idx[dest_patch.location],
                ), "path not connected"

            # mark qubits as occupied for duration of operation
            MOVE_COST = 1
            for t in [step.start_time + Fraction(1 * x, 2) for x in range(1, MOVE_COST * 2)]:
                qubits_involved = set(operation.qubits + operation.routing_qubits + operation.magic_state_qubits)
                assert validation_state.used_qubits[t].isdisjoint(qubits_involved), "patch used by multiple ops in same time step (update clash)"
                validation_state.used_qubits[t].update(qubits_involved)

            # future changes to state after this operation
            assert orig_patch.name not in validation_state.patches[step.start_time + MOVE_COST].keys(), "state update clash"
            assert dest_patch.name not in validation_state.patches[step.start_time + MOVE_COST].keys(), "state update clash"

            new_patch_tb = orig_patch.top_bottom if operation.name == "MOVE" else orig_patch.left_right
            new_patch_lr = orig_patch.left_right if operation.name == "MOVE" else orig_patch.top_bottom

            validation_state.patches[step.start_time + MOVE_COST].update(
                {
                    orig_patch.name: Patch(orig_patch.name, dest_patch.location, new_patch_tb, new_patch_lr, orig_patch.patch_type),
                    dest_patch.name: Patch(dest_patch.name, orig_patch.location, dest_patch.top_bottom, dest_patch.left_right, dest_patch.patch_type),
                }
            )
        case "WALK":
            routing_qubits = [validation_state.patches[step.start_time][q] for q in operation.routing_qubits]
            assert routing_qubits[0].patch_type == PatchType.ANCILLA, "walk destination must be ancilla patch"

            for i in range(1, len(routing_qubits) - 1):
                prev = routing_qubits[i - 1]
                curr = routing_qubits[i]
                next = routing_qubits[i + 1]
                assert validation_state.graph.has_edge(
                    validation_state.loc_to_node_idx[prev.location],
                    validation_state.loc_to_node_idx[curr.location],
                ), "path not connected"
                assert validation_state.graph.has_edge(
                    validation_state.loc_to_node_idx[curr.location],
                    validation_state.loc_to_node_idx[next.location],
                ), "path not connected"

            # mark qubits as occupied for duration of operation
            cost = get_op_cost("WALK")
            for t in [step.start_time + Fraction(1 * x, 2) for x in range(1, cost * 2)]:
                qubits_involved = set(operation.qubits + operation.routing_qubits + operation.magic_state_qubits)
                assert validation_state.used_qubits[t].isdisjoint(qubits_involved), "patch used by multiple ops in same time step (update clash)"
                validation_state.used_qubits[t].update(qubits_involved)

            # future changes to state after this operation
            for i in range(1, len(routing_qubits)):
                dest = routing_qubits[i - 1]
                src = routing_qubits[i]
            
                assert src.name not in validation_state.patches[step.start_time + cost].keys(), "state update clash"

                validation_state.patches[step.start_time + cost].update(
                    {
                        src.name: Patch(src.name, dest.location, src.top_bottom, src.left_right, src.patch_type),
                    }
                )
            ancilla = routing_qubits[0]
            validation_state.patches[step.start_time + cost].update(
                    {
                        ancilla.name: Patch(ancilla.name, routing_qubits[-1].location, ancilla.top_bottom, ancilla.left_right, ancilla.patch_type),
                    }
                )
        case "LITINSKI_ROTATE":
            patch = validation_state.patches[step.start_time][operation.qubits[0]]
            validation_state.patches[step.start_time + OPERATION_COSTS["LITINSKI_ROTATE"]].update(
                {
                    patch.name: Patch(patch.name, patch.location, patch.left_right, patch.top_bottom, patch.patch_type),
                }
            )
        case "H":
            patch = validation_state.patches[step.start_time][operation.qubits[0]]
            validation_state.patches[step.start_time].update(
                {
                    patch.name: Patch(patch.name, patch.location, patch.left_right, patch.top_bottom, patch.patch_type),
                }
            )
        case "CULTIVATE_T_X" | "CULTIVATE_T_Z" | "CULTIVATE_T":
            # check ancilla patch
            if operation.name == "CULTIVATE_T":
                patch = validation_state.patches[step.start_time][operation.magic_state_qubits[0]]
                future_T_patch = validation_state.patches[step.start_time][operation.qubits[0]]
                new_patch_tb = future_T_patch.top_bottom
                new_patch_lr = BoundaryType.X if new_patch_tb == BoundaryType.Z else BoundaryType.Z
            else:
                patch = validation_state.patches[step.start_time][operation.qubits[0]]
                # TODO: is this the correct interpretation of boundaries?
                new_patch_tb = BoundaryType.X if operation.name == "CULTIVATE_T_X" else BoundaryType.Z
                new_patch_lr = BoundaryType.X if new_patch_tb == BoundaryType.Z else BoundaryType.Z
            assert patch.patch_type == PatchType.ANCILLA, "must cultivate on ancilla"

            # mark qubits as occupied for duration of operation
            CULTIVATE_T_COST = 6
            for t in [step.start_time + Fraction(1 * x, 2) for x in range(1, CULTIVATE_T_COST * 2)]:
                qubits_involved = operation.routing_qubits + operation.magic_state_qubits
                if operation.name != "CULTIVATE_T":
                    qubits_involved += operation.qubits
                qubits_involved = set(qubits_involved)
                assert validation_state.used_qubits[t].isdisjoint(qubits_involved), "patch used by multiple ops in same time step (update clash)"
                validation_state.used_qubits[t].update(qubits_involved)

            # future changes to state after this operation
            assert patch.name not in validation_state.patches[step.start_time + CULTIVATE_T_COST].keys(), "state update clash"

            validation_state.patches[step.start_time + CULTIVATE_T_COST].update(
                {
                    patch.name: Patch(patch.name, patch.location, new_patch_tb, new_patch_lr, PatchType.MAGIC_T),
                }
            )
        case "Y_INIT" | "Y_MEAS":
            pass
            # TODO: any constraints?
        case _:
            if re.match(r"^(X+|Z+)(_T)?$", operation.name):
                qubits = [validation_state.patches[step.start_time][q] for q in operation.qubits]
                routing_qubits = [validation_state.patches[step.start_time][q] for q in operation.routing_qubits]

                if operation.name.endswith("_T"):
                    magic_patch = qubits[-1]
                    assert magic_patch.patch_type == PatchType.MAGIC_T, "must have magic state to inject"

                qubit_node_ids = [validation_state.loc_to_node_idx[p.location] for p in qubits]
                anc_node_ids = [validation_state.loc_to_node_idx[p.location] for p in routing_qubits]
                node_ids = anc_node_ids + qubit_node_ids
                subgraph = validation_state.graph.subgraph(node_ids, preserve_attrs=True)

                # check everything is connected
                assert rx.is_connected(subgraph), "patches not connected"

                # check no degree 1 ancilla TODO

                old_id_to_subgraph_id = {}
                for n in subgraph.node_indexes():
                    old_id_to_subgraph_id[subgraph[n]] = n

                # check qubits connected to their respective ancilla entrypoint on correct boundary
                boundary = BoundaryType.X if operation.name[0] == "Z" else BoundaryType.Z
                if len(qubits) == 2 and len(routing_qubits) == 0:
                    assert subgraph.has_edge(old_id_to_subgraph_id[qubit_node_ids[0]], old_id_to_subgraph_id[qubit_node_ids[1]]), "missing edge between ZZ/XX args"

                    if qubits[0].patch_type != PatchType.ANCILLA:
                        if qubits[0].top_bottom == boundary:
                            assert qubits[0].location.x == qubits[1].location.x, f"ZZ/XX args not adjacent to {boundary} boundary of target (vertical)"
                        else:
                            assert qubits[0].left_right == boundary, "patch should have alternating boundaries"
                            assert qubits[0].location.y == qubits[1].location.y, f"ZZ/XX args not adjacent to {boundary} boundary of target (horizontal)"

                    if qubits[1].patch_type != PatchType.ANCILLA:
                        if qubits[1].top_bottom == boundary:
                            assert qubits[0].location.x == qubits[1].location.x, f"ZZ/XX args not adjacent to {boundary} boundary of target (vertical)"
                        else:
                            assert qubits[1].left_right == boundary, "patch should have alternating boundaries"
                            assert qubits[0].location.y == qubits[1].location.y, f"ZZ/XX args not adjacent to {boundary} boundary of target (horizontal)"
                else:
                    for i in range(len(qubit_node_ids)):
                        assert subgraph.has_edge(old_id_to_subgraph_id[qubit_node_ids[i]], old_id_to_subgraph_id[anc_node_ids[i]]), "missing edge between qubit and its ancilla entrypoint"

                        if qubits[i].top_bottom == boundary:
                            assert routing_qubits[i].location.x == qubits[i].location.x, f"ZZ/XX args not adjacent to {boundary} boundary of target (vertical)"
                        else:
                            assert qubits[i].left_right == boundary, "patch should have alternating boundaries"
                            assert routing_qubits[i].location.y == qubits[i].location.y, f"ZZ/XX args not adjacent to {boundary} boundary of target (horizontal)"

                # mark qubits as occupied for duration of operation
                MBP_COST = 1  # TODO: eventually comes from operation decl
                for t in [step.start_time + Fraction(1 * x, 2) for x in range(1, MBP_COST * 2)]:
                    qubits_involved = set(operation.qubits + operation.routing_qubits + operation.magic_state_qubits)
                    assert validation_state.used_qubits[t].isdisjoint(qubits_involved), "patch used by multiple ops in same time step (update clash)"
                    validation_state.used_qubits[t].update(qubits_involved)

                # future changes to state after this operation
                if operation.name.endswith("_T"):
                    assert magic_patch.name not in validation_state.patches[step.start_time + MBP_COST].keys(), "state update clash"

                    validation_state.patches[step.start_time + MBP_COST].update(
                        {
                            magic_patch.name: Patch(magic_patch.name, magic_patch.location, None, None, PatchType.ANCILLA),
                        }
                    )
            else:
                raise RuntimeError(f"unimplemented operation: {operation.name}")


def validate_step(step: Step, validation_state: ValidationState):
    # TODO: keep track of used_qubits through entire schedule?
    used_qubits = validation_state.used_qubits[step.start_time]

    if step.start_time > 0:
        most_recent_prev_step = max([t for t in validation_state.patches.keys() if t < step.start_time and len(validation_state.patches[t]) == len(validation_state.patches[0])])
        steps_between = sorted([t for t in validation_state.patches.keys() if t > most_recent_prev_step and t < step.start_time])
        full_patches = copy(validation_state.patches[most_recent_prev_step])
        for t in steps_between:
            full_patches.update(validation_state.patches[t])
        full_patches.update(validation_state.patches[step.start_time])
        # TODO: do we need to update the ones in the middle too?
        validation_state.patches[step.start_time] = full_patches

    for operation in step.operations:
        if operation.name not in ["H"]:
            prev_used_qubits = copy(used_qubits)

            # check qubits in op not already in qubit_involved
            qubits = operation.routing_qubits + operation.magic_state_qubits
            if operation.name != "CULTIVATE_T":
                qubits += operation.qubits
            for qubit in qubits:
                assert qubit in validation_state.patches[0], f"unknown patch name: {qubit}"

                # TODO: generalize
                if re.match(r"^(X+|Z+)(_T)?$", operation.name) and qubit in operation.routing_qubits:
                    assert qubit not in prev_used_qubits, "patch used by multiple ops in same time step"
                else:
                    assert qubit not in used_qubits, "patch used by multiple ops in same time step"
                # add qubits used to qubits involved
                used_qubits.add(qubit)
        # check things like boundaries match up and magic state available
        validate_operation(operation, step, validation_state)


def validate_map(patches: list[Patch]):
    map = {p.name: p.location for p in patches}

    # patch names unique
    assert len(map) == len(patches), "duplicate/missing patch name"

    # keys should already be unique
    assert len(set(map.values())) == len(map.values()), "map not injective"


def validate_patch(patch: Patch):
    match patch.patch_type:
        case PatchType.ALGORITHM:
            assert patch.top_bottom != patch.left_right, "adjacent boundaries should be different"
            assert patch.top_bottom in [BoundaryType.X, BoundaryType.Z], "top_bottom boundary should be X or Z"
            assert patch.left_right in [BoundaryType.X, BoundaryType.Z], "left_right boundary should be X or Z"
        case PatchType.ANCILLA:
            assert patch.top_bottom is None and patch.left_right is None, "ancilla patch shouldn't have boundary type"
        case _:
            raise RuntimeError(f"unimplemented patch type: {patch.patch_type}")


def validate_schedule(schedule: Schedule, get_time_only=False) -> ValidationState:
    validation_state = ValidationState(schedule)

    validate_map(schedule.arch.initial_patches)

    for patch in schedule.arch.initial_patches:
        validate_patch(patch)

    times = set()
    for step in schedule.steps:
        assert step.start_time not in times, "must have unique step start times"
        times.add(step.start_time)

    steps = {s.start_time: s for s in schedule.steps}
    # volume_by_op_dict = {}
    for step in schedule.steps:
        for operation in step.operations:
            # volume_by_op_dict[operation.name] = volume_by_op_dict.get(operation.name, 0) + get_volume(operation)
            end_time = step.start_time + get_op_cost(operation.name)
            if end_time not in times:
                steps[end_time] = Step(end_time, [])
                # check all ops finished within specified time
                if schedule.total_time is not None:
                    assert end_time <= schedule.total_time, "not possible to finish all ops within specified time"
                times.add(end_time)
    finish_time = max(times)
    total_time = finish_time if schedule.total_time is None else schedule.total_time
    # total_volume = schedule.arch.width * schedule.arch.height * finish_time
    # used_volume = float(sum(volume_by_op_dict.values()))
    print(f"Schedule needs at least {finish_time}d time to finish.")

    schedule = Schedule(schedule.arch, [steps[i] for i in sorted(steps.keys())], total_time)
    validation_state.final_schedule = schedule
    if get_time_only:
        return validation_state

    for step in schedule.steps:
        validate_step(step, validation_state)

    # update to final state
    validate_step(Step(schedule.total_time, []), validation_state)

    count = 0
    for p in schedule.arch.initial_patches:
        if p.patch_type == PatchType.ALGORITHM and (p.location != validation_state.patches[finish_time][p.name].location or p.top_bottom != validation_state.patches[finish_time][p.name].top_bottom):
            count += 1
    print(f"{count}/{len([x for x in schedule.arch.initial_patches if x.patch_type == PatchType.ALGORITHM])} alg patches not restored to initial location/orientation.")
    return validation_state


def count_bends(routing_qubits: list[Patch]) -> int:
    num_bends = 0
    for i in range(1, len(routing_qubits) - 1):
        prev_x, prev_y = (
            routing_qubits[i - 1].location.x,
            routing_qubits[i - 1].location.y,
        )
        next_x, next_y = (
            routing_qubits[i + 1].location.x,
            routing_qubits[i + 1].location.y,
        )

        if prev_x != next_x and prev_y != next_y:
            num_bends += 1
    return num_bends


def find_first_corner(path: list[Patch]) -> Patch:
    for i in range(1, len(path) - 1):
        prev_x, prev_y = (
            path[i - 1].location.x,
            path[i - 1].location.y,
        )
        next_x, next_y = (
            path[i + 1].location.x,
            path[i + 1].location.y,
        )

        if prev_x != next_x and prev_y != next_y:
            return path[i]


def export_to_lassynth(schedule: Schedule, validation_state: ValidationState) -> LatticeSurgerySolution:
    lassynth_solution = {}

    initial_algorithm_qubits: list[Patch] = [p for p in schedule.arch.initial_patches if p.patch_type == PatchType.ALGORITHM]

    lassynth_solution["n_i"] = schedule.arch.width
    lassynth_solution["n_j"] = schedule.arch.height
    lassynth_solution["n_k"] = int(math.ceil(schedule.total_time) + 1)
    lassynth_solution["n_p"] = len(initial_algorithm_qubits) * 2
    lassynth_solution["n_s"] = 0

    lassynth_solution["optional"] = {"t_injections": []}

    lassynth_solution["ports"] = []
    for p in initial_algorithm_qubits:
        lassynth_solution["ports"].append(
            {
                "i": p.location.x,
                "j": p.location.y,
                "k": 0,
                "d": "K",
                "e": "-",
                "c": int(p.top_bottom == BoundaryType.Z),
            }
        )

    last_time_step = max(validation_state.patches)
    final_algorithm_qubits: list[Patch] = [p for p_name, p in validation_state.patches[last_time_step].items() if p.patch_type == PatchType.ALGORITHM]
    for p in final_algorithm_qubits:
        lassynth_solution["ports"].append(
            {
                "i": p.location.x,
                "j": p.location.y,
                "k": lassynth_solution["n_k"] - 1,
                "d": "K",
                "e": "+",
                "c": int(p.top_bottom == BoundaryType.Z),
            }
        )

    lassynth_solution["stabs"] = []

    lassynth_solution["ExistI"] = [[[0 for _ in range(lassynth_solution["n_k"])] for _ in range(lassynth_solution["n_j"])] for _ in range(lassynth_solution["n_i"])]
    lassynth_solution["ExistJ"] = [[[0 for _ in range(lassynth_solution["n_k"])] for _ in range(lassynth_solution["n_j"])] for _ in range(lassynth_solution["n_i"])]
    lassynth_solution["ExistK"] = [[[0 for _ in range(lassynth_solution["n_k"])] for _ in range(lassynth_solution["n_j"])] for _ in range(lassynth_solution["n_i"])]

    lassynth_solution["ColorI"] = [[[0 for _ in range(lassynth_solution["n_k"])] for _ in range(lassynth_solution["n_j"])] for _ in range(lassynth_solution["n_i"])]
    lassynth_solution["ColorJ"] = [[[0 for _ in range(lassynth_solution["n_k"])] for _ in range(lassynth_solution["n_j"])] for _ in range(lassynth_solution["n_i"])]

    lassynth_solution["NodeY"] = [[[0 for _ in range(lassynth_solution["n_k"])] for _ in range(lassynth_solution["n_j"])] for _ in range(lassynth_solution["n_i"])]

    for step in schedule.steps:
        # k pipes for unused qubits
        for p_name, p in validation_state.patches[step.start_time].items():
            if p.patch_type != PatchType.ANCILLA and p_name not in validation_state.used_qubits[step.start_time]:
                future_steps = [s.start_time for s in schedule.steps if s.start_time > step.start_time]
                next_step = min(future_steps) if len(future_steps) > 0 else lassynth_solution["n_k"]
                for t in range(math.ceil(step.start_time), math.ceil(next_step)):
                    lassynth_solution["ExistK"][p.location.x][p.location.y][int(t)] = 1
        step = Step(int(step.start_time) if step.start_time.is_integer() else step.start_time, step.operations)
        for operation in step.operations:
            match operation.name:
                case "CX" | "T":
                    # TODO: assumptions here about duration of CX
                    ancilla_path_k = step.start_time + 2

                    qubits = [validation_state.patches[step.start_time][q] for q in operation.qubits]
                    routing_qubits = [validation_state.patches[step.start_time][q] for q in operation.routing_qubits]
                    magic_qubits = [validation_state.patches[step.start_time][q] for q in operation.magic_state_qubits]

                    if operation.name == "CX":
                        control, target = qubits[0], qubits[1]
                    else:
                        control, target = qubits[0], magic_qubits[0]

                    first_corner_loc = find_first_corner([control] + routing_qubits + [target]).location
                    in_step_1 = False

                    for n in range(len(routing_qubits) - 1):
                        i, j = (
                            routing_qubits[n].location.x,
                            routing_qubits[n].location.y,
                        )
                        next_i, next_j = (
                            routing_qubits[n + 1].location.x,
                            routing_qubits[n + 1].location.y,
                        )

                        if i == first_corner_loc.x and j == first_corner_loc.y:
                            ancilla_path_k -= 1
                            in_step_1 = True

                        if abs(i - next_i) == 1:
                            i_pipe = min(i, next_i)
                            lassynth_solution["ExistI"][i_pipe][j][ancilla_path_k] = 1
                            if in_step_1:
                                lassynth_solution["ColorI"][i_pipe][j][ancilla_path_k] = 1
                        else:
                            j_pipe = min(j, next_j)
                            lassynth_solution["ExistJ"][i][j_pipe][ancilla_path_k] = 1
                            if not in_step_1:
                                lassynth_solution["ColorJ"][i][j_pipe][ancilla_path_k] = 1

                    lassynth_solution["ExistK"][first_corner_loc.x][first_corner_loc.y][step.start_time + 1] = 1

                    lassynth_solution["ExistK"][control.location.x][control.location.y][step.start_time] = 1
                    lassynth_solution["ExistK"][control.location.x][control.location.y][step.start_time + 1] = 1
                    if operation.name == "CX":
                        lassynth_solution["ExistK"][control.location.x][control.location.y][step.start_time + 2] = 1

                    lassynth_solution["ExistK"][target.location.x][target.location.y][step.start_time] = 1
                    lassynth_solution["ExistK"][target.location.x][target.location.y][step.start_time + 1] = 1
                    lassynth_solution["ExistK"][target.location.x][target.location.y][step.start_time + 2] = 1

                    control_anc, target_anc = routing_qubits[0], routing_qubits[-1]
                    if control.location.x == control_anc.location.x:
                        j_pipe = min(control.location.y, control_anc.location.y)
                        lassynth_solution["ExistJ"][control.location.x][j_pipe][step.start_time + 2] = 1
                        lassynth_solution["ColorJ"][control.location.x][j_pipe][step.start_time + 2] = 1
                    else:
                        i_pipe = min(control.location.x, control_anc.location.x)
                        lassynth_solution["ExistI"][i_pipe][control.location.y][step.start_time + 2] = 1

                    if target.location.x == target_anc.location.x:
                        j_pipe = min(target.location.y, target_anc.location.y)
                        lassynth_solution["ExistJ"][target.location.x][j_pipe][step.start_time + 1] = 1
                    else:
                        i_pipe = min(target.location.x, target_anc.location.x)
                        lassynth_solution["ExistI"][i_pipe][target.location.y][step.start_time + 1] = 1
                        lassynth_solution["ColorI"][i_pipe][target.location.y][step.start_time + 1] = 1
                case "S":
                    qubit = validation_state.patches[step.start_time][operation.qubits[0]]
                    anc = validation_state.patches[step.start_time][operation.routing_qubits[0]]

                    if not step.start_time.is_integer():
                        assert (step.start_time + Fraction(3, 2)).is_integer(), "start time must be multiple of 0.5 for now"
                    k = step.start_time + 1 if step.start_time.is_integer() else int(step.start_time + Fraction(3, 2))

                    if qubit.location.x == anc.location.x:
                        j_pipe = min(qubit.location.y, anc.location.y)
                        lassynth_solution["ExistJ"][qubit.location.x][j_pipe][k] = 1
                        # TODO: check
                        lassynth_solution["ColorJ"][qubit.location.x][j_pipe][k] = 1
                    else:
                        i_pipe = min(qubit.location.x, anc.location.x)
                        lassynth_solution["ExistI"][i_pipe][qubit.location.y][k] = 1

                    if step.start_time.is_integer():
                        # Y meas
                        lassynth_solution["NodeY"][anc.location.x][anc.location.y][k + 1] = 1
                        lassynth_solution["ExistK"][anc.location.x][anc.location.y][k] = 1
                    else:
                        # Y init
                        lassynth_solution["NodeY"][anc.location.x][anc.location.y][k - 1] = 1
                        lassynth_solution["ExistK"][anc.location.x][anc.location.y][k - 1] = 1

                    lassynth_solution["ExistK"][qubit.location.x][qubit.location.y][k - 1] = 1
                    lassynth_solution["ExistK"][qubit.location.x][qubit.location.y][k] = 1
                    # lassynth_solution["ExistK"][qubit.location.x][qubit.location.y][k + 1] = 1
                case "RESET_TO_ANCILLA":
                    continue
                case "MOVE" | "MOVE_ROTATE":
                    ancilla_path_k = step.start_time + 1

                    qubits = [validation_state.patches[step.start_time][q] for q in operation.qubits]
                    routing_qubits = [validation_state.patches[step.start_time][q] for q in operation.routing_qubits]

                    orig, dest = qubits[0], qubits[1]

                    orig_anc = routing_qubits[0] if len(routing_qubits) > 0 else dest
                    if orig.location.x == orig_anc.location.x:
                        red_face_up = orig.top_bottom == BoundaryType.X
                        j_pipe = min(orig.location.y, orig_anc.location.y)
                        lassynth_solution["ExistJ"][orig.location.x][j_pipe][ancilla_path_k] = 1
                        if red_face_up:
                            lassynth_solution["ColorJ"][orig.location.x][j_pipe][ancilla_path_k] = 1
                    else:
                        red_face_up = orig.left_right == BoundaryType.X
                        i_pipe = min(orig.location.x, orig_anc.location.x)
                        lassynth_solution["ExistI"][i_pipe][orig.location.y][ancilla_path_k] = 1
                        if not red_face_up:
                            lassynth_solution["ColorI"][i_pipe][orig.location.y][ancilla_path_k] = 1

                    if len(routing_qubits) > 0:
                        dest_anc = routing_qubits[-1]
                        if dest.location.x == dest_anc.location.x:
                            j_pipe = min(dest.location.y, dest_anc.location.y)
                            lassynth_solution["ExistJ"][dest.location.x][j_pipe][ancilla_path_k] = 1
                            if red_face_up:
                                lassynth_solution["ColorJ"][dest.location.x][j_pipe][ancilla_path_k] = 1
                        else:
                            i_pipe = min(dest.location.x, dest_anc.location.x)
                            lassynth_solution["ExistI"][i_pipe][dest.location.y][ancilla_path_k] = 1
                            if not red_face_up:
                                lassynth_solution["ColorI"][i_pipe][dest.location.y][ancilla_path_k] = 1

                        for n in range(len(routing_qubits) - 1):
                            i, j = (
                                routing_qubits[n].location.x,
                                routing_qubits[n].location.y,
                            )
                            next_i, next_j = (
                                routing_qubits[n + 1].location.x,
                                routing_qubits[n + 1].location.y,
                            )
                            if abs(i - next_i) == 1:
                                i_pipe = min(i, next_i)
                                lassynth_solution["ExistI"][i_pipe][j][ancilla_path_k] = 1
                                if not red_face_up:
                                    lassynth_solution["ColorI"][i_pipe][j][ancilla_path_k] = 1
                            else:
                                j_pipe = min(j, next_j)
                                lassynth_solution["ExistJ"][i][j_pipe][ancilla_path_k] = 1
                                if red_face_up:
                                    lassynth_solution["ColorJ"][i][j_pipe][ancilla_path_k] = 1

                    lassynth_solution["ExistK"][orig.location.x][orig.location.y][step.start_time] = 1
                    lassynth_solution["ExistK"][dest.location.x][dest.location.y][ancilla_path_k] = 1
                case "H":
                    patch = validation_state.patches[step.start_time][operation.qubits[0]]
                    lassynth_solution["ExistK"][patch.location.x][patch.location.y][step.start_time] = 1
                case "CULTIVATE_T_X" | "CULTIVATE_T_Z" | "CULTIVATE_T":
                    if operation.name == "CULTIVATE_T":
                        patch = validation_state.patches[step.start_time][operation.magic_state_qubits[0]]
                        future_t_patch = validation_state.patches[step.start_time][operation.qubits[0]]
                        name = "CULTIVATE_T_Z" if future_t_patch.top_bottom == BoundaryType.Z else "CULTIVATE_T_X"
                    else:
                        patch = validation_state.patches[step.start_time][operation.qubits[0]]
                        name = operation.name

                    lassynth_solution["optional"]["t_injections"].append([patch.location.x, patch.location.y, step.start_time + 1])
                    lassynth_solution["ports"].append({"c": int(name == "CULTIVATE_T_Z"), "d": "K", "e": "-", "i": patch.location.x, "j": patch.location.y, "k": step.start_time + 1})
                    lassynth_solution["n_p"] += 1

                    for k in range(step.start_time + 1, step.start_time + get_op_cost(operation.name)):
                        lassynth_solution["ExistK"][patch.location.x][patch.location.y][k] = 1
                case "Y_MEAS" | "Y_INIT":
                    patch = validation_state.patches[step.start_time][operation.qubits[0]]
                    k = step.start_time + 1 if step.start_time.is_integer() else int(step.start_time + Fraction(1, 2))
                    if operation.name == "Y_MEAS":
                        # Y meas
                        lassynth_solution["NodeY"][patch.location.x][patch.location.y][k] = 1
                        lassynth_solution["ExistK"][patch.location.x][patch.location.y][k - 1] = 1
                    else:
                        # Y init
                        lassynth_solution["NodeY"][patch.location.x][patch.location.y][k] = 1
                        lassynth_solution["ExistK"][patch.location.x][patch.location.y][k] = 1
                case _:
                    if re.match(r"^(X+|Z+)(_T)?$", operation.name):
                        qubits = [validation_state.patches[step.start_time][q] for q in operation.qubits]
                        routing_qubits = [validation_state.patches[step.start_time][q] for q in operation.routing_qubits]

                        k = step.start_time + 1

                        anc_node_ids = [validation_state.loc_to_node_idx[p.location] for p in routing_qubits]
                        subgraph = validation_state.graph.subgraph(anc_node_ids, preserve_attrs=True)

                        # add all edges between ancilla
                        for n in subgraph.node_indexes():
                            loc = validation_state.node_idx_to_loc[subgraph[n]]
                            for neighbor in subgraph.neighbors(n):
                                neighbor_loc = validation_state.node_idx_to_loc[subgraph[neighbor]]
                                if loc.x == neighbor_loc.x:
                                    j_pipe = min(loc.y, neighbor_loc.y)
                                    lassynth_solution["ExistJ"][loc.x][j_pipe][k] = 1
                                    if operation.name[0] == "Z":
                                        lassynth_solution["ColorJ"][loc.x][j_pipe][k] = 1
                                else:
                                    i_pipe = min(loc.x, neighbor_loc.x)
                                    lassynth_solution["ExistI"][i_pipe][loc.y][k] = 1
                                    if operation.name[0] == "X":
                                        lassynth_solution["ColorI"][i_pipe][loc.y][k] = 1

                        if len(qubits) == 2 and len(routing_qubits) == 0:
                            q = qubits[0]
                            anc = qubits[1]

                            if q.location.x == anc.location.x:
                                j_pipe = min(q.location.y, anc.location.y)
                                lassynth_solution["ExistJ"][q.location.x][j_pipe][k] = 1
                                if operation.name[0] == "Z":
                                    lassynth_solution["ColorJ"][q.location.x][j_pipe][k] = 1
                            else:
                                i_pipe = min(q.location.x, anc.location.x)
                                lassynth_solution["ExistI"][i_pipe][q.location.y][k] = 1
                                if operation.name[0] == "X":
                                    lassynth_solution["ColorI"][i_pipe][q.location.y][k] = 1
                            for q in qubits:
                                if q.patch_type in [PatchType.ALGORITHM, PatchType.MAGIC_T]:
                                    lassynth_solution["ExistK"][q.location.x][q.location.y][step.start_time] = 1
                                if q.patch_type == PatchType.ALGORITHM:
                                    lassynth_solution["ExistK"][q.location.x][q.location.y][step.start_time + 1] = 1
                        else:
                            for i in range(len(qubits)):
                                q = qubits[i]
                                anc = routing_qubits[i]

                                # add edge for each qubit and its respective ancilla
                                if q.location.x == anc.location.x:
                                    j_pipe = min(q.location.y, anc.location.y)
                                    lassynth_solution["ExistJ"][q.location.x][j_pipe][k] = 1
                                    if operation.name[0] == "Z":
                                        lassynth_solution["ColorJ"][q.location.x][j_pipe][k] = 1
                                else:
                                    i_pipe = min(q.location.x, anc.location.x)
                                    lassynth_solution["ExistI"][i_pipe][q.location.y][k] = 1
                                    if operation.name[0] == "X":
                                        lassynth_solution["ColorI"][i_pipe][q.location.y][k] = 1

                                # add qubit k pipes
                                if q.patch_type in [PatchType.ALGORITHM, PatchType.MAGIC_T]:
                                    lassynth_solution["ExistK"][q.location.x][q.location.y][step.start_time] = 1
                                if q.patch_type == PatchType.ALGORITHM:
                                    lassynth_solution["ExistK"][q.location.x][q.location.y][step.start_time + 1] = 1
                    else:
                        raise RuntimeError(f"unimplemented operation for pipe diagram export: {operation.name}")
    # pprint(lassynth_solution)
    return LatticeSurgerySolution(lassynth_solution).after_color_k_pipes()


if __name__ == "__main__":
    schedule = examples.get_cx_1()

    validation_state = validate_schedule(schedule)

    lassynth_solution = export_to_lassynth(validation_state.final_schedule, validation_state)

    input_dict = {"stabilizers": ["Z.Z.", ".X.X", "X.XX", ".ZZZ"]}

    lassynth_solution.to_3d_model_gltf("pipe_diagrams/cx1.gltf", attach_axes=True, tube_len=0.2)
    lassynth_solution.verify_stabilizers_stimzx(specification=input_dict, print_stabilizers=True)
