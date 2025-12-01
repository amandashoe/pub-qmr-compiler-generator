import json
from structures import Operation, Schedule, Location, BoundaryType, Patch, PatchType, Architecture, Step
from validate_schedule import validate_schedule, export_to_lassynth
from pprint import pprint
from fractions import Fraction
import cirq
import sys
import argparse


def index_to_coord(index, num_cols, num_rows):
    row = index // num_cols
    col = index % num_cols
    x = col
    y = num_rows - 1 - row  # Flip y-axis
    return (x, y)

def parse_schedule(file_path):
    solver_output = json.load(open(file_path))

    width = solver_output["width"]
    height = solver_output["height"]

    initial_patches = []

    ancilla_map = {}

    initial_map = {}

    for qubit, location in solver_output["steps"][0]["map"].items():
        patch_map = solver_output["steps"][0]["patch_map"]
        x, y = index_to_coord(location, width, height)
        initial_map[cirq.NamedQubit(f"q_{qubit}")] = cirq.GridQubit(x, y)
        # print(location, x, y, width, height)
        top_bottom = BoundaryType.X if patch_map[qubit]["top_bottom"] == "X" else BoundaryType.Z if patch_map[qubit]["top_bottom"] == "Z" else None
        left_right = BoundaryType.X if top_bottom == BoundaryType.Z else BoundaryType.Z if top_bottom == BoundaryType.X else None
        patch_type = PatchType.ALGORITHM if patch_map[qubit]["patch_type"] == "ALGORITHM" else PatchType.ANCILLA if patch_map[qubit]["patch_type"] == "ANCILLA" else PatchType.MAGIC_T
        initial_patches.append(Patch(f"{qubit}", Location(x, y), top_bottom, left_right, patch_type))

    arch = Architecture(width, height, initial_patches)

    steps = []

    time = 0

    gates_added = set()
    gates_to_add: list[list[Operation]] = []

    for step in solver_output["steps"]:
        operations = []

        for gate in gates_to_add:
            if len(gate) > 0 and time.is_integer():
                if gate[0][0] == time:
                    operations.append(gate.pop(0)[1])

        for gate in step["implemented_gates"]:
            if gate["gate"]["id"] not in gates_added:
                gates_added.add(gate["gate"]["id"])
            else:
                continue
            if gate["gate"]["operation"] == "Move":
                assert gate["gate"]["qubits"][0] != gate["gate"]["qubits"][1]
                name = "MOVE"
                qubits = [f"{gate['gate']['qubits'][0]}", f"{gate['gate']['qubits'][1]}"]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]][1:-1] if len(gate["implementation"]["tree"]) > 2 else []
                operations.append(Operation(name, qubits, routing_qubits, []))
            elif gate["gate"]["operation"] == "MoveRotate":
                assert gate["gate"]["qubits"][0] != gate["gate"]["qubits"][1]
                name = "MOVE_ROTATE"
                qubits = [f"{gate['gate']['qubits'][0]}", f"{gate['gate']['qubits'][1]}"]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]][1:-1] if len(gate["implementation"]["tree"]) > 2 else []
                operations.append(Operation(name, qubits, routing_qubits, []))
            elif gate["gate"]["operation"] == "PauliMeasurement":
                name = "".join(gate["gate"]["operation"]["PauliMeasurement"]["axis"]).replace("PauliI", "").replace("Pauli", "")
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                operations.append(Operation(name, qubits, routing_qubits, []))
            elif gate["gate"]["operation"] == "CX":
                name = "CX"
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                operations.append(Operation(name, qubits, routing_qubits, []))
            elif gate["gate"]["operation"] == "S":
                name = "S"
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                operations.append(Operation(name, qubits, routing_qubits, []))
            elif gate["gate"]["operation"] == "H":
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                operations.append(Operation("H", qubits, [], []))
                operations.append(Operation("MOVE_ROTATE", [qubits[0], routing_qubits[1]], [routing_qubits[0]], []))

                gates_to_add.append([(time + 1, Operation("MOVE", [qubits[0], routing_qubits[0]], [], [])), (time + 2, Operation("MOVE", [qubits[0], routing_qubits[1]], [], []))])
            elif gate["gate"]["operation"] == "HLitinski":
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                operations.append(Operation("H", qubits, [], []))
                operations.append(Operation("LITINSKI_ROTATE", [qubits[0]], [routing_qubits[0]], []))
            elif gate["gate"]["operation"] == "CultivateTZ":
                name = "CULTIVATE_T_Z"
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = []
                operations.append(Operation(name, qubits, routing_qubits, []))
            elif gate["gate"]["operation"] == "T":
                name = "T"
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                magic_state = [routing_qubits[-1]]
                routing_qubits.pop()
                operations.append(Operation(name, qubits, routing_qubits, magic_state))
            elif gate["gate"]["operation"] == "ResetToAncilla":
                name = "RESET_TO_ANCILLA"
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                operations.append(Operation(name, qubits, [], []))
            elif gate["gate"]["operation"] == "LitinskiRotate":
                name = "LITINSKI_ROTATE"
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = [[[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]][1]]
                operations.append(Operation(name, qubits, routing_qubits, []))
            elif gate["gate"]["operation"] == "TComposite":
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                magic_state = [routing_qubits[-1]]
                routing_qubits.pop()
                # Cultivate
                name = "CULTIVATE_T"
                operations.append(Operation(name, qubits, [], magic_state))
                # Then T
                name = "T"
                gates_to_add.append([(time + 6, Operation(name, qubits, routing_qubits, magic_state))])
            elif gate["gate"]["operation"] == "Walk":
                name = "WALK"
                qubits = [f"{i}" for i in gate["gate"]["qubits"]]
                routing_qubits = [[k for k, v in step["map"].items() if v == i][0] for i in gate["implementation"]["tree"]]
                operations.append(Operation(name, qubits, routing_qubits, []))
            else:
                raise RuntimeError(f"todo: {gate['gate']['operation']}")

        steps.append(Step(time, operations))
        time += Fraction(1, 2)

    while not all([len(g) == 0 for g in gates_to_add]):
        if not time.is_integer():
            time += Fraction(1, 2)
        operations = []

        for gate in gates_to_add:
            if len(gate) > 0:
                operations.append(gate.pop(0))

        steps.append(Step(time, operations))
        time += 1

    schedule = Schedule(arch, steps, None)
    return {
        "initial_map": initial_map,
        "schedule": schedule
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("solution_path")
    parser.add_argument("-e", "--export_gltf", action="store_true")
    args = parser.parse_args()

    parsed_schedule = parse_schedule(args.solution_path)
    validation_state = validate_schedule(parsed_schedule["schedule"])
    print(validation_state.final_schedule.total_time)
    print("Valid!")

    if args.export_gltf:
        print("Exporting pipe diagram gltf...")
        lassynth_solution = export_to_lassynth(validation_state.final_schedule, validation_state)
        lassynth_solution = lassynth_solution.after_default_optimizations()
        lassynth_solution.to_3d_model_gltf(args.solution_path.replace(".json", ".gltf"), attach_axes=True, tube_len=2)
