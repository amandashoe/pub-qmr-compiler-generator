from structures import Operation, Schedule, Location, BoundaryType, Patch, PatchType, Architecture, Step
from fractions import Fraction


def get_two_cx_1() -> Schedule:
    width = 3
    height = 4
    initial_patches = [
        Patch("q0", Location(2, 3), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q1", Location(0, 3), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q2", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 3), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 2), None, None, PatchType.ANCILLA),
        Patch("a2", Location(2, 2), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("CX", ["q0", "q1"], ["a2", "a1", "a0"], [])]), Step(3, [Operation("CX", ["q0", "q1"], ["a2", "a1", "a0"], [])])]
    total_time = 5
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_bad_cx_1() -> Schedule:
    width = 3
    height = 4
    initial_patches = [
        Patch("q0", Location(2, 3), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("q1", Location(0, 3), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 3), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 2), None, None, PatchType.ANCILLA),
        Patch("a2", Location(2, 2), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("CX", ["q0", "q1"], ["a0"], [])])]
    total_time = 2
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_cx_1() -> Schedule:
    input_dict = {"stabilizers": ["Z.Z.", ".X.X", "X.XX", ".ZZZ"]}
    width = 4
    height = 4
    initial_patches = [
        Patch("q0", Location(2, 3), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q1", Location(0, 3), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 3), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 2), None, None, PatchType.ANCILLA),
        Patch("a2", Location(1, 1), None, None, PatchType.ANCILLA),
        Patch("a3", Location(1, 0), None, None, PatchType.ANCILLA),
        Patch("a4", Location(2, 0), None, None, PatchType.ANCILLA),
        Patch("a5", Location(2, 1), None, None, PatchType.ANCILLA),
        Patch("a6", Location(2, 2), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("CX", ["q0", "q1"], ["a6", "a5", "a4", "a3", "a2", "a1", "a0"], [])])]
    total_time = 2
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_cx_2() -> Schedule:
    input_dict = {"stabilizers": [".Z.Z", "X.X.", ".XXX", "Z.ZZ"]}
    width = 3
    height = 3
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q1", Location(2, 2), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
        Patch("a1", Location(2, 0), None, None, PatchType.ANCILLA),
        Patch("a2", Location(2, 1), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("CX", ["q1", "q0"], ["a2", "a1", "a0"], [])])]
    total_time = 2
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_xxx_1() -> Schedule:
    input_dict = {"stabilizers": ["X...XX", "Z.ZZ.Z", ".ZZ.ZZ", "..XXX.", "...XXX", ".X.X.X"]}
    width = 2
    height = 4
    initial_patches = [
        Patch("q0", Location(0, 1), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("q1", Location(0, 3), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("q2", Location(1, 0), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("a0", Location(0, 2), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 1), None, None, PatchType.ANCILLA),
        Patch("a2", Location(1, 2), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("XXX", ["q0", "q1", "q2"], ["a0", "a0", "a1", "a2"], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_zzz_1() -> Schedule:
    input_dict = {"stabilizers": ["Z...ZZ", "X.XX.X", ".XX.XX", "..ZZZ.", "...ZZZ", ".Z.Z.Z"]}
    width = 2
    height = 4
    initial_patches = [
        Patch("q0", Location(0, 1), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q1", Location(0, 3), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q2", Location(1, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(0, 2), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 1), None, None, PatchType.ANCILLA),
        Patch("a2", Location(1, 2), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("ZZZ", ["q0", "q1", "q2"], ["a0", "a0", "a1", "a2"], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_ss_1() -> Schedule:
    input_dict = {"stabilizers": [".X.Y", "Z.Z.", ".Z.Z", "X.Y."]}
    width = 2
    height = 2
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("q1", Location(1, 1), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("S", ["q0"], ["a0"], [])]), Step(Fraction(3, 2), [Operation("S", ["q1"], ["a0"], [])])]
    total_time = Fraction(3)
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_move_1() -> Schedule:
    input_dict = {"stabilizers": ["ZZ", "XY"]}
    width = 4
    height = 1
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
        Patch("a1", Location(2, 0), None, None, PatchType.ANCILLA),
        Patch("a2", Location(3, 0), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("MOVE", ["q0", "a2"], ["a0", "a1"], [])]), Step(1, [Operation("S", ["q0"], ["a1"], [])])]
    total_time = Fraction(5, 2)
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_bad_move_2() -> Schedule:
    input_dict = {"stabilizers": ["ZZ", "XX"]}
    width = 4
    height = 4
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 1), None, None, PatchType.ANCILLA),
        Patch("a2", Location(1, 2), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("MOVE", ["q0", "a2"], ["a0", "a1"], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_move_2() -> Schedule:
    input_dict = {"stabilizers": ["ZZ", "XX"]}
    width = 4
    height = 4
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
        Patch("a1", Location(2, 0), None, None, PatchType.ANCILLA),
        Patch("a2", Location(2, 1), None, None, PatchType.ANCILLA),
        Patch("a3", Location(2, 2), None, None, PatchType.ANCILLA),
        Patch("a4", Location(1, 2), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("MOVE", ["q0", "a4"], ["a0", "a1", "a2", "a3"], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_move_3() -> Schedule:
    input_dict = {"stabilizers": ["ZZ", "XX"]}
    width = 2
    height = 2
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("MOVE", ["q0", "a0"], [], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_move_rotate_1() -> Schedule:
    input_dict = {"stabilizers": ["ZZ", "XX"]}
    width = 2
    height = 2
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 1), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("MOVE_ROTATE", ["q0", "a1"], ["a0"], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_h() -> Schedule:
    input_dict = {"stabilizers": ["ZX", "XZ"]}
    width = 2
    height = 2
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("H", ["q0"], [], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_h_rotate_move_move() -> Schedule:
    input_dict = {"stabilizers": ["ZX", "XZ"]}
    width = 2
    height = 2
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 1), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [
        Step(0, [Operation("H", ["q0"], [], []), Operation("MOVE_ROTATE", ["q0", "a1"], ["a0"], [])]),
        Step(1, [Operation("MOVE", ["q0", "a0"], [], [])]),
        Step(2, [Operation("MOVE", ["q0", "a1"], [], [])]),
    ]
    total_time = 3
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_cult_t_x() -> Schedule:
    width = 2
    height = 2
    initial_patches = [
        Patch("q0", Location(1, 0), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("a0", Location(0, 1), None, None, PatchType.ANCILLA),
        Patch("a1", Location(0, 0), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("CULTIVATE_T_Z", ["a0"], [], [])]), Step(6, [Operation("T", ["q0"], ["a1"], ["a0"])])]
    total_time = 8
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_mbp_t_z() -> Schedule:
    width = 2
    height = 4
    initial_patches = [
        Patch("q0", Location(0, 1), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q1", Location(0, 3), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q2", Location(1, 0), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(0, 2), None, None, PatchType.ANCILLA),
        Patch("a1", Location(1, 1), None, None, PatchType.ANCILLA),
        Patch("a2", Location(1, 2), None, None, PatchType.ANCILLA),
        Patch("a3", Location(1, 3), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("CULTIVATE_T_X", ["a3"], [], [])]), Step(6, [Operation("ZZZ_T", ["q0", "q1", "q2", "a3"], ["a0", "a0", "a1", "a2"], [])])]
    total_time = 7
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_s_from_y_basis_init() -> Schedule:
    input_dict = {"stabilizers": [".X.Y", "Z.Z.", ".Z.Z", "X.Y."]}
    width = 2
    height = 2
    initial_patches = [
        Patch("q0", Location(0, 0), BoundaryType.Z, BoundaryType.X, PatchType.ALGORITHM),
        Patch("q1", Location(1, 1), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("a0", Location(1, 0), None, None, PatchType.ANCILLA),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [
        Step(0, [Operation("ZZ", ["q0", "a0"], [], [])]),
        Step(1, [Operation("Y_MEAS", ["a0"], [], [])]),
        Step(Fraction(3, 2), [Operation("Y_INIT", ["a0"], [], [])]),
        Step(2, [Operation("ZZ", ["q1", "a0"], [], [])]),
    ]
    total_time = Fraction(3)
    schedule = Schedule(arch, steps, total_time)
    return schedule


def get_zz() -> Schedule:
    input_dict = {"stabilizers": [".ZZ.", "Z..Z", "XXXX", "ZZ.."]}
    width = 2
    height = 4
    initial_patches = [
        Patch("q0", Location(0, 1), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
        Patch("q1", Location(0, 2), BoundaryType.X, BoundaryType.Z, PatchType.ALGORITHM),
    ]
    arch = Architecture(width, height, initial_patches)
    steps = [Step(0, [Operation("ZZ", ["q0", "q1"], [], [])])]
    total_time = 1
    schedule = Schedule(arch, steps, total_time)
    return schedule

