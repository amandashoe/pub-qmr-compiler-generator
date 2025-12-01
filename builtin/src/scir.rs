use core::num;
use itertools::Itertools;
use petgraph::Directed;
use petgraph::{graph::NodeIndex, Graph};
use rand::Rng;
use serde_json::{from_value, Value};
use solver::backend;
use solver::structures::GateType::*;
use solver::structures::*;
use solver::utils::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use solver::config::CONFIG;
// const GATE_TYPES: &[&str] = &["Pauli", "CX", "S", "H", "T"];
// const GATE_TYPES: &[&str] = &["Pauli", "CX", "S", "HLitinski", "TComposite"];
#[derive(Hash, PartialEq, Eq, Clone, serde::Serialize, Debug)]
pub struct SCIRGate {
    tree: Vec<Location>,
}
#[derive(Clone)]
struct CustomArch {
    graph: Graph<Location, ()>,
    index_map: HashMap<Location, NodeIndex>,
    magic_state_qubits: Vec<Location>,
    alg_qubits: Vec<Location>,
    anc_qubits: Vec<Location>,
    patches: Vec<Patch>,
    width: usize,
    height: usize,
    initial_map: Option<QubitMap>,
}
#[derive(Clone, serde::Serialize, Debug)]
pub struct SCIRTrans {
    path: Vec<Location>,
    operation: Operation,
    next_step: Step<SCIRGate>,
}
impl GateImplementation for SCIRGate {}
impl SCIRGate {
    pub fn tree(&self) -> Vec<Location> {
        return self.tree.clone();
    }
}

impl Architecture for CustomArch {
    fn locations(&self) -> Vec<Location> {
        self.alg_qubits()
    }
    fn anc_locations(&self) -> Vec<Location> {
        self.anc_qubits()
    }
    fn patches(&self) -> Vec<Patch> {
        self.patches()
    }
    fn initial_map(&self) -> Option<QubitMap> {
        self.initial_map()
    }
    fn graph(&self) -> (Graph<Location, ()>, HashMap<Location, NodeIndex>) {
        return (self.graph.clone(), self.index_map.clone());
    }
    fn dims(&self) -> (usize, usize) {
        return (self.height, self.width);
    }
}

impl CustomArch {
    fn from_file(path: &str) -> Self {
        let file = File::open(path).expect("Opening architecture file");
        let parsed: Value = serde_json::from_reader(file).expect("Parsing architecture file");
        let graph = graph_from_json_entry(parsed["graph"].clone());
        let patches = patches_from_json_entry(parsed["patches"].clone());
        let mut index_map = HashMap::new();
        for ind in graph.node_indices() {
            index_map.insert(graph[ind], ind);
        }
        return CustomArch {
            graph,
            index_map,
            magic_state_qubits: from_value(parsed["magic_state_qubits"].clone()).expect(&format!(
                "Parsing {} field",
                "magic_state_qubits".to_string()
            )),
            alg_qubits: from_value(parsed["alg_qubits"].clone())
                .expect(&format!("Parsing {} field", "alg_qubits".to_string())),
            anc_qubits: from_value(parsed["anc_qubits"].clone())
                .expect(&format!("Parsing {} field", "anc_qubits".to_string())),
            patches,
            width: from_value(parsed["width"].clone())
                .expect(&format!("Parsing {} field", "width".to_string())),
            height: from_value(parsed["height"].clone())
                .expect(&format!("Parsing {} field", "height".to_string())),
            initial_map: initial_map_from_json(parsed["initial_map"].clone()),
        };
    }
    fn contains_edge(&self, edge: (Location, Location)) -> bool {
        self.graph
            .contains_edge(self.index_map[&edge.0], self.index_map[&edge.1])
    }
    pub fn edges(&self) -> Vec<(Location, Location)> {
        let mut edges = Vec::new();
        for edge in self.graph.edge_indices() {
            let (u, v) = self.graph.edge_endpoints(edge).unwrap();
            edges.push((self.graph[u], self.graph[v]));
        }
        return edges;
    }
    pub fn magic_state_qubits(&self) -> Vec<Location> {
        return self.magic_state_qubits.clone();
    }
    pub fn alg_qubits(&self) -> Vec<Location> {
        return self.alg_qubits.clone();
    }
    pub fn anc_qubits(&self) -> Vec<Location> {
        return self.anc_qubits.clone();
    }
    pub fn patches(&self) -> Vec<Patch> {
        return self.patches.clone();
    }
    pub fn initial_map(&self) -> Option<QubitMap> {
        return self.initial_map.clone();
    }
}

impl Transition<SCIRGate, CustomArch> for SCIRTrans {
    fn apply(&self, step: &Step<SCIRGate>) -> Step<SCIRGate> {
        let mut new_step = self.next_step.clone();
        new_step.implemented_gates = self.next_step.implemented_gates.clone();

        let mut rng = rand::thread_rng();
        // TODO: in theory could have duplicates
        let random_number: usize = rng.gen();

        match self.operation {
            Operation::CultivateTX | Operation::CultivateTZ => {
                new_step.implemented_gates.insert(ImplementedGate {
                    gate: Gate {
                        operation: self.operation.clone(),
                        qubits: vec![get_key(&new_step.map, self.path[0]).unwrap()],
                        id: random_number,
                    },
                    implementation: SCIRGate {
                        tree: self.path.clone(),
                    },
                });
            }
            Operation::ResetToAncilla => {
                new_step.implemented_gates.insert(ImplementedGate {
                    gate: Gate {
                        operation: self.operation.clone(),
                        qubits: vec![get_key(&new_step.map, self.path[0]).unwrap()],
                        id: random_number,
                    },
                    implementation: SCIRGate {
                        tree: self.path.clone(),
                    },
                });
            }
            Operation::LitinskiRotate => {
                new_step.implemented_gates.insert(ImplementedGate {
                    gate: Gate {
                        operation: self.operation.clone(),
                        qubits: vec![get_key(&new_step.map, self.path[0]).unwrap()],
                        id: random_number,
                    },
                    implementation: SCIRGate {
                        tree: self.path.clone(),
                    },
                });
            }
            Operation::Move | Operation::MoveRotate => {
                if get_key(&new_step.map, self.path[0]).unwrap()
                    != get_key(&new_step.map, self.path[self.path.len() - 1]).unwrap()
                {
                    new_step.implemented_gates.insert(ImplementedGate {
                        gate: Gate {
                            operation: self.operation.clone(),
                            qubits: vec![
                                get_key(&new_step.map, self.path[0]).unwrap(),
                                get_key(&new_step.map, self.path[self.path.len() - 1]).unwrap(),
                            ],
                            id: random_number,
                        },
                        implementation: SCIRGate {
                            tree: self.path.clone(),
                        },
                    });
                }
            }
            Operation::Walk => {
                new_step.implemented_gates.insert(ImplementedGate {
                        gate: Gate {
                            operation: self.operation.clone(),
                            qubits: vec![],
                            id: random_number,
                        },
                        implementation: SCIRGate {
                            tree: self.path.clone(),
                        },
                    });
            }
            _ => { }
        }

        return new_step;
    }
    fn repr(&self) -> String {
        return format!("{:?}", self);
    }
    fn cost(&self, arch: &CustomArch) -> f64 {
        if self.operation == Operation::Id 
        {
            0f64
        } else {
            1f64
        }
    }
}

pub fn anc_neighbors(
    step: &Step<SCIRGate>,
    loc: Location,
    width: usize,
    height: usize,
    boundary_type: Option<BoundaryType>,
) -> Vec<Location> {
    let mut neighbors = Vec::new();
    let patch = step.patch_map[&get_key(step.map(), loc).unwrap()];

    if boundary_type == None || patch.top_bottom == boundary_type {
        // horizontal
        if loc.get_index() % width > 0 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() - 1)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                neighbors.push(Location::new(loc.get_index() - 1));
            }
        }
        if loc.get_index() % width < width - 1 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() + 1)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                neighbors.push(Location::new(loc.get_index() + 1));
            }
        }
    }

    if boundary_type == None || patch.top_bottom != boundary_type {
        // vertical
        if loc.get_index() / width > 0 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() - width)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                neighbors.push(Location::new(loc.get_index() - width));
            }
        }
        if loc.get_index() / width < height - 1 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() + width)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                neighbors.push(Location::new(loc.get_index() + width));
            }
        }
    }

    return neighbors;
}

pub fn anc_neighbors_corner(
    step: &Step<SCIRGate>,
    loc: Location,
    width: usize,
    height: usize,
    boundary_type: Option<BoundaryType>,
) -> Vec<(Location, Location)> {
    let mut neighbors = Vec::new();
    let patch = step.patch_map[&get_key(step.map(), loc).unwrap()];

    if boundary_type == None || patch.top_bottom == boundary_type {
        // horizontal
        if loc.get_index() % width > 0 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() - 1)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                anc_neighbors(step, Location::new(loc.get_index() - 1), width, height, None)
                .iter()
                .for_each(|loc2| if *loc2 != loc { neighbors.push((Location::new(loc.get_index() - 1), *loc2)) });
            }
        }
        if loc.get_index() % width < width - 1 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() + 1)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                anc_neighbors(step, Location::new(loc.get_index() + 1), width, height, None)
                .iter()
                .for_each(|loc2| if *loc2 != loc { neighbors.push((Location::new(loc.get_index() + 1), *loc2)) });
            }
        }
    }

    if boundary_type == None || patch.top_bottom != boundary_type {
        // vertical
        if loc.get_index() / width > 0 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() - width)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                anc_neighbors(step, Location::new(loc.get_index() - width), width, height, None)
                .iter()
                .for_each(|loc2| if *loc2 != loc { neighbors.push((Location::new(loc.get_index() - width), *loc2)) });
            }
        }
        if loc.get_index() / width < height - 1 {
            if step.patch_map[&get_key(step.map(), Location::new(loc.get_index() + width)).unwrap()]
                .patch_type
                == PatchType::ANCILLA
            {
                anc_neighbors(step, Location::new(loc.get_index() + width), width, height, None)
                .iter()
                .for_each(|loc2| if *loc2 != loc { neighbors.push((Location::new(loc.get_index() + width), *loc2)) });
            }
        }
    }

    return neighbors;
}

pub fn count_bends(path: Vec<Location>) -> usize {
    let mut num_bends = 0;
    for i in 1..path.len() - 1 {
        let prev = path[i - 1];
        let curr = path[i];
        let next = path[i + 1];

        if prev.get_index().abs_diff(curr.get_index())
            != curr.get_index().abs_diff(next.get_index())
        {
            num_bends = num_bends + 1;
        }
    }

    return num_bends;
}

pub fn get_used_locs(step: &Step<SCIRGate>) -> HashSet<Location> {
    return step
        .implemented_gates()
        .into_iter()
        .map(|x| {
            extend_and_return(
                x.implementation.tree(),
                x.gate.qubits.iter().map(|q| step.map[q]),
            )
        })
        .flatten()
        .collect();
}

fn available_transitions(arch: &CustomArch, step: &Step<SCIRGate>) -> Vec<SCIRTrans> {
    let mut new_step = step.clone();
    new_step.map = step.map.clone();
    new_step.patch_map = step.patch_map.clone();
    new_step.counter_map = step.counter_map.clone();
    new_step.implemented_gates = HashSet::new();
    new_step.time = step.time + 1;

    // add new op to count tracker
    for gate in step.implemented_gates.clone() {
        if !new_step.counter_map.contains_key(&gate.gate.id) {
            let duration = match gate.gate.operation {
                Operation::S => 3,
                Operation::CX => 4,
                Operation::T => 4,
                Operation::TComposite => 16,
                Operation::H | Operation::HLitinski | Operation::LitinskiRotate => 6,
                Operation::CultivateTX | Operation::CultivateTZ => 12,
                Operation::Move
                | Operation::MoveRotate
                | Operation::PauliMeasurement { .. }
                | Operation::PauliRot { .. } => 2,
                Operation::ResetToAncilla => 2,
                Operation::Id => 0,
                Operation::Walk => 4
            };
            new_step
                .counter_map
                .insert(gate.gate.id, (gate.clone(), duration));
        }
    }

    // decrement counter
    // if counter 0 then implement effect, otherwise add gate to next step
    for (id, (gate, counter)) in new_step.counter_map.clone() {
        new_step.counter_map.insert(id, (gate.clone(), counter - 1));

        if counter - 1 == 0 {
            match gate.clone().gate.operation {
                Operation::Move => {
                    new_step.map = swap_keys(
                        &new_step.map,
                        gate.clone().implementation.tree[0],
                        gate.clone().implementation.tree[gate.clone().implementation.tree.len() - 1],
                    );
                }
                Operation::MoveRotate => {
                    new_step.patch_map = rotate(
                        &new_step.map,
                        &new_step.patch_map,
                        gate.implementation.tree[0],
                    );
                    new_step.map = swap_keys(
                        &new_step.map,
                        gate.clone().implementation.tree[0],
                        gate.clone().implementation.tree[gate.clone().implementation.tree.len() - 1],
                    );
                }
                Operation::Walk => {
                    // first element is empty patch we're walking into
                    for i in 1..gate.implementation.tree.len() {
                        new_step.map = swap_keys(
                            &new_step.map,
                            gate.clone().implementation.tree[i],
                            gate.clone().implementation.tree[i - 1],
                        );
                    }
                }
                Operation::LitinskiRotate => {
                    new_step.patch_map = rotate(
                        &new_step.map,
                        &new_step.patch_map,
                        gate.implementation.tree[0],
                    );
                }
                Operation::CultivateTZ => {
                    new_step.patch_map = magic_t_init(
                        &new_step.map,
                        &new_step.patch_map,
                        gate.implementation.tree[0],
                        BoundaryType::Z,
                    );
                }
                Operation::CultivateTX => {
                    new_step.patch_map = magic_t_init(
                        &new_step.map,
                        &new_step.patch_map,
                        gate.implementation.tree[0],
                        BoundaryType::X,
                    );
                }
                Operation::H => {
                    new_step.map = swap_keys(
                        &new_step.map,
                        gate.clone().implementation.tree[0],
                        gate.clone().implementation.tree[1],
                    );
                }
                Operation::T | Operation::ResetToAncilla => {
                    new_step.patch_map = magic_t_reset(
                        &new_step.map,
                        &new_step.patch_map,
                        *gate.implementation.tree.last().unwrap(),
                    );
                }
                Operation::Id | Operation::S | Operation::TComposite | Operation::HLitinski | Operation::CX | Operation::PauliMeasurement { .. } | Operation::PauliRot { .. } => {
                    // no need to update
                }
            }
        } else if counter > 0 {
            new_step.implemented_gates.insert(gate.clone());
        }
    }

    let mut transitions: Vec<SCIRTrans> = vec![SCIRTrans {
            path: vec![],
            operation: Operation::Id,
            next_step: new_step.clone(),
        }];

    if step.time % 2 == 0 {
        // only allow identity transition when next step is odd time
        return transitions;
    }

    let used_locs: HashSet<Location> = get_used_locs(&new_step);

    if CONFIG.transitions_allowed.contains(&"MOVE".to_string()) {
        new_step
            .map
            .iter()
            .filter(|(q, l)| !used_locs.contains(l))
            .filter(|(q, l)| new_step.patch_map[*q].patch_type == PatchType::ALGORITHM)
            .map(|(q, l)| {
                all_paths(
                    arch,
                    anc_neighbors(&new_step, *l, arch.width, arch.height, None),
                    new_step
                        .map
                        .iter()
                        .filter(|(q, l)| !used_locs.contains(l))
                        .filter(|(q, _)| new_step.patch_map[*q].patch_type == PatchType::ANCILLA)
                        .map(|(_, l2)| *l2)
                        .collect_vec(),
                    extend_and_return(
                        used_locs.clone(),
                        alg_qubit_locs(new_step.map(), new_step.patch_map()),
                    ),
                )
                .map(|p| extend_and_return(vec![*l], p))
            })
            .flatten()
            .filter(|p| count_bends((*p).clone()) % 2 == 0)
            .for_each(|x| 
                transitions.push(SCIRTrans {
                path: x,
                operation: Operation::Move,
                next_step: new_step.clone(),
            }));
    }

    if CONFIG.transitions_allowed.contains(&"MOVE_ROTATE".to_string()) {
        new_step
            .map
            .iter()
            .filter(|(q, l)| !used_locs.contains(l))
            .filter(|(q, l)| new_step.patch_map[*q].patch_type == PatchType::ALGORITHM)
            .map(|(q, l)| {
                all_paths(
                    arch,
                    anc_neighbors(&new_step, *l, arch.width, arch.height, None),
                    new_step
                        .map
                        .iter()
                        .filter(|(q, l)| !used_locs.contains(l))
                        .filter(|(q, l)| new_step.patch_map[*q].patch_type == PatchType::ANCILLA)
                        .map(|(q, l)| *l)
                        .collect_vec(),
                    extend_and_return(
                        used_locs.clone(),
                        alg_qubit_locs(new_step.map(), new_step.patch_map()),
                    ),
                )
                .map(|p| extend_and_return(vec![*l], p))
            })
            .flatten()
            .filter(|p| count_bends((*p).clone()) % 2 == 1)
            .for_each(|x| {
                transitions.push(SCIRTrans {
                    path: x,
                    operation: Operation::MoveRotate,
                    next_step: new_step.clone(),
                })
            });
    }

    if CONFIG.transitions_allowed.contains(&"LITINSKI_ROTATE".to_string()) {
        new_step
            .map
            .iter()
            .filter(|(q, l)| !used_locs.contains(l))
            .filter(|(q, l)| new_step.patch_map[*q].patch_type != PatchType::ANCILLA)
            .map(|(q, l)| {
                anc_neighbors(
                    &new_step,
                    *l,
                    arch.width,
                    arch.height,
                    None
                )
                .into_iter()
                .filter(|l| !used_locs.contains(l))
                .map(|p| vec![*l, p])
            })
            .flatten()
            .for_each(|x| {
                transitions.push(SCIRTrans {
                    path: x,
                    operation: Operation::LitinskiRotate,
                    next_step: new_step.clone(),
                })
            });
    }

    if CONFIG.transitions_allowed.contains(&"RESET_TO_ANCILLA".to_string()) {
        if new_step.time % 16 == 0 {
            new_step
                .map
                .iter()
                .filter(|(q, l)| !used_locs.contains(l))
                .filter(|(q, l)| new_step.patch_map[*q].patch_type == PatchType::MAGICT)
                .for_each(|(q, l)| {
                    transitions.push(SCIRTrans {
                        path: vec![*l],
                        operation: Operation::ResetToAncilla,
                        next_step: new_step.clone(),
                    })
                });
        }
    }

    if CONFIG.transitions_allowed.contains(&"CULTIVATE_T_Z".to_string()) {
        new_step
            .map
            .iter()
            .filter(|(q, l)| !used_locs.contains(l))
            .filter(|(q, l)| {
                new_step.patch_map[*q].patch_type == PatchType::ANCILLA
                    && arch.magic_state_qubits().contains(l)
            })
            .for_each(|(q, l)| {
                transitions.push(SCIRTrans {
                    path: vec![*l],
                    operation: Operation::CultivateTZ,
                    next_step: new_step.clone(),
                })
            });
    }

    if CONFIG.transitions_allowed.contains(&"WALK".to_string()) {
        new_step
            .map
            .iter()
            .filter(|(q, l)| !used_locs.contains(l))
            .filter(|(q, l)| {
                new_step.patch_map[*q].patch_type == PatchType::ANCILLA
            })
            .map(|x| {
                // in each direction (up, down, left, right) find longest chain of unoccupied algorithm qubits to walk
                let mut paths = Vec::new();
                let directions = vec![-1isize, 1, -(arch.width as isize), arch.width as isize];
                for dir in directions {
                    let mut current_path = vec![];
                    let mut current_index = x.1.get_index() as isize;
                    loop {
                        let old_loc = Location::new(current_index as usize);
                        current_index = current_index + dir;
                        if current_index < 0 || current_index >= (arch.width * arch.height) as isize {
                            break;
                        }
                        let next_loc = Location::new(current_index as usize);
                        
                        if arch.contains_edge((old_loc, next_loc))
                            && new_step.patch_map[&get_key(new_step.map(), next_loc).unwrap()]
                                .patch_type
                                == PatchType::ALGORITHM
                            && !used_locs.contains(&next_loc)
                        {
                            current_path.push(next_loc);
                        } else {
                            break;
                        }
                    }
                    if current_path.len() > 0 {
                        let mut full_path = vec![*x.1];
                        full_path.extend(current_path);
                        println!("path length : {}", full_path.len());
                        paths.push(full_path);
                    }
                }
                paths
            })
            .flatten()
            .for_each(|p| {
                transitions.push(SCIRTrans {
                    path: p,
                    operation: Operation::Walk,
                    next_step: new_step.clone(),
                })
            });
    }

    println!("num transitions: {}", transitions.len());

    return transitions;
}

fn realize_gate(
    step: &Step<SCIRGate>,
    arch: &CustomArch,
    gate: &Gate,
) -> Box<dyn Iterator<Item = SCIRGate>> {
    if step.time % 2 == 1 && gate.operation != Operation::S {
        return Box::new(Vec::new().into_iter());
    }

    let used_locs: HashSet<Location> = get_used_locs(step);

    let is_disjoint = gate
        .qubits
        .iter()
        .fold(true, |acc, x| acc && !used_locs.contains(&step.map[x]));

    if !is_disjoint {
        return Box::new(Vec::new().into_iter());
    }

    if (gate.gate_type()) == (CX) {
        let (cpos, tpos) = (step.map[&gate.qubits[0]], step.map[&gate.qubits[1]]);
        let (cpatch, tpatch) = (
            step.patch_map[&gate.qubits[0]],
            step.patch_map[&gate.qubits[1]],
        );
        Box::new(
            all_paths(
                arch,
        anc_neighbors(&step,  cpos, arch.width, arch.height, Some(BoundaryType::Z)).into_iter().filter(|x| !used_locs.contains(x)).collect_vec(),
        anc_neighbors(&step,  tpos, arch.width, arch.height, Some(BoundaryType::X)).into_iter().filter(|x| !used_locs.contains(x)).collect_vec(),
        extend_and_return(alg_qubit_locs(step.map(), step.patch_map()), used_locs),
            )
            .into_iter()
            .filter(move |x: &Vec<Location>| {
                count_bends(extend_and_return(
                    extend_and_return(vec![cpos], (*x).clone()),
                    vec![tpos],
                )) > 0
            })
            .map(|x| SCIRGate { tree: x }),
        )
    } else if (gate.gate_type()) == (T) {
        let cpos = step.map[&gate.qubits[0]];
        let magic_states = step
            .clone()
            .patch_map
            .into_iter()
            .filter(|(q, p)| p.patch_type == PatchType::MAGICT)
            .map(|(q, p)| q);
        let cpatch = step.patch_map[&gate.qubits[0]];
        let mut realizations : Box<dyn Iterator<Item = Vec<Location>>> = Box::new(std::iter::empty::<Vec<Location>>());
        magic_states.for_each(|target| {
            let tpatch = step.patch_map[&target];
            let tpos = step.map[&target];

            if !used_locs.contains(&tpos) {
                let old = std::mem::replace(
            &mut realizations,
            Box::new(std::iter::empty::<Vec<Location>>()),
        );
                realizations = 
                    Box::new(old.chain(
                    all_paths(
                        arch,
                anc_neighbors(&step,  cpos, arch.width, arch.height, Some(BoundaryType::Z)).into_iter().filter(|x| !used_locs.contains(x)).collect_vec(),
                anc_neighbors(&step,  tpos, arch.width, arch.height, Some(BoundaryType::X)).into_iter().filter(|x| !used_locs.contains(x)).collect_vec(),
                extend_and_return(
                            alg_qubit_locs(step.map(), step.patch_map()),
                            used_locs.clone(),
                        ),
                    )
                    .filter(move |x| {
                        count_bends(extend_and_return(
                            vec![cpos],
                            extend_and_return((*x).clone(), vec![tpos]),
                        )) > 0
                    })
                    .map(move |p| extend_and_return(p, vec![tpos]))
                ));
            }
        });
        Box::new(realizations.map(|x| SCIRGate { tree: x }))
    } else if gate.gate_type() == TComposite {
        let pos = step.map[&gate.qubits[0]];
        Box::new(
            anc_neighbors_corner(&step, pos, arch.width, arch.height, Some(BoundaryType::Z))
                .into_iter()
                .filter(move |(l1, l2)| !used_locs.contains(l1) && !used_locs.contains(l2))
                .filter(move |(l1, l2)| {
                    (pos.get_index().abs_diff(l1.get_index()) == 1
                        && l2.get_index().abs_diff(l1.get_index()) != 1)
                        || (pos.get_index().abs_diff(l1.get_index()) != 1
                            && l2.get_index().abs_diff(l1.get_index()) == 1)
                })
                .map(|(l1, l2)| SCIRGate { tree: vec![l1, l2] }),
        )
    } 
    else if gate.gate_type() == S {
        let pos = step.map[&gate.qubits[0]];
        Box::new(
            anc_neighbors(&step, pos, arch.width, arch.height, Some(BoundaryType::Z))
                .into_iter()
                .filter(move |l: &Location| !used_locs.contains(l))
                .map(|l| SCIRGate { tree: vec![l] }),
        )
    } else if gate.gate_type() == HLitinski {
        let pos = step.map[&gate.qubits[0]];
        Box::new(
            anc_neighbors(&step, pos, arch.width, arch.height, None)
                .into_iter()
                .filter(move |l: &Location| !used_locs.contains(l))
                .map(|l| SCIRGate { tree: vec![l] }),
        )
    } else if gate.gate_type() == H {
        let pos = step.map[&gate.qubits[0]];
        Box::new(
            anc_neighbors_corner(&step, pos, arch.width, arch.height, None)
                .into_iter()
                .filter(move |(l1, l2)| !used_locs.contains(l1) && !used_locs.contains(l2))
                .filter(move |(l1, l2)| {
                    (pos.get_index().abs_diff(l1.get_index()) == 1
                        && l2.get_index().abs_diff(l1.get_index()) != 1)
                        || (pos.get_index().abs_diff(l1.get_index()) != 1
                            && l2.get_index().abs_diff(l1.get_index()) == 1)
                })
                .map(|(l1, l2)| SCIRGate { tree: vec![l1, l2] }),
        )
    } 
    else if gate.gate_type() == PauliMeasurement || gate.gate_type() == PauliRot {
        let container = if (gate.gate_type()) == (PauliMeasurement) {
            steiner_trees(
                arch,
                extend_and_return(
                    extend_and_return(
                        Vec::new(),
                        gate.x_indices().into_iter().map(|x| {
                            anc_neighbors(&step, step.map[&x], arch.width, arch.height, Some(BoundaryType::X))
                        }),
                    ),
                    gate.z_indices().into_iter().map(|x| {
                        anc_neighbors(&step, step.map[&x], arch.width, arch.height, Some(BoundaryType::Z))
                    }),
                ),
                extend_and_return(alg_qubit_locs(step.map(), step.patch_map()), used_locs),
            )
        } else {
            steiner_trees(
                arch,
                extend_and_return(
                    extend_and_return(
                        extend_and_return(
                            Vec::new(),
                            gate.x_indices().into_iter().map(|x| {
                                anc_neighbors(&step, step.map[&x], arch.width, arch.height, Some(BoundaryType::X))
                            }),
                        ),
                        gate.z_indices().into_iter().map(|x| {
                            anc_neighbors(&step, step.map[&x], arch.width, arch.height, Some(BoundaryType::Z))
                        }),
                    ),
                    arch.magic_state_qubits()
                        .into_iter()
                        .map(|x| horizontal_neighbors(x, arch.width)), // TODO: Not getting magic states correctly
                ),
                extend_and_return(
                    extend_and_return(values(step.map()), arch.magic_state_qubits()),
                    step.implemented_gates()
                        .into_iter()
                        .map(|x| x.implementation.tree())
                        .into_iter()
                        .fold(Vec::new(), |acc, x| extend_and_return(acc, x)),
                ),
            )
        };
        Box::new(container.into_iter().map(|x| SCIRGate { tree: x }))
    } else {
        panic!("Unexpected realize gate type {:?}", gate.gate_type())
    }
}

fn custom_step_cost(step: &Step<SCIRGate>, arch: &CustomArch) -> f64 {
    return 1f64;
}

// fn mapping_heuristic(arch: &CustomArch, c: &Circuit, map: &HashMap<Qubit, Location>) -> f64 {
//     let graph = &arch.graph;
//     let mut cost = 0;
//     for gate in &c.gates {
//         if gate.qubits.len() < 2 {
//             continue;
//         }
//         let (cpos, tpos) = (map.get(&gate.qubits[0]), map.get(&gate.qubits[1]));
//         let (cind, tind) = (arch.index_map[cpos.unwrap()], arch.index_map[tpos.unwrap()]);
//         let sp_res = petgraph::algo::astar(graph, cind, |n| n == tind, |_| 1, |_| 1);
//         match sp_res {
//             Some((c, _)) => cost += c,
//             None => {
//                 panic!(
//                     "Disconnected graph. No path found from {:?} to {:?}",
//                     cpos, tpos
//                 )
//             }
//         }
//     }
//     return cost as f64;
// }

fn my_solve(c: &Circuit, a: &CustomArch) -> CompilerResult<SCIRGate> {
    return backend::solve(
        c,
        a,
        &|s| available_transitions(a, s),
        &realize_gate,
        custom_step_cost,
        None,
        false,
    );
}

fn my_sabre_solve(c: &Circuit, a: &CustomArch) -> CompilerResult<SCIRGate> {
    return backend::sabre_solve(
        c,
        a,
        &|s| available_transitions(a, s),
        &realize_gate,
        custom_step_cost,
        None,
        true,
    );
}

fn my_joint_solve_parallel(c: &Circuit, a: &CustomArch) -> CompilerResult<SCIRGate> {
    return backend::solve_joint_optimize_parallel(
        c,
        a,
        &|s| available_transitions(a, s),
        &realize_gate,
        custom_step_cost,
        None,
        true,
    );
}
