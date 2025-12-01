use std::{default, fs};

use once_cell::sync::Lazy;

use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Debug)]
pub struct SolverConfig {
    #[serde(default = "default_alpha")]
    pub alpha: f64,

    #[serde(default = "default_beta")]
    pub beta: f64,

    #[serde(default = "default_gamma")]
    pub gamma: f64,

    #[serde(default = "default_delta")]
    pub delta: f64,

    #[serde(default = "default_extended_set_weight")]
    pub extended_set_weight: f64,

    #[serde(default = "default_mapping_search_initial_temp")]
    pub mapping_search_initial_temp: f64,

    #[serde(default = "default_mapping_search_term_temp")]
    pub mapping_search_term_temp: f64,

    #[serde(default = "default_mapping_search_cool_rate")]
    pub mapping_search_cool_rate: f64,

    #[serde(default = "default_exhaustive_search_threshold")]
    pub exhaustive_search_threshold: usize,

    #[serde(default = "default_routing_search_initial_temp")]
    pub routing_search_initial_temp: f64,

    #[serde(default = "default_routing_search_term_temp")]
    pub routing_search_term_temp: f64,

    #[serde(default = "default_routing_search_cool_rate")]
    pub routing_search_cool_rate: f64,

    #[serde(default = "default_sabre_iterations")]
    pub sabre_iterations: usize,

    #[serde(default = "default_isom_search_timeout")]
    pub isom_search_timeout: u64,

    #[serde(default = "default_parallel_searches")]
    pub parallel_searches: usize,

    #[serde(default = "default_limited_search_cool_rates")]
    pub limited_search_cool_rates: [f64; 4],

    #[serde(default = "default_transitions_allowed")]
    pub transitions_allowed: Vec<String>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        return SolverConfig {
            alpha: default_alpha(),
            beta: default_beta(),
            gamma: default_gamma(),
            delta: default_delta(),
            extended_set_weight: default_extended_set_weight(),
            mapping_search_initial_temp: default_mapping_search_initial_temp(),
            mapping_search_term_temp: default_mapping_search_term_temp(),
            mapping_search_cool_rate: default_mapping_search_cool_rate(),
            exhaustive_search_threshold: default_exhaustive_search_threshold(),
            routing_search_initial_temp: default_routing_search_initial_temp(),
            routing_search_term_temp: default_routing_search_term_temp(),
            routing_search_cool_rate: default_routing_search_cool_rate(),
            sabre_iterations: default_sabre_iterations(),
            isom_search_timeout: default_isom_search_timeout(),
            parallel_searches: default_parallel_searches(),
            limited_search_cool_rates: default_limited_search_cool_rates(),
            transitions_allowed: default_transitions_allowed(),
        };
    }
}

fn default_alpha() -> f64 {
    return 1.0;
}

fn default_beta() -> f64 {
    return 1.0;
}

fn default_gamma() -> f64 {
    return 1.0;
}
fn default_delta() -> f64 {
    return 1.0;
}

fn default_extended_set_weight() -> f64 {
    return 0.5;
}

fn default_mapping_search_initial_temp() -> f64 {
    return 10.0;
}

fn default_mapping_search_term_temp() -> f64 {
    return 0.00001;
}

fn default_mapping_search_cool_rate() -> f64 {
    return 0.999;
}

fn default_exhaustive_search_threshold() -> usize {
    return 8;
}

fn default_routing_search_initial_temp() -> f64 {
    return 10.0;
}

fn default_routing_search_term_temp() -> f64 {
    return 0.00001;
}

fn default_routing_search_cool_rate() -> f64 {
    return 0.999;
}

fn default_sabre_iterations() -> usize {
    return 3;
}

fn default_isom_search_timeout() -> u64 {
    return 300;
}

fn default_parallel_searches() -> usize {
    return 16;
}

fn default_limited_search_cool_rates() -> [f64; 4] {
    return [0.0, 0.349, 0.99, 0.9];
}

fn default_transitions_allowed() -> Vec<String> {
    return vec![];
}

pub static CONFIG: Lazy<SolverConfig> = Lazy::new(|| {
    let data = fs::read_to_string("config.json").unwrap_or_else(|_| "".to_string());
    serde_json::from_str(&data).unwrap_or_else(|_| SolverConfig::default())
});
