import numpy as np
import copy
import json

#LOAD_PROFILE = 2/3 * np.array([
#    168.0, 154.0, 145.6, 140.0, 145.6, 162.4,
#    187.6, 215.6, 243.6, 254.8, 260.4, 263.2,
#    257.6, 249.2, 243.6, 246.4, 257.6, 271.6,
#    280.0, 274.4, 263.2, 246.4, 218.4, 190.4,
#])
shape1 = np.interp(np.arange(24), [0, 12, 24], [0, 1, 0])
#shape2 = np.interp(np.arange(24), [-1, 3, 11, 15, 23], [0, 1, 0, 1, 0])
shape2 = np.interp(np.arange(24), [0, 6, 12, 18, 24], [0.2, 1, 0, 1, 0])
PEAK = 150
weight = 0.8
LOAD2 = (weight * shape1 + 1 - weight) * PEAK
LOAD4 = (weight * shape2 + 1 - weight) * PEAK


# Make simple load profile. Properties:
# Contains high L4, low L2
# Contains high L4, high L2
# Contains high ramp rate
# Start at 0 load.


def line_model(bus_fr, bus_to, rating=100., angle_lim=90, **kwargs):
    d = {
        "from_bus": bus_fr, "to_bus": bus_to, "reactance": 0.002, 
        "in_service": True, "branch_type": "line", 
    }
    d["angle_diff_min"] = -angle_lim
    d["angle_diff_max"] = angle_lim
    d["rating_long_term"] = rating
    d["rating_short_term"] = rating
    d["rating_emergency"] = rating
    d.update(kwargs)
    return d

G_TYPE1 = {
    #"p_max": 400.0,
    "min_up_time": 8,
    "min_down_time": 5,
    "startup_cost": [[5, 15.0]],
    "p_cost": {
      "data_type": "cost_curve",
      "cost_curve_type": "piecewise",
      "values": [
        [0.25, 22.00],
        [0.50, 21.00],
        [0.75, 21.00],
        [1.00, 21.25]
      ]
    }
}

G_TYPE2 = {
    #"p_max": 200.0,
    "min_up_time": 4,
    "min_down_time": 3,
    "startup_cost": [[3, 16.0]],
    "p_cost": {
      "data_type": "cost_curve",
      "cost_curve_type": "piecewise",
      "values": [
        [0.25, 28.0],
        [0.50, 26.0],
        [0.75, 26.0],
        [1.00, 26.5]
      ]
    }
}

G_TYPE3 = {
    #"p_max": 100.0,
    "min_up_time": 1,
    "min_down_time": 1,
    "startup_cost": [[1, 15.0]],
    "p_cost": {
      "data_type": "cost_curve",
      "cost_curve_type": "piecewise",
      "values": [
        [0.2, 40.0],
        [0.5, 40.0],
        [0.8, 42.5],
        [1.0, 45.0]
      ]
    }
}

G_TYPE4 = {
    #"p_max": 160.0,
    "min_up_time": 3,
    "min_down_time": 2,
    "startup_cost": [[2, 15.0]],
    "p_cost": {
      "data_type": "cost_curve",
      "cost_curve_type": "piecewise",
      "values": [
        [0.25, 27.5],
        [0.50, 25.0],
        [0.75, 25.0],
        [1.00, 25.6]
      ]
    }
}

G_DEFAULT = {
    "generator_type": "thermal",
    "fuel": "G",
    "fast_start": False,
    "fixed_commitment": None,
    "in_service": True,
    "zone": "None",
    "failure_rate": 0.0,
    "shutdown_cost": 0.0,
}

def generator_model(bus, p_max, gtype, ramp=0.5, startup_capacity=0.25, p_min=0.25, cost_scale=1.0, **kwargs):
    d = G_DEFAULT.copy()
    d.update(copy.deepcopy(gtype))

    d["bus"] = bus
    d["ramp_up_60min"] = d["ramp_down_60min"] = ramp * p_max
    d["startup_capacity"] = d["shutdown_capacity"] = startup_capacity * p_max
    for i, s in enumerate(d["startup_cost"]):
        d["startup_cost"][i] = [s[0], s[1] * d["startup_capacity"]]
    for i, v in enumerate(d["p_cost"]["values"]):
        v0 = v[0] * p_max * cost_scale
        d["p_cost"]["values"][i] = [v0, v[1] * v0]
    d["p_min"] = p_min * p_max
    d["p_max"] = p_max
    d["initial_status"] = d["min_up_time"]
    d["initial_p_output"] = d["shutdown_capacity"]

    return d
    
# Max L4: 190
# Max L2: 190?
# what's the point of gens at 2, 4? change voltage. 
# not interested in infeasible problems: make load penalty 100 instead of large.
gen_list = []
for i in range(2):
    c = 1.0 + 0.01 * i
    gen_list.extend([
      generator_model("Bus1", 200.0, G_TYPE1, cost_scale=c, p_min=0.1, ramp=0.2),
      generator_model("Bus2",  50.0, G_TYPE3, cost_scale=c, p_min=0., startup_capacity=0.4),
      generator_model("Bus3", 200.0, G_TYPE2, cost_scale=c, p_min=0.1, ramp=1.0, startup_capacity=0.2),
      generator_model("Bus4",  50.0, G_TYPE3, cost_scale=c, startup_capacity=0.4),
    ])

og_dict = {
  "elements": {
    "bus": {
      "Bus1": {"base_kv": 115.0},
      "Bus2": {"base_kv": 115.0},
      "Bus3": {"base_kv": 115.0},
      "Bus4": {"base_kv": 115.0}
    },
    "branch": {
      "L12": line_model("Bus1", "Bus2", rating=100.),
      "L23": line_model("Bus2", "Bus3", rating=100.),
      "L34": line_model("Bus3", "Bus4", rating=103.),
      "L41": line_model("Bus4", "Bus1", rating=103.),
      "L13": line_model("Bus1", "Bus3", rating=58.),
    },
    "generator": {f"G{i+1}": g for i, g in enumerate(gen_list)},
    "load": {
      "Load2": {
        "bus": "Bus2",
        "in_service": True,
        "p_load": {
          "data_type": "time_series",
          "values": (LOAD2).tolist()
        }
      },
      "Load4": {
        "bus": "Bus4",
        "in_service": True,
        "p_load": {
          "data_type": "time_series",
          "values": (LOAD4).tolist()
        }
      }
    }
  },
  "system": {
    "model_name": "many_gen_four_bus_uc",
    "time_keys": [str(i) for i in range(1, 25)],
    "time_period_length_minutes": 60,
    "load_mismatch_cost": 1000.0,
    "reserve_shortfall_cost": 1000.0,
    "baseMVA": 1.0,
    "reference_bus": "Bus1",
    "reference_bus_angle": 0.0,
    "reserve_requirement": {
      "data_type": "time_series",
      "values": [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
      ]
    }
  }
}

with open("../data/gen8_four_bus.json", "w") as f:
    json.dump(og_dict, f)
