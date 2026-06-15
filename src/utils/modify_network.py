
def scale_all_loads(md, a):
    """Scale all loads from L to a*L."""
    load_names = list(md.data["elements"]["load"].keys())
    for name in load_names:
        md.data["elements"]["load"][name]["p_load"] *= a
        md.data["elements"]["load"][name]["q_load"] *= a
    
def scale_load(md, a, name):
    md.data["elements"]["load"][name]["p_load"] *= a
    md.data["elements"]["load"][name]["q_load"] *= a
    
def scale_gen(md, a, name):
    md.data["elements"]["generator"][name]["p_min"] *= a
    md.data["elements"]["generator"][name]["q_min"] *= a
    md.data["elements"]["generator"][name]["p_max"] *= a
    md.data["elements"]["generator"][name]["q_max"] *= a
