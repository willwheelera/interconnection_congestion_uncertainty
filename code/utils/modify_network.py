
def scale_all_loads(md, a):
    """Scale all loads from L to a*L."""
    load_names = list(md.data["elements"]["load"].keys())
    for name in load_names:
        md.data["elements"]["load"][name]["p_load"] *= a
        md.data["elements"]["load"][name]["q_load"] *= a
    
