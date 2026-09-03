"""Moon gravity task variant.

``make_moon_variant(cfg)`` turns any microduck env cfg into a lunar-gravity 
counterpart by injecting a spec_fn that overrides the global gravity to -1.62 m/s^2.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

def make_moon_variant(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Convert a microduck env cfg to use lunar gravity."""
    original_spec_fn = cfg.scene.spec_fn
    
    def moon_spec_fn(spec):
        if original_spec_fn is not None:
            original_spec_fn(spec)
        spec.option.gravity = [0.0, 0.0, -1.62]

    cfg.scene.spec_fn = moon_spec_fn
    return cfg
