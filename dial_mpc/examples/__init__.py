# names resolve to dial_mpc/examples/<name>.yaml (io_utils.get_example_path)
examples = [  # sync runner: dial-mpc --example <name>
    "unitree_h1_jog",
    "unitree_h1_push_crate",
    "unitree_h1_loco",
    "unitree_go2_trot",
    "unitree_go2_seq_jump",
    "unitree_go2_crate_climb",
    "allegro_reorient",
    "block1d",  # added: block1d push task
]

deploy_examples = [  # async runners: dial-mpc-sim/-plan/-real/-lcm-bridge --example <name>
    "unitree_go2_trot_deploy",
    "unitree_go2_seq_jump_deploy",
    "unitree_h1_loco_deploy",
    "block1d_deploy",  # added: block1d vs internal sim (dial-mpc-sim + dial-mpc-plan)
    # "block1d_lcm",  # LCM-bridge example parked in discard/examples/ (yaml moved there too)
]
