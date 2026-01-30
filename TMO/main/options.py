import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default = 'cuda:0')
    parser.add_argument('--repeat', type=int, default = 1)

    parser.add_argument('--time_span', type=float, default = 5)
    parser.add_argument('--constraint_lambda', type=float, default = 10)
    parser.add_argument('--nearest_neighbors', type=float, default = 5)

    parser.add_argument('--alpha', type=float, default = 1)
    parser.add_argument('--beta_association', type=float, default = 1/3)
    parser.add_argument('--beta_latency', type=float, default = 1/3)
    parser.add_argument('--beta_usage', type=float, default = 1/3)
    parser.add_argument('--beta_security', type=float, default = 0.0)
    parser.add_argument('--beta_modality_privacy', type=float, default=0.2,
                       help='Weight for modality privacy risk in reward')
    parser.add_argument('--privacy_budget', type=float, default = 1.0)
    parser.add_argument('--use_privacy_rl', action='store_true', default=False)
    parser.add_argument('--use_image_sensitivity', action='store_true', default=False,
                       help='Enable image sensitivity analysis (requires opencv)')
    parser.add_argument('--simulate_image_sensitivity', action='store_true', default=False,
                       help='Simulate image sensitivity when actual images unavailable')
    parser.add_argument('--latency_budget', type=float, default = 30)
    parser.add_argument('--usage_budget', type=float, default = 0.05)
    parser.add_argument('--local_device', type=str, default = 'Jetson TX2')
    parser.add_argument('--cloud_server', type=str, default = 'Wired')
    args = parser.parse_args()
    return args