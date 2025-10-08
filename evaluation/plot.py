import os
import sys
import logging

sys.path.insert(0, os.path.abspath('.'))

from evaluation.visualization import Visualization
from configs import get_cfg

def plot2():
    """
    Create grouped bar plot comparing algorithms across multiple target hospitals
    """

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Get config to determine task type
    cfg = get_cfg()
    class MockArgs:
        def __init__(self):
            self.config_file = 'configs/eicu_config.yaml'
            self.opts = []
    mock_args = MockArgs()
    cfg.setup(mock_args)
    cfg.mode = 'standalone'

    output_path = './output/evaluation_results/'


    hospital_ids = [167, 199, 420, 458]

    algorithms = ['fedavg', 'fedprox', 'fedfed', 'centralized']

    print(f"Creating grouped bar plots for hospitals: {hospital_ids}")
    print(f"Output directory: {output_path}")

    vis = Visualization(output_path)

    print(f"\\nCreating grouped bar plot ...")

    fig, ax = vis.create_grouped_boxplot(
        hospital_ids=hospital_ids,
        metric=None,  # auto-detect logic
        algorithms=algorithms,
        args=cfg
    )

    
if __name__ == '__main__':
    plot2()