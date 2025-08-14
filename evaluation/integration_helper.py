import logging
import torch
from evaluation.bootstrap_evaluator import BootstrapEvaluator
from evaluation.visualization import Visualization

def run_post_training_evaluation(args, trained_model, target_hospital_data, device):
    """
    a simple helper function to run bootstrap evaluation after training completes
    
    Args:
        args: Configuration object with bootstrap evaluation settings
        trained_model: The trained model to evaluate
        target_hospital_data: Dict with 'x_target' and 'y_target' from excluded hospital
        device: Device to run evaluation on
    
    Returns:
        dict: Bootstrap evaluation results
    """
    
    if not getattr(args, 'run_bootstrap_evaluation', False):
        logging.info("Bootstrap evaluation not enabled")
        return None
    
    logging.info("Starting bootstrap evaluation")
    
    try:
        evaluator = BootstrapEvaluator(args, device)
        
        algorithm_name = 'fedfed' if getattr(args, 'VAE', False) else 'fedavg'
        if getattr(args, 'fedprox', False):
            algorithm_name = 'fedprox'
            
        target_hospital_id = getattr(args, 'target_hospital_id', 'unknown')
        
        prepared_data = evaluator.prepare_target_hospital_data(
            target_hospital_data['x_target'],
            target_hospital_data['y_target']
        )
        
        results = evaluator.run_bootstrap_evaluation(
            trained_model, prepared_data, algorithm_name, target_hospital_id
        )
        
        if results and 'auprc' in results:
            mean_score = results['auprc']['mean']
            std_score = results['auprc']['std']
            logging.info(f"{algorithm_name.upper()} Hospital {target_hospital_id}: AUPRC {mean_score:.4f} ± {std_score:.4f}")
            
        if getattr(args, 'create_evaluation_plots', False):
            try:
                visualizer = Visualization(getattr(args, 'output_path', './output/evaluation_results/'))
                results_data = visualizer.load_comparison_results(target_hospital_id)
                if results_data is not None:
                    fig, ax = visualizer.create_algorithm_comparison_boxplot(
                        results_data, metric='auprc', target_hospital_id=target_hospital_id
                    )
                    if fig is not None:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                        logging.info("Plots created")
            except Exception as e:
                logging.warning(f"Could not create plots: {e}")
        
        logging.info("Bootstrap evaluation completed")
        return results
        
    except Exception as e:
        logging.error(f"Bootstrap evaluation failed: {e}")
        return None

