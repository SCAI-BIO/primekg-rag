import os
import csv
from datetime import datetime
from typing import Dict, List, Any

def export_detailed_path(path_info: Dict[str, Any], output_dir: str = 'exported_paths') -> str:
    """Save detailed path information to a CSV file.
    
    Args:
        path_info: Dictionary containing path information
        output_dir: Directory to save the CSV file (default: 'exported_paths')
        
    Returns:
        str: Path to the saved CSV file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'path_{timestamp}.csv')
        
        # Extract detailed path steps
        detailed_path = path_info.get('Detailed_Path', '')
        steps = [step.strip() for step in detailed_path.split('•') if step.strip()]
        
        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'From', 'Relationship', 'To'])
            
            for i, step in enumerate(steps, 1):
                if '--' in step and '→' in step:
                    # Format: "Node A [type] --relationship → Node B [type]"
                    left_part, right_part = step.split('→', 1)
                    from_part = left_part.split('--', 1)[0].strip()
                    rel_part = left_part.split('--', 1)[1].strip()
                    to_part = right_part.strip()
                    
                    writer.writerow([f'Step {i}', from_part, rel_part, to_part])
        
        return filename
        
    except Exception as e:
        print(f"Error exporting path to CSV: {str(e)}")
        raise

# Example usage when run directly
if __name__ == "__main__":
    # Sample data for testing
    sample_path = {
        'Source_Node': 'Death in adolescence [effect/phenotype]',
        'Target_Node': 'drug/alcohol-induced mental disorder [disease]',
        'Status': 'Connected',
        'Path_Length': 5,
        'Detailed_Path': """
        • Death in adolescence [effect/phenotype] --parent-child → Age of death [effect/phenotype]
        • Age of death [effect/phenotype] --parent-child → Stillbirth [effect/phenotype]
        • Stillbirth [effect/phenotype] --side effect → Methotrexate [drug]
        • Methotrexate [drug] --contraindication → substance abuse/dependence [disease]
        • substance abuse/dependence [disease] --parent-child → drug/alcohol-induced mental disorder [disease]
        """
    }
    
    try:
        output_file = export_detailed_path(sample_path)
        print(f"Successfully saved detailed path to: {output_file}")
    except Exception as e:
        print(f"Error during export: {e}")
        print(f'Error exporting path: {str(e)}')
