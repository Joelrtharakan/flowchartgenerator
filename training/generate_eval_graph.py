import matplotlib.pyplot as plt

def generate_graph():
    # Metrics based on monitor.py mock evaluation data
    metrics = ['Validation Accuracy', 'Structural Correctness', 'Minimum Threshold']
    values = [87.5, 94.2, 80.0]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#2ca02c', '#d62728'])

    # Add labels and title
    plt.ylim(0, 100)
    plt.ylabel('Percentage (%)')
    plt.title('Model Evaluation Metrics')

    # Add percentage labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')

    # Save the plot
    output_path = 'evaluation_graph.png'
    plt.savefig(output_path)
    print(f"Graph successfully generated and saved as {output_path}")

if __name__ == '__main__':
    generate_graph()
