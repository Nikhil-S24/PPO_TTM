import pandas as pd
import matplotlib.pyplot as plt

def get_final_revenue(filename):
    try:
        # We only need the last row to get the final revenue
        df = pd.read_csv(filename)
        return float(df["total_revenue"].iloc[-1])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0.0

def main():
    files = {
        "Baseline": "output_baseline_fix1.csv",
        "TTM": "output_ttm_fix1.csv",
        "PPO": "output_ppo_fix1.csv",
        "PPO+TTM": "output_ppo_ttm_fix1.csv"
    }
    
    revenues = {}
    for label, file in files.items():
        revenues[label] = get_final_revenue(file)
        
    print("Final Revenues:")
    for label, rev in revenues.items():
        print(f"{label}: ${rev:,.2f}")
        
    labels = list(revenues.keys())
    values = list(revenues.values())
    
    plt.figure(figsize=(9, 6))
    colors = ['#7f7f7f', '#2ca02c', '#1f77b4', '#ff7f0e']
    
    bars = plt.bar(labels, values, color=colors, width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01),
                 f'${yval:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                 
    plt.xlabel("Policy", fontweight='bold', fontsize=12)
    plt.ylabel("Final Revenue ($)", fontweight='bold', fontsize=12)
    plt.title("Final Revenue Comparison of Different Policies", fontweight='bold', fontsize=14)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(7, 7))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    output_filename = "final_revenue_bar.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    print(f"Saved bar chart to {output_filename}")
    
if __name__ == '__main__':
    main()
