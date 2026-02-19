import pandas as pd
import numpy as np

# Load the existing fruit dataset
df = pd.read_csv('fruit_data.csv')

print("=" * 60)
print("FRUIT DATA SAMPLE GENERATION")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)
print("\nOriginal fruit distribution:")
print(df['target'].value_counts())

# Get statistics for each fruit type to generate realistic samples
print("\n" + "=" * 60)
print("STATISTICS BY FRUIT TYPE")
print("=" * 60)

fruit_stats = {}
for fruit in df['target'].unique():
    fruit_data = df[df['target'] == fruit]
    fruit_stats[fruit] = {
        'colors': fruit_data['color'].unique().tolist(),
        'diameter_mean': fruit_data['diameter'].mean(),
        'diameter_std': fruit_data['diameter'].std(),
        'color_distribution': fruit_data['color'].value_counts().to_dict()
    }
    
    print(f"\n{fruit}:")
    print(f"  Colors: {fruit_stats[fruit]['colors']}")
    print(f"  Diameter: Mean={fruit_stats[fruit]['diameter_mean']:.2f}, Std={fruit_stats[fruit]['diameter_std']:.2f}")
    print(f"  Color distribution: {fruit_stats[fruit]['color_distribution']}")

# Generate 20 synthetic samples
np.random.seed(42)
new_samples = []

samples_per_fruit = {
    'Apple': 5,
    'Grape': 5,
    'Lemon': 5,
    'Orange': 5
}

print("\n" + "=" * 60)
print("GENERATING SYNTHETIC SAMPLES")
print("=" * 60)

for fruit, num_samples in samples_per_fruit.items():
    for _ in range(num_samples):
        # Randomly select a color from the fruit's typical colors
        color = np.random.choice(fruit_stats[fruit]['colors'])
        
        # Generate diameter from normal distribution
        diameter = np.random.normal(
            fruit_stats[fruit]['diameter_mean'],
            fruit_stats[fruit]['diameter_std']
        )
        # Ensure diameter is positive and reasonable
        diameter = max(0.5, round(diameter, 1))
        
        new_samples.append({
            'color': color,
            'diameter': diameter,
            'target': fruit
        })

# Create dataframe from new samples
new_df = pd.DataFrame(new_samples)

print("\nGenerated samples:")
print(new_df)

# Combine original and new samples
combined_df = pd.concat([df, new_df], ignore_index=True)

print("\n" + "=" * 60)
print("COMBINED DATASET")
print("=" * 60)
print(f"\nOriginal dataset size: {len(df)}")
print(f"New samples generated: {len(new_df)}")
print(f"Combined dataset size: {len(combined_df)}")

print("\nFruit distribution in combined dataset:")
print(combined_df['target'].value_counts())

# Save the combined dataset
output_file = 'fruit_data_augmented.csv'
combined_df.to_csv(output_file, index=False)

print(f"\n✓ Augmented dataset saved to '{output_file}'")

# Display first and last rows to verify
print("\nFirst 5 rows of combined dataset:")
print(combined_df.head())
print("\nLast 5 rows (new samples):")
print(combined_df.tail())
