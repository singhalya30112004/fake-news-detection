import pandas as pd
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
fake_path = os.path.join(script_dir, '../Dataset/Fake.csv')
true_path = os.path.join(script_dir, '../Dataset/True.csv')
output_path = os.path.join(script_dir, '../Dataset/Combined_News.csv')


# Read CSVs
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# Add 'label' column: 0 for fake, 1 for real
fake_df['label'] = 0
true_df['label'] = 1

# Combine both dataframes
combined_df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Shuffle the data
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined CSV
combined_df.to_csv(output_path, index=False)

print(f"Combined dataset saved to: {output_path}")