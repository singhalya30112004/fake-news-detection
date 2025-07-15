import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
main_data_path = os.path.join(script_dir, "../Dataset/Cleaned_News.csv")
feedback_data_path = os.path.join(script_dir, "../Dataset/Feedback_Data.csv")
output_path = os.path.join(script_dir, "../Dataset/Combined_News.csv")

# Load datasets
main_df = pd.read_csv(main_data_path)
feedback_df = pd.read_csv(feedback_data_path)

# Combine and deduplicate
combined_df = pd.concat([main_df, feedback_df], ignore_index=True)
combined_df.drop_duplicates(subset="text", keep="last", inplace=True)

# Save combined file
combined_df.to_csv(output_path, index=False)
print(f"Combined dataset saved to {output_path}")
print(f"Total rows after merge: {combined_df.shape[0]}")