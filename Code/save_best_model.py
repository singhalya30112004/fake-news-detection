import shutil
import os

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))

best_model_src = os.path.join(script_dir, "../Models/Random_Forest_Model.pkl")
vectoriser_src = os.path.join(script_dir, "../Models/tfidf_vectoriser.pkl")

best_model_dest = os.path.join(script_dir, "../Models/final_model.pkl")
vectoriser_dest = os.path.join(script_dir, "../Models/final_vectoriser.pkl")

# Copy files
shutil.copy(best_model_src, best_model_dest)
shutil.copy(vectoriser_src, vectoriser_dest)

print("Best model and vectoriser saved as final_model.pkl and final_vectoriser.pkl")