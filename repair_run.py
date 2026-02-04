import os
import shutil
import json
import os.path as osp
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations,
)
from ai_scientist.llm import create_client
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_plotting import aggregate_plots

# Define the run to repair
idea_dir = "experiments/2026-02-04_00-10-05_tensor_fold_transform_attempt_0"
model_agg = "gemini-3-flash-preview"
model_writeup = "gemini-3-pro-preview"
model_citation = "gemini-3-flash-preview"
model_review = "gemini-3-pro-preview"

print(f"Repairing run in: {idea_dir}")

def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.token_counts, f, indent=4)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.interactions, f, indent=4)

# 1. Prepare experiment results for aggregation (Skipping - already done/error prone)
# experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
# if osp.exists(experiment_results_dir):
#     print("Copying experiment results to top level...")
#     try:
#         shutil.copytree(
#             experiment_results_dir,
#             osp.join(idea_dir, "experiment_results"),
#             dirs_exist_ok=True,
#         )
#     except Exception as e:
#         print(f"Warning: Failed to copy experiment results: {e}")

# 2. Plot Aggregation (Skipping - already done)
# print("\n--- Starting Plot Aggregation ---")
# try:
#     aggregate_plots(base_folder=idea_dir, model=model_agg)
# except Exception as e:
#     print(f"Error in aggregate_plots: {e}")

# 3. Citation Gathering (Skipping - already done)
# print("\n--- Gathering Citations ---")
# try:
#     citations_text = gather_citations(
#         idea_dir,
#         num_cite_rounds=20,
#         small_model=model_citation,
#     )
# except Exception as e:
#     print(f"Error in gather_citations: {e}")
#     citations_text = ""
citations_text = None # Load from cache

# 4. Writeup
print("\n--- Starting Writeup ---")
try:
    writeup_success = perform_icbinb_writeup(
        base_folder=idea_dir,
        small_model=model_citation,
        big_model=model_writeup,
        page_limit=4,
        citations_text=citations_text,
    )
except Exception as e:
    print(f"Error in perform_icbinb_writeup: {e}")
    writeup_success = False

if writeup_success:
    print("Writeup successful.")
else:
    print("Writeup failed.")

# 5. Review (if writeup succeeded)
def find_pdf_path_for_review(idea_dir):
    for f in os.listdir(idea_dir):
        if f.endswith(".pdf"):
            return osp.join(idea_dir, f)
    return ""

if writeup_success:
    print("\n--- Starting Review ---")
    pdf_path = find_pdf_path_for_review(idea_dir)
    if os.path.exists(pdf_path):
        print(f"Paper found at: {pdf_path}")
        try:
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(model_review)
            review_text = perform_review(paper_content, client_model, client)
            
            with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            print("Paper review completed.")
        except Exception as e:
            print(f"Error in review: {e}")
    else:
        print("No PDF found for review.")

# Cleanup
if os.path.exists(osp.join(idea_dir, "experiment_results")):
    shutil.rmtree(osp.join(idea_dir, "experiment_results"))

save_token_tracker(idea_dir)
print("\nRepair process finished.")
