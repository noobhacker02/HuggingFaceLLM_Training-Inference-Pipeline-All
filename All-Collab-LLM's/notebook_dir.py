import json
import glob
import os

# 🔹 Set the folder where your .ipynb files are stored in Drive
notebook_dir = "C:\\Users\\Talha Shaikh\\Downloads\\All-Collab-LLM's"


# Find all ipynb files
for nb_file in glob.glob(os.path.join(notebook_dir, "*.ipynb")):
    with open(nb_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If metadata has "widgets", remove it
    if "widgets" in data.get("metadata", {}):
        print(f"Cleaning widgets from: {os.path.basename(nb_file)}")
        del data["metadata"]["widgets"]

        # Save back
        with open(nb_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        print(f"No widget metadata in: {os.path.basename(nb_file)}")

print("✅ All notebooks cleaned. Outputs are safe.")
