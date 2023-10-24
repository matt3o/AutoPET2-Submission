import json
import sys

if __name__=="__main__":
    file_name = sys.argv[1]

    with open(file_name, "r") as json_file:
        results = json.load(json_file)
   
    print(f"{len(results['tumor'])} tumor clicks, {len(results['background'])} background clicks")
