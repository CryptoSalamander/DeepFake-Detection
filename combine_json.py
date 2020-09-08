import json
import os
from glob import glob
from pathlib import Path

result = {}
for json_path in glob(os.path.join("../dfdc_train_all/dfdc_test", "*/metadata.json")):
    dir = Path(json_path).parent
    with open(json_path, "r") as f:
        jsondata = json.load(f)
        result.update(jsondata)

with open('output.json','w') as f:
    json.dump(result,f,indent='\t')

print("Combined Finished : output.json")
