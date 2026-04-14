import json

f = r'd:\in-class\Advance_AI2\advanced_ai_project\notebooks\02_model_training.ipynb'
try:
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                if 'YOLO_MODEL = "yolov10n.pt"' in line:
                    line = line.replace('YOLO_MODEL = "yolov10n.pt"', 'YOLO_MODEL = str(PRETRAINED_DIR / "yolov10n.pt")')
                elif "YOLO_MODEL = 'yolov10n.pt'" in line:
                    line = line.replace("YOLO_MODEL = 'yolov10n.pt'", "YOLO_MODEL = str(PRETRAINED_DIR / 'yolov10n.pt')")
                new_source.append(line)
            cell['source'] = new_source

    with open(f, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=1)
        
    print("Notebook updated successfully.")
except Exception as e:
    print("Error:", e)
