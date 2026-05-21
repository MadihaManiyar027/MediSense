# 🏥 MediSense — AI Symptom Checker

A Flask web app that predicts diseases from symptoms and recommends hospitals.

---

## 📁 Project Structure

```
medisense/
│
├── app.py                  ← Flask backend (all logic lives here)
├── requirements.txt        ← Python packages to install
├── README.md               ← This file
│
├── data/
│   ├── Symptoms.xlsx       ← Disease + Symptoms dataset (649 records)
│   └── Hospitals.xlsx      ← Hospital info dataset (63 records)
│
└── templates/
    └── index.html          ← Frontend (HTML + CSS + JavaScript)
```

---

## ⚙️ Setup & Run Instructions (Step by Step)

### Step 1 — Install Python
Make sure Python 3.9+ is installed.
Check: open terminal and type `python --version`
Download from: https://www.python.org/downloads/

### Step 2 — Open the project in VS Code
- Open VS Code
- Go to File → Open Folder → select the `medisense` folder

### Step 3 — Open the terminal in VS Code
- Press Ctrl + ` (backtick) or go to Terminal → New Terminal

### Step 4 — Create a virtual environment (recommended)
```bash
python -m venv venv
```

Activate it:
- Windows:   venv\Scripts\activate
- Mac/Linux: source venv/bin/activate

You'll see `(venv)` appear in your terminal — that means it's active.

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```

This installs Flask, pandas, and openpyxl.

### Step 6 — Run the app
```bash
python app.py
```

You should see:
```
Loaded 649 symptom records, 63 hospital records
 * Running on http://127.0.0.1:5000
```

### Step 7 — Open in browser
Go to: http://127.0.0.1:5000

---

## 🧪 How to Use MediSense

1. Type symptoms in the text box, separated by commas
   Example: `Fever, Cough, Headache`
2. Click "Analyse Symptoms"
3. See disease predictions ranked by match score
4. See recommended hospitals for each disease
5. Click "Open in Google Maps" to navigate to a hospital

---

## 📊 Dataset Formats

### Symptoms.xlsx (must have these columns)
| Disease     | Symptoms                            |
|-------------|-------------------------------------|
| Malaria     | Fever, Jaundice, Fever that comes.. |
| Dengue      | Fever, Joint pain                   |
| Tuberculosis| Persistent cough, Night sweats, ... |

### Hospitals.xlsx (must have these columns)
| Name | Address | Phone | Rating (/5) | Type | Specialisation | Working Hours | Google Maps |
|------|---------|-------|-------------|------|----------------|---------------|-------------|

---

## ❌ Common Errors & Fixes

### Error: `ModuleNotFoundError: No module named 'flask'`
Fix: Run `pip install -r requirements.txt`
Make sure your virtual environment is activated.

### Error: `FileNotFoundError: Symptoms.xlsx not found`
Fix: Make sure both `.xlsx` files are inside the `data/` folder.

### Error: `Port 5000 already in use`
Fix: Either close whatever is using port 5000, or change the port:
```python
app.run(debug=True, port=5001)  # change this line at bottom of app.py
```

### Error: `TemplateNotFound: index.html`
Fix: Make sure `index.html` is inside the `templates/` folder (not in root).

### Error: Flask shows blank page / no results
Fix: Open browser DevTools (F12) → Console tab to see JavaScript errors.
Also check your terminal for Python errors.

### Error: `pip` is not recognized
Fix: Try `pip3` instead of `pip`

---

## 💡 Tips for Beginners

- Always keep the terminal open while using the app (Flask runs there)
- Press Ctrl+C in the terminal to stop the Flask server
- If you edit `app.py`, the server auto-reloads (debug=True does this)
- If you edit `index.html`, just refresh the browser
- Use Ctrl+Enter in the symptom box as a keyboard shortcut to search

