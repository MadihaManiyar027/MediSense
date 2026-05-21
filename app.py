from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import numpy as np
from geopy.distance import geodesic
from difflib import get_close_matches

app = Flask(__name__)

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ---------------- LOAD SYMPTOMS ----------------
def load_symptoms():
    path = os.path.join(DATA_DIR, "Symptoms.csv")
    df = pd.read_csv(path, encoding="cp1252", encoding_errors="replace")
    df.columns = [c.strip().lower() for c in df.columns]
    disease_col = next((c for c in df.columns if "disease" in c), df.columns[0])
    symptom_col = next((c for c in df.columns if "symptom" in c), df.columns[1])
    df = df.rename(columns={disease_col: "disease", symptom_col: "symptoms"})
    df = df.dropna()
    df["disease"] = df["disease"].astype(str).str.strip()
    df["symptoms"] = df["symptoms"].astype(str).str.lower().str.strip()
    return df


# ---------------- LOAD HOSPITALS ----------------
def load_hospitals():
    path = os.path.join(DATA_DIR, "Hospitals.csv")
    df = pd.read_csv(path, encoding="cp1252", encoding_errors="replace")
    df.columns = [c.strip().lower().replace("ï»¿", "") for c in df.columns]

    col_map = {
        "name": "name",
        "address": "address",
        "latitude": "latitude",
        "longitude": "longitude",
        "phone": "phone",
        "rating (/5)": "rating",
        "type": "type",
        "specialisation": "specialisation",
        "working hours": "working_hours",
        "google maps": "maps_link"
    }
    df = df.rename(columns=col_map)

    needed_cols = ["name", "address", "latitude", "longitude", "phone",
                   "rating", "type", "specialisation", "working_hours", "maps_link"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["specialisation"] = df["specialisation"].astype(str).str.lower()

    df["specialisation"] = df["specialisation"].str.replace(
        r"orthodonti\w*", "dental-orthodontics", regex=True)
    df["specialisation"] = df["specialisation"].str.replace(
        r"orthopaedi\w*", "orthopaedics", regex=True)

    return df


symptoms_df = load_symptoms()
hospitals_df = load_hospitals()


# ---------------- BUILD SYMPTOM VOCAB ----------------
all_symptoms = sorted(set(
    s.strip().lower()
    for row in symptoms_df["symptoms"]
    for s in str(row).split(",")
    if s.strip()
))
symptom_index = {s: i for i, s in enumerate(all_symptoms)}


# ---------------- CLEAN INPUT WITH FUZZY SYMPTOM CORRECTION ----------------
def clean_input(raw):
    """
    Cleans and fuzzy-corrects each symptom the user types.
    e.g. "fver" -> "fever", "headche" -> "headache"
    """
    tokens = [s.strip().lower() for s in raw.split(",") if s.strip()]
    corrected = []
    for token in tokens:
        if token in symptom_index:
            # Exact match — use as is
            corrected.append(token)
        else:
            # Fuzzy match against known symptoms
            close = get_close_matches(token, all_symptoms, n=1, cutoff=0.6)
            if close:
                corrected.append(close[0])
            else:
                # Keep original even if unrecognised — won't affect vector
                corrected.append(token)
    return corrected


# ---------------- VECTOR ----------------
def to_vector(symptom_list):
    vec = np.zeros(len(all_symptoms))
    for s in symptom_list:
        if s in symptom_index:
            vec[symptom_index[s]] = 1
    return vec


# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------- DISEASE PREDICTION ----------------
def predict_disease(user_symptoms):
    user_vec = to_vector(user_symptoms)
    results = []
    for _, row in symptoms_df.iterrows():
        db_symptoms = [s.strip().lower() for s in str(row["symptoms"]).split(",")]
        db_vec = to_vector(db_symptoms)
        score = cosine_similarity(user_vec, db_vec)
        if score > 0:
            matched = list(set(user_symptoms) & set(db_symptoms))
            results.append({
                "disease": row["disease"],
                "score": float(score),
                "matched": matched
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:2]


# ================================================================
# DISEASE → SPECIALISATION MAP  (covers every disease in dataset)
# ================================================================
DISEASE_SPECIALTY_MAP = {

    # ── GENERAL / INTERNAL MEDICINE ──────────────────────────────
    "malaria":                               ["general", "medicine", "multispeciality"],
    "dengue":                                ["general", "medicine", "multispeciality"],
    "typhoid":                               ["general", "medicine", "multispeciality"],
    "covid-19":                              ["general", "medicine", "multispeciality"],
    "viral fever":                           ["general", "medicine", "multispeciality"],
    "viral infection":                       ["general", "medicine", "multispeciality"],
    "chikungunya":                           ["general", "medicine", "multispeciality"],
    "food poisoning":                        ["general", "medicine", "multispeciality"],
    "gastroenteritis":                       ["general", "medicine", "multispeciality"],
    "cholera":                               ["general", "medicine", "multispeciality"],
    "measles":                               ["general", "medicine", "multispeciality"],
    "chickenpox":                            ["general", "medicine", "multispeciality"],
    "infection":                             ["general", "medicine", "multispeciality"],
    "parasitic infection":                   ["general", "medicine", "multispeciality"],
    "immunodeficiency":                      ["general", "medicine", "multispeciality"],
    "hiv":                                   ["general", "medicine", "multispeciality"],
    "dehydration":                           ["general", "medicine", "multispeciality"],
    "severe infection (sepsis":              ["general", "medicine", "multispeciality"],
    "sepsis (in elderly/infants)":           ["general", "medicine", "multispeciality"],
    "scarlet fever":                         ["general", "medicine", "multispeciality"],
    "tetanus":                               ["general", "surgery"],
    "motion sickness":                       ["general", "ent"],
    "allergic reaction":                     ["general", "medicine"],
    "allergies":                             ["general", "medicine"],
    "edema":                                 ["general", "medicine"],
    "jaundice":                              ["gastro", "general"],
    "malnutrition":                          ["general", "medicine"],
    "nutritional deficiency":                ["general", "medicine"],
    "vitamin deficiency":                    ["general", "medicine"],
    "vitamin c deficiency":                  ["general", "medicine"],
    "vitamin a deficiency":                  ["general", "medicine"],
    "zinc deficiency":                       ["general", "medicine"],
    "electrolyte imbalance":                 ["general", "medicine"],
    "vitamin d deficiency":                  ["orthopaedics", "medicine"],
    "vitamin b12 deficiency":                ["neuro", "medicine"],
    "vitamin k deficiency":                  ["haematology", "medicine"],
    "iron deficiency":                       ["haematology", "medicine"],
    "iron deficiency anemia":                ["haematology", "medicine"],
    "low blood pressure":                    ["cardiology", "general"],
    "low blood sugar":                       ["general", "endocrinology"],
    "shock":                                 ["general", "surgery"],
    "obesity":                               ["general", "endocrinology"],
    "chronic fatigue syndrome":              ["medicine", "psychiatry"],
    "delirium":                              ["neuro", "general"],
    "delirium tremens":                      ["psychiatry", "general"],
    "withdrawal":                            ["psychiatry", "general"],

    # ── DIABETES / ENDOCRINOLOGY ─────────────────────────────────
    "diabetes":                              ["endocrinology", "medicine"],
    "diabetes mellitus":                     ["endocrinology", "medicine"],
    "diabetes insipidus":                    ["endocrinology", "medicine"],
    "hyperthyroidism":                       ["endocrinology", "medicine"],
    "hypothyroidism":                        ["endocrinology", "medicine"],
    "thyroid disorders":                     ["endocrinology", "medicine"],
    "thyroid cancer":                        ["endocrinology", "oncology"],
    "thyroid nodule":                        ["endocrinology", "medicine"],
    "goiter":                                ["endocrinology", "medicine"],
    "cushing's syndrome":                    ["endocrinology", "medicine"],
    "adrenal disorders":                     ["endocrinology", "medicine"],
    "congenital adrenal hyperplasia":        ["endocrinology", "paediatric"],
    "pheochromocytoma":                      ["endocrinology", "medicine"],
    "hypoglycemia":                          ["endocrinology", "general"],
    "insulin resistance":                    ["endocrinology", "medicine"],
    "growth hormone deficiency":             ["endocrinology", "paediatric"],
    "precocious puberty":                    ["endocrinology", "paediatric"],
    "hormone-secreting tumors":              ["endocrinology", "oncology"],
    "polycystic ovary syndrome":             ["gynaecology", "endocrinology"],
    "polycystic ovary syndrome (pcos)":      ["gynaecology", "endocrinology"],
    "menopause":                             ["gynaecology", "endocrinology"],
    "gigantism":                             ["endocrinology", "paediatric"],
    "klinefelter syndrome":                  ["endocrinology", "paediatric"],
    "hypogonadism":                          ["endocrinology", "paediatric"],
    "constitutional delay":                  ["endocrinology", "paediatric"],
    "marfan syndrome":                       ["orthopaedics", "paediatric"],
    "prolactinoma":                          ["endocrinology", "gynaecology"],
    "graves' disease":                       ["endocrinology", "ophthalmology"],

    # ── PULMONOLOGY / CHEST ──────────────────────────────────────
    "asthma":                                ["pulmo", "chest"],
    "pneumonia":                             ["pulmo", "chest"],
    "copd":                                  ["pulmo", "chest"],
    "tuberculosis":                          ["pulmo", "chest"],
    "bronchitis":                            ["pulmo", "chest"],
    "lung cancer":                           ["pulmo", "oncology"],
    "pulmonary embolism":                    ["pulmo", "chest", "cardiology"],
    "cystic fibrosis":                       ["pulmo", "chest", "paediatric"],
    "sleep apnea":                           ["pulmo", "ent"],
    "asthma attack":                         ["pulmo", "chest"],
    "severe copd":                           ["pulmo", "chest"],
    "severe lung disease":                   ["pulmo", "chest"],
    "chronic lung disease":                  ["pulmo", "chest"],
    "bronchiectasis":                        ["pulmo", "chest"],
    "respiratory muscle weakness":           ["pulmo", "neuro"],
    "bacterial infection (bronchitis":       ["pulmo", "chest"],

    # ── CARDIOLOGY ───────────────────────────────────────────────
    "hypertension":                          ["cardiology", "medicine"],
    "angina":                                ["cardiology"],
    "heart failure":                         ["cardiology"],
    "arrhythmia":                            ["cardiology"],
    "heart attack":                          ["cardiology"],
    "pericarditis":                          ["cardiology"],
    "congenital heart disease":              ["cardiology", "paediatric"],
    "cardiovascular disease":                ["cardiology"],
    "venous insufficiency":                  ["cardiology", "vascular"],
    "chronic venous insufficiency":          ["cardiology", "vascular"],
    "peripheral vascular disease":           ["vascular", "cardiology"],
    "peripheral artery disease":             ["vascular", "cardiology"],
    "raynaud's disease":                     ["vascular", "medicine"],
    "deep vein thrombosis (dvt)":            ["vascular", "cardiology"],
    "superior vena cava obstruction":        ["cardiology", "oncology"],
    "vasculitis":                            ["medicine", "rheumatology"],
    "arrhythmia (svt":                       ["cardiology"],
    "heart block":                           ["cardiology"],
    "sick sinus syndrome":                   ["cardiology"],
    "syncope":                               ["cardiology", "neuro"],
    "vasovagal syncope":                     ["cardiology", "neuro"],
    "orthostatic hypotension":               ["cardiology", "neuro"],
    "kawasaki disease":                      ["cardiology", "paediatric"],

    # ── NEUROLOGY ────────────────────────────────────────────────
    "migraine":                              ["neuro"],
    "epilepsy":                              ["neuro"],
    "meningitis":                            ["neuro"],
    "encephalitis":                          ["neuro"],
    "parkinson's disease":                   ["neuro"],
    "multiple sclerosis":                    ["neuro"],
    "dementia":                              ["neuro", "psychiatry"],
    "alzheimer's disease":                   ["neuro"],
    "brain tumor":                           ["neuro", "oncology"],
    "bell's palsy":                          ["neuro"],
    "guillain-barré syndrome":               ["neuro"],
    "peripheral neuropathy":                 ["neuro"],
    "trigeminal neuralgia":                  ["neuro"],
    "essential tremor":                      ["neuro"],
    "myasthenia gravis":                     ["neuro"],
    "normal pressure hydrocephalus":         ["neuro"],
    "ataxia":                                ["neuro"],
    "cerebral palsy":                        ["neuro", "paediatric"],
    "hydrocephalus":                         ["neuro", "paediatric"],
    "concussion":                            ["neuro"],
    "subdural hematoma":                     ["neuro", "neurosurgery"],
    "cluster headache":                      ["neuro"],
    "tension headache":                      ["neuro"],
    "migraine aura":                         ["neuro"],
    "stroke":                                ["neuro", "cardiology"],
    "transient ischemic attack":             ["neuro", "cardiology"],
    "benign paroxysmal positional vertigo (bppv)": ["neuro", "ent"],
    "vertigo":                               ["neuro", "ent"],
    "inner ear infection":                   ["ent", "neuro"],
    "meniere's disease":                     ["ent", "neuro"],
    "restless legs syndrome":                ["neuro", "general"],
    "charcot-marie-tooth disease":           ["neuro", "orthopaedics"],
    "muscular dystrophy":                    ["neuro", "paediatric"],
    "rett syndrome":                         ["neuro", "paediatric"],
    "neurogenic bladder":                    ["neuro", "urology"],
    "optic neuritis":                        ["neuro", "ophthalmology"],

    # ── ORTHOPAEDICS ─────────────────────────────────────────────
    "arthritis":                             ["orthopaedics"],
    "rheumatoid arthritis":                  ["orthopaedics", "rheumatology"],
    "osteoarthritis":                        ["orthopaedics"],
    "gout":                                  ["orthopaedics", "rheumatology"],
    "reactive arthritis":                    ["orthopaedics", "rheumatology"],
    "juvenile idiopathic arthritis":         ["orthopaedics", "paediatric"],
    "juvenile arthritis":                    ["orthopaedics", "paediatric"],
    "ankylosing spondylitis":                ["orthopaedics", "rheumatology"],
    "fracture":                              ["orthopaedics"],
    "herniated disc":                        ["orthopaedics"],
    "muscle strain":                         ["orthopaedics"],
    "sciatica":                              ["orthopaedics"],
    "frozen shoulder":                       ["orthopaedics"],
    "rotator cuff injury":                   ["orthopaedics"],
    "bursitis":                              ["orthopaedics"],
    "ligament injury":                       ["orthopaedics"],
    "ligament tear (acl/pcl)":               ["orthopaedics"],
    "meniscus tear":                         ["orthopaedics"],
    "osteoporosis":                          ["orthopaedics"],
    "paget's disease":                       ["orthopaedics"],
    "scoliosis":                             ["orthopaedics"],
    "kyphosis":                              ["orthopaedics"],
    "degenerative disc disease":             ["orthopaedics"],
    "cervical spondylosis":                  ["orthopaedics"],
    "tennis elbow":                          ["orthopaedics"],
    "golfer's elbow":                        ["orthopaedics"],
    "carpal tunnel syndrome":                ["orthopaedics"],
    "tendinitis":                            ["orthopaedics"],
    "trigger finger":                        ["orthopaedics"],
    "plantar fasciitis":                     ["orthopaedics"],
    "achilles tendinitis":                   ["orthopaedics"],
    "heel spurs":                            ["orthopaedics"],
    "hip dysplasia":                         ["orthopaedics", "paediatric"],
    "leg length discrepancy":                ["orthopaedics"],
    "blount's disease":                      ["orthopaedics", "paediatric"],
    "rickets":                               ["orthopaedics", "paediatric"],
    "tight achilles tendon":                 ["orthopaedics"],
    "patellar dislocation":                  ["orthopaedics"],
    "patellar instability":                  ["orthopaedics"],
    "bone cancer":                           ["orthopaedics", "oncology"],
    "rhabdomyolysis":                        ["orthopaedics", "medicine"],
    "fibromyalgia":                          ["rheumatology", "medicine"],
    "toxic synovitis":                       ["orthopaedics", "paediatric"],
    "torticollis":                           ["orthopaedics", "paediatric"],

    # ── DERMATOLOGY ──────────────────────────────────────────────
    "eczema":                                ["derma"],
    "fungal infection":                      ["derma"],
    "psoriasis":                             ["derma"],
    "allergic dermatitis":                   ["derma"],
    "contact dermatitis":                    ["derma"],
    "scabies":                               ["derma"],
    "acne":                                  ["derma"],
    "melanoma":                              ["derma", "oncology"],
    "skin cancer":                           ["derma", "oncology"],
    "basal cell carcinoma":                  ["derma", "oncology"],
    "vitiligo":                              ["derma"],
    "melasma":                               ["derma"],
    "tinea versicolor":                      ["derma"],
    "rosacea":                               ["derma"],
    "lupus":                                 ["derma", "rheumatology"],
    "impetigo":                              ["derma"],
    "shingles":                              ["derma", "medicine"],
    "alopecia":                              ["derma"],
    "telogen effluvium":                     ["derma"],
    "androgenetic alopecia":                 ["derma"],
    "seborrheic dermatitis":                 ["derma"],
    "folliculitis":                          ["derma"],
    "hidradenitis suppurativa":              ["derma"],
    "pemphigus":                             ["derma"],
    "scleroderma":                           ["derma", "rheumatology"],
    "lymphedema":                            ["derma", "vascular"],
    "chronic eczema":                        ["derma"],
    "burns":                                 ["derma", "surgery"],
    "sunburn":                               ["derma"],
    "herpes":                                ["derma", "medicine"],
    "herpes simplex virus":                  ["derma", "medicine"],
    "syphilis":                              ["derma", "medicine"],
    "chancroid":                             ["derma", "medicine"],
    "human papillomavirus (hpv)":            ["derma", "gynaecology"],
    "warts":                                 ["derma"],
    "blisters":                              ["derma"],
    "calluses":                              ["derma"],
    "corns":                                 ["derma"],
    "varicose veins":                        ["vascular", "derma"],
    "spider veins":                          ["vascular", "derma"],
    "cold sores":                            ["derma"],
    "hives":                                 ["derma", "medicine"],
    "urticaria":                             ["derma"],
    "systemic lupus erythematosus (sle)":    ["rheumatology", "derma"],
    "autoimmune disorder":                   ["rheumatology", "medicine"],
    "inflammatory conditions":               ["rheumatology", "medicine"],
    "autoimmune condition":                  ["rheumatology", "medicine"],

    # ── ENT ──────────────────────────────────────────────────────
    "sinusitis":                             ["ent"],
    "tonsillitis":                           ["ent"],
    "common cold":                           ["ent", "general"],
    "pharyngitis":                           ["ent"],
    "allergic rhinitis":                     ["ent"],
    "rhinitis":                              ["ent"],
    "ear infection":                         ["ent"],
    "swimmer's ear":                         ["ent"],
    "laryngitis":                            ["ent"],
    "vocal cord nodules":                    ["ent"],
    "tinnitus":                              ["ent"],
    "age-related hearing loss":              ["ent"],
    "acoustic neuroma":                      ["ent", "neuro"],
    "chronic sinusitis":                     ["ent"],
    "nasal polyps":                          ["ent"],
    "adenoid hypertrophy":                   ["ent", "paediatric"],
    "deviated nasal septum":                 ["ent"],
    "eustachian tube dysfunction":           ["ent"],
    "mastoiditis":                           ["ent"],
    "otitis media":                          ["ent"],
    "otitis externa (swimmer's ear)":        ["ent"],
    "peritonsillar abscess":                 ["ent"],
    "chronic tonsillitis":                   ["ent"],
    "throat cancer":                         ["ent", "oncology"],
    "enlarged tonsils":                      ["ent"],
    "pulsatile tinnitus":                    ["ent", "vascular"],
    "infectious mononucleosis":              ["ent", "medicine"],
    "vocal cord paralysis":                  ["ent"],
    "globus sensation":                      ["ent"],
    "esophageal stricture":                  ["gastro", "ent"],

    # ── GASTROENTEROLOGY ─────────────────────────────────────────
    "gastritis":                             ["gastro"],
    "appendicitis":                          ["gastro", "surgery"],
    "gastroesophageal reflux":               ["gastro"],
    "gastroesophageal reflux disease (gerd)": ["gastro"],
    "irritable bowel syndrome":              ["gastro"],
    "irritable bowel syndrome (ibs)":        ["gastro"],
    "inflammatory bowel disease (ibd)":      ["gastro"],
    "colorectal cancer":                     ["gastro", "oncology"],
    "hemorrhoids":                           ["gastro", "surgery"],
    "dysentery":                             ["gastro"],
    "hepatitis":                             ["gastro"],
    "liver cirrhosis":                       ["gastro"],
    "alcoholic liver disease":               ["gastro"],
    "liver disease":                         ["gastro"],
    "gallstones":                            ["gastro", "surgery"],
    "cholecystitis":                         ["gastro", "surgery"],
    "pancreatitis":                          ["gastro"],
    "hiatal hernia":                         ["gastro", "surgery"],
    "functional dyspepsia":                  ["gastro"],
    "malabsorption":                         ["gastro"],
    "celiac disease":                        ["gastro"],
    "esophageal cancer":                     ["gastro", "oncology"],
    "achalasia":                             ["gastro"],
    "peptic ulcer":                          ["gastro"],
    "stomach ulcer":                         ["gastro"],
    "stomach cancer":                        ["gastro", "oncology"],
    "anal fissure":                          ["gastro", "surgery"],
    "anal fistula":                          ["gastro", "surgery"],
    "rectal prolapse":                       ["gastro", "surgery"],
    "bowel perforation":                     ["gastro", "surgery"],
    "peritonitis":                           ["gastro", "surgery"],
    "constipation":                          ["gastro"],
    "lactose intolerance":                   ["gastro"],
    "pinworms":                              ["gastro"],
    "proctitis":                             ["gastro"],
    "esophagitis":                           ["gastro"],
    "gastroparesis":                         ["gastro"],
    "bile duct obstruction":                 ["gastro", "surgery"],
    "ulcerative colitis":                    ["gastro"],
    "lower gastrointestinal bleeding (hemorrhoids": ["gastro", "surgery"],
    "abdominal migraine":                    ["gastro", "paediatric"],

    # ── NEPHROLOGY / UROLOGY ─────────────────────────────────────
    "kidney stones":                         ["urology", "nephro"],
    "urinary tract infection":               ["urology", "nephro"],
    "kidney disease":                        ["nephro"],
    "kidney infection":                      ["nephro"],
    "kidney failure":                        ["nephro"],
    "pyelonephritis":                        ["nephro"],
    "bladder cancer":                        ["urology", "oncology"],
    "urinary obstruction":                   ["urology"],
    "urinary retention":                     ["urology"],
    "urethral stricture":                    ["urology"],
    "prostatitis":                           ["urology"],
    "prostate enlargement":                  ["urology"],
    "prostate enlargement (bph)":            ["urology"],
    "overactive bladder":                    ["urology"],
    "stress incontinence":                   ["urology", "gynaecology"],
    "interstitial cystitis":                 ["urology"],
    "hematuria (blood in urine)":            ["urology", "nephro"],
    "urethritis":                            ["urology"],
    "epididymitis":                          ["urology"],
    "testicular torsion":                    ["urology", "surgery"],
    "orchitis":                              ["urology"],
    "hydrocele":                             ["urology"],
    "varicocele":                            ["urology"],
    "testicular cancer":                     ["urology", "oncology"],
    "seminal vesiculitis":                   ["urology"],
    "erectile dysfunction":                  ["urology"],
    "balanitis":                             ["urology"],
    "delayed bladder control":               ["urology", "paediatric"],
    "overflow incontinence":                 ["urology"],
    "bladder muscle weakness":               ["urology"],

    # ── DENTAL / ORAL ─────────────────────────────────────────────
    "toothache":                             ["dental"],
    "gum disease":                           ["dental"],
    "dental cavity":                         ["dental"],
    "dental abscess":                        ["dental"],
    "dental infection":                      ["dental"],
    "dental caries":                         ["dental"],
    "gingivitis":                            ["dental"],
    "periodontal disease":                   ["dental"],
    "abscess":                               ["dental", "surgery"],
    "temporomandibular joint disorder":      ["dental"],
    "bruxism":                               ["dental"],
    "dental malocclusion":                   ["dental"],
    "dental erosion":                        ["dental"],
    "oral cancer":                           ["dental", "oncology"],
    "oral candidiasis":                      ["dental"],
    "oral thrush":                           ["dental", "general"],
    "oral lichen planus":                    ["dental"],
    "leukoplakia":                           ["dental"],
    "erythroplakia":                         ["dental"],
    "glossitis":                             ["dental"],
    "angular cheilitis":                     ["dental"],
    "burning mouth syndrome":                ["dental"],
    "canker sores":                          ["dental"],
    "poor dental hygiene":                   ["dental"],
    "dental problems":                       ["dental"],
    "gum recession":                         ["dental"],
    "cracked tooth":                         ["dental"],
    "teeth sensitivity":                     ["dental"],
    "black spots on teeth":                  ["dental"],
    "tooth discoloration":                   ["dental"],
    "frequent cavities":                     ["dental"],
    "sores on gums":                         ["dental"],
    "white patches in mouth":                ["dental"],
    "red patches in mouth":                  ["dental"],
    "geographic tongue":                     ["dental"],
    "swollen gums":                          ["dental"],
    "bleeding gums":                         ["dental"],
    "painful gums":                          ["dental"],

    # ── GYNAECOLOGY / OBSTETRICS ──────────────────────────────────
    "pregnancy":                             ["gynaecology", "maternity", "obstetrics"],
    "endometriosis":                         ["gynaecology"],
    "pelvic inflammatory disease":           ["gynaecology"],
    "vaginitis":                             ["gynaecology"],
    "cervical cancer":                       ["gynaecology", "oncology"],
    "uterine fibroids":                      ["gynaecology"],
    "adenomyosis":                           ["gynaecology"],
    "ovarian cysts":                         ["gynaecology"],
    "fibrocystic disease":                   ["gynaecology"],
    "breast cancer":                         ["gynaecology", "oncology"],
    "mastitis":                              ["gynaecology"],
    "inflammatory breast cancer":            ["gynaecology", "oncology"],
    "fat necrosis":                          ["gynaecology", "surgery"],
    "uterine polyps":                        ["gynaecology"],
    "endometrial cancer":                    ["gynaecology", "oncology"],
    "vaginal atrophy":                       ["gynaecology"],
    "yeast infection":                       ["gynaecology"],
    "bacterial vaginosis":                   ["gynaecology"],
    "trichomoniasis":                        ["gynaecology"],
    "sexually transmitted infection":        ["gynaecology", "urology", "medicine"],
    "fibrocystic changes":                   ["gynaecology"],
    "duct ectasia":                          ["gynaecology"],
    "hormonal imbalance":                    ["gynaecology", "endocrinology"],
    "irregular periods":                     ["gynaecology"],
    "bleeding between periods":              ["gynaecology"],
    "bleeding after menopause":              ["gynaecology"],
    "heavy menstrual bleeding":              ["gynaecology"],
    "painful periods (dysmenorrhea)":        ["gynaecology"],

    # ── PSYCHIATRY / MENTAL HEALTH ───────────────────────────────
    "depression":                            ["psychiatry", "mental health"],
    "anxiety":                               ["psychiatry", "mental health"],
    "stress":                                ["psychiatry", "mental health"],
    "bipolar disorder":                      ["psychiatry", "mental health"],
    "schizophrenia":                         ["psychiatry", "mental health"],
    "ocd":                                   ["psychiatry", "mental health"],
    "obsessive-compulsive disorder (ocd)":   ["psychiatry", "mental health"],
    "ptsd":                                  ["psychiatry", "mental health"],
    "post-traumatic stress disorder (ptsd)": ["psychiatry", "mental health"],
    "acute stress disorder":                 ["psychiatry", "mental health"],
    "panic disorder":                        ["psychiatry", "mental health"],
    "agoraphobia":                           ["psychiatry", "mental health"],
    "social anxiety disorder":               ["psychiatry", "mental health"],
    "generalized anxiety disorder":          ["psychiatry", "mental health"],
    "specific phobia":                       ["psychiatry", "mental health"],
    "dissociative disorder":                 ["psychiatry", "mental health"],
    "personality disorder":                  ["psychiatry", "mental health"],
    "bipolar disorder (mania)":              ["psychiatry", "mental health"],
    "delusional disorder":                   ["psychiatry", "mental health"],
    "depersonalization disorder":            ["psychiatry", "mental health"],
    "derealization disorder":                ["psychiatry", "mental health"],
    "eating disorder":                       ["psychiatry", "mental health"],
    "anhedonia":                             ["psychiatry", "mental health"],
    "substance use disorder":                ["psychiatry", "mental health"],
    "addiction":                             ["psychiatry", "mental health"],
    "substance abuse":                       ["psychiatry", "mental health"],
    "insomnia":                              ["psychiatry", "general"],
    "sleep deprivation":                     ["psychiatry", "general"],
    "sleep disorders":                       ["psychiatry", "general"],
    "trichotillomania":                      ["psychiatry", "paediatric"],
    "emotional dysregulation":               ["psychiatry", "paediatric"],

    # ── OPHTHALMOLOGY ────────────────────────────────────────────
    "cataract":                              ["ophthalmology", "eye"],
    "cataracts":                             ["ophthalmology", "eye"],
    "glaucoma":                              ["ophthalmology", "eye"],
    "diabetic retinopathy":                  ["ophthalmology", "eye"],
    "conjunctivitis":                        ["ophthalmology", "eye"],
    "corneal abrasion":                      ["ophthalmology", "eye"],
    "iritis":                                ["ophthalmology", "eye"],
    "retinal detachment":                    ["ophthalmology", "eye"],
    "macular degeneration":                  ["ophthalmology", "eye"],
    "strabismus":                            ["ophthalmology", "eye"],
    "amblyopia":                             ["ophthalmology", "eye"],
    "orbital cellulitis":                    ["ophthalmology", "eye"],
    "orbital tumor":                         ["ophthalmology", "oncology"],
    "computer vision syndrome":              ["ophthalmology", "eye"],
    "sjögren's syndrome":                    ["ophthalmology", "rheumatology"],
    "blepharitis":                           ["ophthalmology", "eye"],
    "dacryocystitis":                        ["ophthalmology", "eye"],
    "dry eye syndrome":                      ["ophthalmology", "eye"],
    "retinitis pigmentosa":                  ["ophthalmology", "eye"],
    "uveitis":                               ["ophthalmology", "eye"],
    "corneal ulcer":                         ["ophthalmology", "eye"],
    "presbyopia (age-related farsightedness)": ["ophthalmology", "eye"],
    "keratoconus":                           ["ophthalmology", "eye"],
    "refractive errors":                     ["ophthalmology", "eye"],
    "vision problems":                       ["ophthalmology", "eye"],
    "lazy eye (amblyopia)":                  ["ophthalmology", "eye"],
    "retinal tear":                          ["ophthalmology", "eye"],
    "posterior vitreous detachment":         ["ophthalmology", "eye"],
    "blocked tear duct":                     ["ophthalmology", "eye"],
    "uveitis":                               ["ophthalmology", "eye"],
    "corneal ulcer":                         ["ophthalmology", "eye"],

    # ── PAEDIATRICS ──────────────────────────────────────────────
    "autism spectrum disorder":              ["paediatric"],
    "adhd":                                  ["paediatric", "psychiatry"],
    "down syndrome":                         ["paediatric"],
    "colic":                                 ["paediatric"],
    "failure to thrive":                     ["paediatric"],
    "developmental delay":                   ["paediatric"],
    "periodic fever syndromes":              ["paediatric"],
    "lead poisoning":                        ["paediatric"],
    "intellectual disability":               ["paediatric", "psychiatry"],
    "learning disabilities":                 ["paediatric", "psychiatry"],
    "tic disorders":                         ["paediatric", "psychiatry"],
    "transient tic disorder":                ["paediatric", "psychiatry"],
    "tourette syndrome":                     ["paediatric", "psychiatry"],
    "oppositional defiant disorder":         ["paediatric", "psychiatry"],
    "conduct disorder":                      ["paediatric", "psychiatry"],
    "normal development":                    ["paediatric"],
    "normal growth":                         ["paediatric"],
    "normal variant":                        ["paediatric"],
    "psychogenic polydipsia":                ["paediatric", "psychiatry"],
    "metabolic disorders":                   ["paediatric", "endocrinology"],
    "genetic disorders":                     ["paediatric"],
    "microcephaly":                          ["paediatric", "neuro"],
    "chronic disease":                       ["paediatric", "general"],
    "specific learning disorder (dyslexia":  ["paediatric", "psychiatry"],

    # ── HAEMATOLOGY / ONCOLOGY ───────────────────────────────────
    "anemia":                                ["haematology", "medicine"],
    "lymphoma":                              ["haematology", "oncology"],
    "leukemia":                              ["haematology", "oncology"],
    "cancer":                                ["oncology"],
    "platelet disorders":                    ["haematology"],
    "bleeding disorders":                    ["haematology"],
    "blood disorders":                       ["haematology"],
    "thrombocytopenia":                      ["haematology"],

    # ── RHEUMATOLOGY ─────────────────────────────────────────────
    "systemic lupus erythematosus (sle)":    ["rheumatology", "derma"],
    "autoimmune disorder":                   ["rheumatology", "medicine"],
    "autoimmune condition":                  ["rheumatology", "medicine"],
    "inflammatory conditions":               ["rheumatology", "medicine"],
}


# ── FUZZY DISEASE → SPECIALISATION ───────────────────────────────────────────
from difflib import get_close_matches

def get_specialties_for_disease(disease):
    """Return speciality keywords. Handles typos via fuzzy matching."""
    d = disease.strip().lower()

    # 1. Exact match
    if d in DISEASE_SPECIALTY_MAP:
        return DISEASE_SPECIALTY_MAP[d]

    # 2. Substring match
    for key, val in DISEASE_SPECIALTY_MAP.items():
        if key in d or d in key:
            return val

    # 3. Fuzzy match (handles misspellings)
    all_keys = list(DISEASE_SPECIALTY_MAP.keys())
    close = get_close_matches(d, all_keys, n=1, cutoff=0.6)
    if close:
        return DISEASE_SPECIALTY_MAP[close[0]]

    # 4. Fallback
    return ["general", "medicine", "multispeciality"]

# ---------------- GROUP PREDICTIONS BY SPECIALISATION ----------------
def group_predictions_by_speciality(predictions):
    """
    Group diseases that share identical speciality keywords so hospitals
    are recommended only once per group.
    """
    groups = []
    for pred in predictions:
        specialties     = get_specialties_for_disease(pred["disease"])
        specialties_key = tuple(sorted(specialties))

        found = False
        for grp in groups:
            if tuple(sorted(grp["specialties"])) == specialties_key:
                grp["diseases"].append(pred["disease"])
                grp["confidences"].append(round(pred["score"], 3))
                grp["matched_symptoms"] = list(
                    set(grp["matched_symptoms"]) | set(pred["matched"])
                )
                found = True
                break

        if not found:
            groups.append({
                "diseases":         [pred["disease"]],
                "confidences":      [round(pred["score"], 3)],
                "matched_symptoms": list(pred["matched"]),
                "specialties":      specialties
            })
    return groups


# ---------------- HOSPITAL MATCHING ----------------
def find_hospitals(specialties, user_lat=None, user_lon=None):
    """
    Find hospitals matching given speciality keywords.
    Returns top 3 best + next 3 nearby.
    """
    pattern  = "|".join(specialties)
    filtered = hospitals_df[
        hospitals_df["specialisation"].str.contains(pattern, case=False, na=False)
    ].copy()

    if filtered.empty:
        filtered = hospitals_df.copy()

    if user_lat and user_lon:
        distances = []
        for _, row in filtered.iterrows():
            try:
                dist = geodesic(
                    (float(user_lat), float(user_lon)),
                    (float(row["latitude"]), float(row["longitude"]))
                ).km
            except Exception:
                dist = 9999
            distances.append(dist)
        filtered["distance_km"] = distances
        filtered = filtered.sort_values(
            by=["distance_km", "rating"], ascending=[True, False]
        )
    else:
        filtered = filtered.sort_values(by="rating", ascending=False)
        filtered["distance_km"] = 0

    filtered = filtered.drop_duplicates(subset=["name", "address"])

    OUTPUT_COLS = [
        "name", "address", "phone", "rating",
        "type", "specialisation", "working_hours",
        "maps_link", "distance_km"
    ]

    return {
        "top_hospitals":    filtered.head(3)[OUTPUT_COLS].to_dict(orient="records"),
        "nearby_hospitals": filtered.iloc[3:6][OUTPUT_COLS].to_dict(orient="records")
    }


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json()
    raw      = data.get("symptoms", "").strip()
    user_lat = data.get("latitude")
    user_lon = data.get("longitude")

    if not raw:
        return jsonify({"error": "Please enter symptoms"}), 400

    # Fuzzy-corrected symptom list
    user_symptoms         = clean_input(raw)
    original_tokens       = [s.strip().lower() for s in raw.split(",") if s.strip()]
    corrected_symptoms    = user_symptoms  # already corrected inside clean_input

    predictions = predict_disease(user_symptoms)

    if not predictions:
        return jsonify({"error": "No disease found"}), 404

    groups = group_predictions_by_speciality(predictions)

    output = []
    for grp in groups:
        hospitals = find_hospitals(grp["specialties"], user_lat, user_lon)
        output.append({
            "diseases":         grp["diseases"],
            "confidences":      grp["confidences"],
            "matched_symptoms": grp["matched_symptoms"],
            "specialties_used": grp["specialties"],
            "top_hospitals":    hospitals["top_hospitals"],
            "nearby_hospitals": hospitals["nearby_hospitals"]
        })

    return jsonify({
        "input_symptoms":     original_tokens,
        "corrected_symptoms": corrected_symptoms,   # shows what was fuzzy-corrected
        "predictions":        output
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)