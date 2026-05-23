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

    df = pd.read_csv(
        path,
        encoding="cp1252",
        encoding_errors="replace"
    )

    df.columns = [c.strip().lower() for c in df.columns]

    disease_col = next(
        (c for c in df.columns if "disease" in c),
        df.columns[0]
    )

    symptom_col = next(
        (c for c in df.columns if "symptom" in c),
        df.columns[1]
    )

    df = df.rename(columns={
        disease_col: "disease",
        symptom_col: "symptoms"
    })

    df = df.dropna()

    df["disease"] = (
        df["disease"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["symptoms"] = (
        df["symptoms"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


# ---------------- LOAD HOSPITALS ----------------
def load_hospitals():

    path = os.path.join(DATA_DIR, "Hospitals.csv")

    df = pd.read_csv(
        path,
        encoding="cp1252",
        encoding_errors="replace"
    )

    df.columns = [
        c.strip().lower().replace("ï»¿", "")
        for c in df.columns
    ]

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

    needed_cols = [
        "name",
        "address",
        "latitude",
        "longitude",
        "phone",
        "rating",
        "type",
        "specialisation",
        "working_hours",
        "maps_link"
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    df = df.fillna("")

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    ).fillna(0)

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # ---------------- NORMALIZE SPECIALISATIONS ----------------
    df["specialisation"] = (
        df["specialisation"]
        .astype(str)
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace("/", ",", regex=False)
        .str.replace("|", ",", regex=False)
    )

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

symptom_index = {
    s: i
    for i, s in enumerate(all_symptoms)
}


# ---------------- CLEAN INPUT ----------------
def clean_input(raw):

    tokens = [
        s.strip().lower()
        for s in raw.split(",")
        if s.strip()
    ]

    corrected = []

    for token in tokens:

        if token in symptom_index:
            corrected.append(token)

        else:
            close = get_close_matches(
                token,
                all_symptoms,
                n=1,
                cutoff=0.6
            )

            if close:
                corrected.append(close[0])
            else:
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

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ---------------- DISEASE PREDICTION ----------------
def predict_disease(user_symptoms):

    user_vec = to_vector(user_symptoms)

    results = []

    for _, row in symptoms_df.iterrows():

        db_symptoms = [
            s.strip().lower()
            for s in str(row["symptoms"]).split(",")
        ]

        db_vec = to_vector(db_symptoms)

        score = cosine_similarity(
            user_vec,
            db_vec
        )

        if score > 0:

            matched = list(
                set(user_symptoms) &
                set(db_symptoms)
            )

            results.append({
                "disease": row["disease"],
                "score": float(score),
                "matched": matched
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:2]


# ---------------- DISEASE SPECIALTY MAP ----------------
DISEASE_SPECIALTY_MAP = {

    # FIXED MOTION SICKNESS
    "motion sickness": ["ent", "vertigo", "neurology"],

    "vertigo": ["ent", "neurology"],
    "benign paroxysmal positional vertigo (bppv)": ["ent", "neurology"],
    "inner ear infection": ["ent"],
    "meniere's disease": ["ent"],

    "migraine": ["neurology"],
    "sinusitis": ["ent"],
    "ear infection": ["ent"],

    "arthritis": ["orthopaedics"],
    "fracture": ["orthopaedics"],

    "toothache": ["dental"],
    "gum disease": ["dental"],

    "asthma": ["pulmo"],
    "diabetes": ["endocrinology"],

    "gastritis": ["gastro"],
    "appendicitis": ["gastro"],

    "depression": ["psychiatry"],

    "cataract": ["ophthalmology"],

    "pregnancy": ["gynaecology"],

    "anemia": ["haematology"],

    "eczema": ["derma"]
}


# ---------------- SPECIALTY FINDER ----------------
def get_specialties_for_disease(disease):

    d = disease.strip().lower()

    # exact match
    if d in DISEASE_SPECIALTY_MAP:
        return DISEASE_SPECIALTY_MAP[d]

    # partial match
    for key, val in DISEASE_SPECIALTY_MAP.items():
        if key in d or d in key:
            return val

    # fuzzy match
    close = get_close_matches(
        d,
        list(DISEASE_SPECIALTY_MAP.keys()),
        n=1,
        cutoff=0.6
    )

    if close:
        return DISEASE_SPECIALTY_MAP[close[0]]

    # fallback
    return ["general"]


# ---------------- GROUP PREDICTIONS ----------------
def group_predictions_by_speciality(predictions):

    groups = []

    for pred in predictions:

        specialties = get_specialties_for_disease(
            pred["disease"]
        )

        specialties_key = tuple(
            sorted(specialties)
        )

        found = False

        for grp in groups:

            if tuple(sorted(grp["specialties"])) == specialties_key:

                grp["diseases"].append(
                    pred["disease"]
                )

                grp["confidences"].append(
                    round(pred["score"], 3)
                )

                grp["matched_symptoms"] = list(
                    set(grp["matched_symptoms"]) |
                    set(pred["matched"])
                )

                found = True
                break

        if not found:

            groups.append({
                "diseases": [pred["disease"]],
                "confidences": [round(pred["score"], 3)],
                "matched_symptoms": list(pred["matched"]),
                "specialties": specialties
            })

    return groups


# ---------------- HOSPITAL MATCHING ----------------
# ---------------- HOSPITAL MATCHING ----------------
def find_hospitals(specialties, user_lat=None, user_lon=None):

    filtered_rows = []

    specialties = [
        s.lower().strip()
        for s in specialties
    ]

    for _, row in hospitals_df.iterrows():

        hospital_spec = str(
            row["specialisation"]
        ).lower()

        # normalize text
        hospital_spec = (
            hospital_spec
            .replace("-", " ")
            .replace("/", ",")
            .replace("|", ",")
        )

        hospital_specs = [
            s.strip()
            for s in hospital_spec.split(",")
            if s.strip()
        ]

        matched_score = 0

        for spec in specialties:

            for hs in hospital_specs:

                # EXACT MATCH
                if spec == hs:
                    matched_score += 5

                # WORD MATCH
                elif spec in hs:
                    matched_score += 2

                # PARTIAL MATCH
                elif hs in spec:
                    matched_score += 1

        # only keep meaningful matches
        if matched_score >= 2:

            row = row.copy()
            row["match_score"] = matched_score

            filtered_rows.append(row)

    filtered = pd.DataFrame(filtered_rows)

    # ---------------- NO MATCH FALLBACK ----------------
    if filtered.empty:

        return {
            "top_hospitals": [],
            "nearby_hospitals": []
        }

    # ---------------- DISTANCE ----------------
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
            by=["match_score", "distance_km", "rating"],
            ascending=[False, True, False]
        )

    else:

        filtered["distance_km"] = 0

        filtered = filtered.sort_values(
            by=["match_score", "rating"],
            ascending=[False, False]
        )

    # remove duplicates
    filtered = filtered.drop_duplicates(
        subset=["name", "address"]
    )

    OUTPUT_COLS = [
        "name",
        "address",
        "phone",
        "rating",
        "type",
        "specialisation",
        "working_hours",
        "maps_link",
        "distance_km"
    ]

    return {

        "top_hospitals":
            filtered.head(3)[OUTPUT_COLS]
            .to_dict(orient="records"),

        "nearby_hospitals":
            filtered.iloc[3:6][OUTPUT_COLS]
            .to_dict(orient="records")
    }
    # ---------------- DISTANCE ----------------
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
            by=["distance_km", "rating"],
            ascending=[True, False]
        )

    else:

        filtered = filtered.sort_values(
            by="rating",
            ascending=False
        )

        filtered["distance_km"] = 0

    # remove duplicates
    filtered = filtered.drop_duplicates(
        subset=["name", "address"]
    )

    OUTPUT_COLS = [
        "name",
        "address",
        "phone",
        "rating",
        "type",
        "specialisation",
        "working_hours",
        "maps_link",
        "distance_km"
    ]

    return {

        "top_hospitals":
            filtered.head(3)[OUTPUT_COLS]
            .to_dict(orient="records"),

        "nearby_hospitals":
            filtered.iloc[3:6][OUTPUT_COLS]
            .to_dict(orient="records")
    }


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    raw = data.get(
        "symptoms",
        ""
    ).strip()

    user_lat = data.get("latitude")
    user_lon = data.get("longitude")

    if not raw:
        return jsonify({
            "error": "Please enter symptoms"
        }), 400

    user_symptoms = clean_input(raw)

    original_tokens = [
        s.strip().lower()
        for s in raw.split(",")
        if s.strip()
    ]

    predictions = predict_disease(
        user_symptoms
    )

    if not predictions:
        return jsonify({
            "error": "No disease found"
        }), 404

    groups = group_predictions_by_speciality(
        predictions
    )

    output = []

    for grp in groups:

        hospitals = find_hospitals(
            grp["specialties"],
            user_lat,
            user_lon
        )

        output.append({

            "diseases":
                grp["diseases"],

            "confidences":
                grp["confidences"],

            "matched_symptoms":
                grp["matched_symptoms"],

            "specialties_used":
                grp["specialties"],

            "top_hospitals":
                hospitals["top_hospitals"],

            "nearby_hospitals":
                hospitals["nearby_hospitals"]
        })

    return jsonify({

        "input_symptoms":
            original_tokens,

        "corrected_symptoms":
            user_symptoms,

        "predictions":
            output
    })


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
