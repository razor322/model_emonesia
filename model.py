import os
import subprocess
import torch
import pandas as pd
from flask import Flask, request, jsonify, send_file
from transformers import BertTokenizer, BertForSequenceClassification
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from collections import Counter
from datetime import datetime

# ==========================
# 1. INISIALISASI FLASK APP
# ==========================
app = Flask(__name__)

# ==========================
# 2. LOAD MODEL & TOKENIZER
# ==========================
model_name = "gybran/model-prediction-social-media-emotion"
tokenizer = BertTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()
id2label = model.config.id2label

# ==========================
# 3. KONFIGURASI LABEL & HARI
# ==========================
LABEL_MAPPING = {
    "anger": "Marah",
    "sadness": "Sedih",
    "Neutral": "Netral",
    "Joy": "Bahagia",
    "fear": "Takut"
}

hari_dict = {
    "Monday": "Senin",
    "Tuesday": "Selasa",
    "Wednesday": "Rabu",
    "Thursday": "Kamis",
    "Friday": "Jumat",
    "Saturday": "Sabtu",
    "Sunday": "Minggu"
}

# ==========================
# 4. UTILITAS
# ==========================
def convert_twitter_date(twitter_date_str):
    try:
        dt = datetime.strptime(twitter_date_str, "%a %b %d %H:%M:%S %z %Y")
        hari = hari_dict[dt.strftime("%A")]
        return f"{hari}, {dt.day} {dt.strftime('%B')} {dt.year} {dt.strftime('%H:%M')}"
    except Exception:
        return twitter_date_str

def convert_excel_to_pdf_with_wrap(excel_path, keywords, start_clean, end_clean, timestamp_now):
    hasil_dir = os.path.join(os.getcwd(), 'hasil')
    os.makedirs(hasil_dir, exist_ok=True)
    df = pd.read_excel(excel_path)

    if "created_at" in df.columns:
        df["tanggal"] = df["created_at"].astype(str).apply(convert_twitter_date)
    else:
        df["tanggal"] = ""

    df["predicted_label_indonesia"] = df["predicted_label"].map(LABEL_MAPPING)
    label_counts = Counter(df["predicted_label_indonesia"])
    label_order = [label for label, _ in label_counts.most_common()]
    df["predicted_label_indonesia"] = pd.Categorical(df["predicted_label_indonesia"], categories=label_order, ordered=True)
    df = df.sort_values("predicted_label_indonesia")
    df.insert(0, "No", range(1, len(df)+1))

    keyword_clean = keywords.lower().replace(" ", "_")
    pdf_output_path = os.path.join(hasil_dir, f"hasil_prediksi_{keyword_clean}_{start_clean}_{end_clean}_{timestamp_now}.pdf")

    doc = SimpleDocTemplate(pdf_output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(f"\ud83d\udcca Hasil Analisis Emosi '{keywords.title()}'", styles["Heading2"]),
        Spacer(1, 12)
    ]

    data = [["No", "Tanggal & Waktu", "Teks", "Hasil Prediksi"]]
    for _, row in df.iterrows():
        teks_paragraf = Paragraph(str(row["full_text"]), styles["Normal"])
        data.append([row["No"], row["tanggal"], teks_paragraf, row["predicted_label_indonesia"]])

    table = Table(data, colWidths=[30, 120, 300, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)

    doc.build(elements)
    return pdf_output_path

def predict_emotions_from_csv(keywords, file_path, start_clean, end_clean, timestamp_now):
    hasil_dir = os.path.join(os.getcwd(), 'hasil')
    os.makedirs(hasil_dir, exist_ok=True)
    df = pd.read_csv(file_path)

    if "full_text" not in df.columns:
        return None, None, "Kolom 'full_text' tidak ditemukan."

    texts = df["full_text"].astype(str).tolist()
    preds, probs = [], []

    for i in range(0, len(texts), 32):
        batch = tokenizer(texts[i:i+32], truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**batch).logits
            preds += logits.argmax(dim=1).tolist()
            probs += logits.softmax(dim=1).max(dim=1).values.tolist()

    df["predicted_label"] = [id2label[p] for p in preds]
    df["predicted_label_indonesia"] = [LABEL_MAPPING[l] for l in df["predicted_label"]]
    df["confidence"] = probs

    keyword_clean = keywords.lower().replace(" ", "_")
    output_excel = os.path.join(hasil_dir, f"hasil_prediksi_{keyword_clean}_{start_clean}_{end_clean}_{timestamp_now}.xlsx")
    df.to_excel(output_excel, index=False)

    pdf_path = convert_excel_to_pdf_with_wrap(output_excel, keywords, start_clean, end_clean, timestamp_now)

    label_counts = Counter(df["predicted_label_indonesia"])
    summary = {
        "label": label_counts.most_common(1)[0][0] if label_counts else "-",
        "count": label_counts.most_common(1)[0][1] if label_counts else 0,
        "percentage": (label_counts.most_common(1)[0][1] / len(df)) * 100 if label_counts else 0,
        "emosi_summary": [{"label_emosi": l, "jumlah": label_counts.get(l, 0)} for l in ["Bahagia", "Marah", "Sedih", "Netral", "Takut"]]
    }

    return pdf_path, summary, None

# ==========================
# 5. ENDPOINT PREDIKSI
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    keyword = data.get("keyword")
    start_date = data.get("startDate")
    end_date = data.get("endDate")
    typeData = "--tab LATEST" if data.get("type") == "latest" else ""

    if not keyword or not start_date or not end_date:
        return jsonify({"error": "Parameter keyword, start_date, end_date wajib diisi"}), 400

    folder = os.path.join(os.getcwd(), "tweets-data")
    os.makedirs(folder, exist_ok=True)

    timestamp_now = datetime.now().strftime("%H%M%S")
    start_clean = start_date.replace("-", "")
    end_clean = end_date.replace("-", "")
    keyword_clean = keyword.lower().replace(" ", "_")
    filename = f"hasil_scraping_{keyword_clean}_{start_clean}_{end_clean}_{timestamp_now}.csv"
    filepath = os.path.join(folder, filename)

    search_string = f"{keyword} since:{start_date} until:{end_date} lang:id"
    command = f"npx -y tweet-harvest@2.6.1 -o \"{filepath}\" -s \"{search_string}\" {typeData} -l 100 --token f9404361514ee7d5dadc20c68d3a31c8bfd1526f"

    try:
        subprocess.run(command, shell=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("⏰ Timeout saat scraping")

    if os.path.exists(filepath) and not pd.read_csv(filepath).empty:
        pdf_path, summary, error = predict_emotions_from_csv(keyword, filepath, start_clean, end_clean, timestamp_now)
        if error:
            return jsonify({"error": error}), 500

        return jsonify({
            "keyword": keyword,
            "output_file": pdf_path,
            "dominant_label": summary["label"],
            "jumlah_dominan": summary["count"],
            "persentase": f"{summary['percentage']:.2f}%",
            "detail_emosi": summary["emosi_summary"],
            "total_emosi": sum([i["jumlah"] for i in summary["emosi_summary"]]),
            "pdf_url": f"/download-pdf?filename={os.path.basename(pdf_path)}"
        })

    return jsonify({"error": "Gagal mengambil data tweet."}), 500

# ==========================
# 6. DOWNLOAD PDF
# ==========================
@app.route("/download-pdf")
def download_pdf():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Nama file tidak tersedia"}), 400

    file_path = os.path.join(os.getcwd(), "hasil", filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File tidak ditemukan"}), 404

    return send_file(file_path, as_attachment=True)

# ==========================
# 7. JALANKAN APP
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
