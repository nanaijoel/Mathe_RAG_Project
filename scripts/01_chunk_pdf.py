import os
import fitz  # PyMuPDF

# ==== KONFIGURATION ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "Skript_Math_2_EDIT.pdf")
CHUNK_DIR = os.path.join(BASE_DIR, "chunks")

os.makedirs(CHUNK_DIR, exist_ok=True)

# ==== PDF laden ====
doc = fitz.open(PDF_PATH)
print(f"PDF geladen: {PDF_PATH}")
print(f"Seitenanzahl: {len(doc)}")

# ==== Seitenweise speichern ====
for i, page in enumerate(doc):
    text = page.get_text()
    if not text.strip():
        continue  # leere Seiten überspringen

    chunk_path = os.path.join(CHUNK_DIR, f"page_{i + 1:03}.txt")
    with open(chunk_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Seite {i + 1} gespeichert als: {chunk_path}")

print("\nAlle Seiten als Text-Chunks gespeichert.")
