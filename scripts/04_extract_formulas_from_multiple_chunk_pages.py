import os
import re
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# Setup
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK_DIR = os.path.join(BASE_DIR, "chunks")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = "gpt-4-turbo"

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_gpt(context, user_question):
    system_prompt = (
        "Du bist ein mathematischer Assistent. "
        "Extrahiere **alle mathematischen Formeln und vollständigen Rechenschritte** aus dem gegebenen Text. "
        "Gruppiere die Ausgabe **nach Seitenzahlen** und verwende je Seite einen LaTeX-Abschnitt: \\\\subsection*{Seite XX}. "
        "Nutze `align*`-Umgebungen für mehrzeilige Rechnungen und gib alle Gleichungen, Titel und Herleitungen vollständig wieder."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_question}"}
    ]
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def extract_page_range(frage: str):
    match = re.search(r"Seite[n]*\s*(\d+)\s*[-\u2013\s]+(\d+)", frage, re.IGNORECASE)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return range(start, end + 1)
    else:
        return None

def main():
    fragen = [
        "Extrahiere alle Formeln inklusive ihrer Titel aus Seite 126-130",
        "Extrahiere alle Formeln inklusive ihrer Titel aus Seite 131-135"
    ]

    for frage in fragen:
        print(f"\nFrage: {frage}")
        seitenbereich = extract_page_range(frage)
        if not seitenbereich:
            print("Seitenbereich nicht erkannt.")
            continue

        context_blocks = []
        for seite in seitenbereich:
            filename = f"page_{seite:03}.txt"
            filepath = os.path.join(CHUNK_DIR, filename)

            if not os.path.exists(filepath):
                print(f"Seite {seite} nicht gefunden.")
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                context_blocks.append(f"Seite {seite}\n---\n{content}")

        if not context_blocks:
            print("Keine gültigen Seiteninhalte gefunden.")
            continue

        gesamtkontext = "\n\n".join(context_blocks)
        print("Sende gesamten Kontext an GPT...")
        try:
            antwort = ask_gpt(gesamtkontext, frage)
            start, end = seitenbereich.start, seitenbereich.stop - 1
            raw_output_path = os.path.join(OUTPUT_DIR, f"gpt_rohantwort_s{start}_s{end}.txt")
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(antwort)
        except Exception as e:
            print(f"Fehler bei GPT-Abfrage: {e}")
            continue

        tex_path = os.path.join(OUTPUT_DIR, f"formelzusammenfassung_s{start}_s{end}.tex")
        pdf_path = os.path.join(OUTPUT_DIR, f"formelzusammenfassung_s{start}_s{end}.pdf")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(r"\documentclass{article}\n")
            f.write(r"\usepackage[utf8]{inputenc}\n")
            f.write(r"\usepackage{amsmath,amssymb,amsthm}\n")
            f.write(r"\usepackage[a4paper,margin=2.5cm]{geometry}\n")
            f.write(r"\setlength{\parskip}{1ex}\n")
            f.write(r"\setlength{\parindent}{0pt}\n")
            f.write(r"\begin{document}\n\n")
            f.write(rf"\section*{{Formelzusammenfassung Seiten {start}--{end}}}\n\n")
            f.write(antwort)
            f.write("\n\\end{document}")

        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path], cwd=OUTPUT_DIR, check=True)
            if os.path.exists(pdf_path):
                print(f"PDF erstellt: {pdf_path}")
            else:
                raise FileNotFoundError("PDF wurde nicht erzeugt.")
        except Exception as e:
            print(f"Fehler bei PDF-Erzeugung: {e}")

if __name__ == "__main__":
    main()
