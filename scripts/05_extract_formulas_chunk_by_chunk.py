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
        "Gib alle Schritte an, auch Zwischenrechnungen, Grenzwerte, Umformungen und alle Integrale. "
        "Nutze `align*`-Umgebungen, um mehrzeilige Rechnungen darzustellen. "
        "Kennzeichne jeden Titel von den Formeln mit Fettschrift (\\textbf{...}), sofern vorhanden. "
        "Gib **nur die exakten Formeln und Herleitungen aus dem Kontext**. "
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

def safe_latex(text):
    text = text.replace("\\", "\\\\")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("#", "\\#")
    text = text.replace("_", "\\_")
    text = text.replace("$", "\\$")
    return text

def main():
    start = int(input("Startseite: ").strip())
    end = int(input("Endseite: ").strip())

    tex_output = []
    for page in range(start, end + 1):
        filename = f"page_{page:03}.txt"
        filepath = os.path.join(CHUNK_DIR, filename)

        if not os.path.exists(filepath):
            print(f"Seite {page} nicht gefunden.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            context = f.read()

        if not context.strip():
            print(f"Seite {page} ist leer.")
            continue

        print(f"Bearbeite Seite {page}...")
        try:
            antwort_roh = ask_gpt(context, "Gib alle mathematischen Formeln dieser Seite inklusive Titel aus.")
            with open(os.path.join(OUTPUT_DIR, f"seite_{page:03}_rohantwort.txt"), "w", encoding="utf-8") as dbg:
                dbg.write(antwort_roh)

            # naive Sanity-Check: bricht zu lange align* Blöcke auf
            antwort_roh = re.sub(r'(\\begin\{align\*})', r'\[\1', antwort_roh)
            antwort_roh = re.sub(r'(\\end\{align\*})', r'\1\]', antwort_roh)

            latex_block = f"\\subsection*{{Seite {page}}}\n\n{antwort_roh}\n"
            tex_output.append(latex_block)

        except Exception as e:
            print(f"Fehler auf Seite {page}: {e}")
            tex_output.append(f"\\subsection*{{Seite {page}}}\n\nFehler beim Verarbeiten dieser Seite.\\\\\n")

    # Ausgabe zusammenfassen
    tex_path = os.path.join(OUTPUT_DIR, f"formelzusammenfassung_s{start}_s{end}.tex")
    pdf_path = os.path.join(OUTPUT_DIR, f"formelzusammenfassung_s{start}_s{end}.pdf")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"\documentclass{article}" + "\n")
        f.write(r"\usepackage[utf8]{inputenc}" + "\n")
        f.write(r"\usepackage{amsmath,amssymb,amsthm}" + "\n")
        f.write(r"\usepackage[a4paper,margin=2.5cm]{geometry}" + "\n")
        f.write(r"\setlength{\parskip}{1ex}" + "\n")
        f.write(r"\setlength{\parindent}{0pt}" + "\n")
        f.write(r"\begin{document}" + "\n\n")
        f.write(r"\section*{Formelzusammenfassung Seiten " + f"{start}--{end}" + "}" + "\n\n")
        f.write("\n".join(tex_output))
        f.write("\n" + r"\end{document}")

    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex_path], cwd=OUTPUT_DIR, check=True)
        if os.path.exists(pdf_path):
            print(f"\nPDF erstellt: {pdf_path}")
        else:
            raise FileNotFoundError("PDF wurde nicht erzeugt.")
    except Exception as e:
        print(f"Fehler bei PDF-Erzeugung: {e}")

if __name__ == "__main__":
    main()
