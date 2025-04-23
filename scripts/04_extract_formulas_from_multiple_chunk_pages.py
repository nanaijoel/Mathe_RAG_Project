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

def get_main_keyword(text):
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that extracts the most relevant technical keyword "
                    "from a math-related user question. "
                    "Return only a single uppercase keyword, suitable for a filename. "
                    "Avoid generic words like 'bitte', 'frage', 'thema', etc. "
                    "Replace German umlauts and avoid spaces or special characters."
                )},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        keyword = response.choices[0].message.content.strip().lower()
        keyword = keyword.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        keyword = re.sub(r"\W+", "_", keyword)
        return keyword[:40]
    except Exception as e:
        print(f"GPT-Keyword-Erkennung fehlgeschlagen: {e}")
        return "antwort"

def markdown_to_latex(text):
    lines = text.splitlines()
    latex_lines = []
    in_align = False
    in_itemize = False
    in_matrix = False

    for line in lines:
        raw_line = line.strip()

        if r"\begin{bmatrix}" in raw_line or r"\begin{matrix}" in raw_line:
            in_matrix = True
        if r"\end{bmatrix}" in raw_line or r"\end{matrix}" in raw_line:
            in_matrix = False

        if not in_matrix and not in_align:
            raw_line = raw_line.replace("&", r"\&")

        if re.match(r"^#{1,6}\s", raw_line):
            if in_align:
                latex_lines.append(r"\end{align*}")
                in_align = False
            if in_itemize:
                latex_lines.append(r"\end{itemize}")
                in_itemize = False
            content = raw_line.lstrip('#').strip()
            latex_lines.append(r"\textbf{" + content + r"}\\")
            continue

        if raw_line.startswith("- "):
            if in_align:
                latex_lines.append(r"\end{align*}")
                in_align = False
            if not in_itemize:
                latex_lines.append(r"\begin{itemize}")
                in_itemize = True
            latex_lines.append(r"\item " + raw_line[2:])
            continue

        if raw_line.count("=") >= 1 or raw_line.count("+") >= 3 or any(char in raw_line for char in ["\\", "∫", "Σ", "π", "α"]):
            if in_itemize:
                latex_lines.append(r"\end{itemize}")
                in_itemize = False
            if not in_align:
                latex_lines.append(r"\begin{align*}")
                in_align = True
            segments = re.split(r"\s{2,}", raw_line)
            for segment in segments:
                cleaned = segment.strip()
                if cleaned:
                    latex_lines.append(cleaned + r" \\\\[1ex]")
            continue

        if in_align:
            latex_lines.append(r"\end{align*}")
            in_align = False
        if in_itemize:
            latex_lines.append(r"\end{itemize}")
            in_itemize = False

        latex_lines.append(raw_line)

    if in_align:
        latex_lines.append(r"\end{align*}")
    if in_itemize:
        latex_lines.append(r"\end{itemize}")

    return "\n".join(latex_lines)

def ask_gpt(context, user_question):
    system_prompt = (
        "Du bist ein mathematischer Assistent. "
        "Extrahiere **alle mathematischen Formeln und vollständigen Rechenschritte** aus dem gegebenen Text. "
        "Gib alle Schritte an, auch Zwischenrechnungen, Grenzwerte, Umformungen und alle Integrale. "
        "Nutze `align*`-Umgebungen, um mehrzeilige Rechnungen darzustellen. "
        "Kennzeichne jeden Titel von den Formeln mit Fettschrift (\\textbf{...}), sofern vorhanden. "
        "Gib **nur die exakten Formeln und Herleitungen aus dem Kontext**. "
        "Wenn du eine Gleichung schreibst, nicht nur Anfang und Endresultat, sondern die ganze Rechnung des Skripts. "
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
        "Extrahiere alle Formeln inklusive ihrer Titel aus Seite 77-78"
    ]

    texts = []
    pages = []
    for filename in sorted(os.listdir(CHUNK_DIR)):
        match = re.match(r"page_(\d{3})\.txt", filename)
        if match:
            page_num = int(match.group(1))
            path = os.path.join(CHUNK_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    texts.append(content)
                    pages.append(page_num)

    for frage in fragen:
        print(f"\nFrage: {frage}")
        keyword = get_main_keyword(frage)
        seitenbereich = extract_page_range(frage)
        print(f"Erkannter Seitenbereich: {seitenbereich}")

        if not seitenbereich:
            print("Konnte Seitenbereich nicht erkennen.")
            continue

        chunks = [text for text, page in zip(texts, pages) if page in seitenbereich]
        print(f"Chunks für Seitenbereich gefunden: {len(chunks)}")

        if not chunks:
            print("Keine passenden Chunks im gewünschten Seitenbereich gefunden.")
            continue

        context_text = "\n\n---\n\n".join(chunks)
        antwort_roh = ask_gpt(context_text, frage)
        antwort_latex = markdown_to_latex(antwort_roh)

        seiten_suffix = f"_s{seitenbereich.start}_s{seitenbereich.stop - 1}"
        filename = f"{keyword}{seiten_suffix}"

        tex_path = os.path.join(OUTPUT_DIR, f"{filename}.tex")
        pdf_path = os.path.join(OUTPUT_DIR, f"{filename}.pdf")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(r"\documentclass{article}" + "\n")
            f.write(r"\usepackage[utf8]{inputenc}" + "\n")
            f.write(r"\usepackage{amsmath,amssymb,amsthm}" + "\n")
            f.write(r"\usepackage[a4paper,margin=2.5cm]{geometry}" + "\n")
            f.write(r"\setlength{\mathindent}{0pt}" + "\n")
            f.write(r"\sloppy" + "\n")
            f.write(r"\begin{document}" + "\n\n")
            f.write(r"\section*{" + filename.replace("_", " ").title() + "}" + "\n\n")
            f.write(antwort_latex + "\n\n")
            f.write(r"\end{document}")

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
