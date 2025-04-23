import os
import pickle
import re
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# Setup
load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
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

    for line in lines:
        line = line.strip()
        line = line.replace("&", r"\&")

        # Markdown-Titel wie "1. **Faktorregel:**" erkennen
        title_match = re.match(r"^\d+\.\s+\*\*(.*?)\*\*", line)
        if title_match:
            if in_align:
                latex_lines.append(r"\end{align*}")
                in_align = False
            if in_itemize:
                latex_lines.append(r"\end{itemize}")
                in_itemize = False
            latex_lines.append(r"\textbf{" + title_match.group(1).strip() + r"}\\")
            continue

        # Überschriften wie "# Abschnittstitel"
        if re.match(r"^#{1,6}\s", line):
            if in_align:
                latex_lines.append(r"\end{align*}")
                in_align = False
            if in_itemize:
                latex_lines.append(r"\end{itemize}")
                in_itemize = False
            content = line.lstrip('#').strip()
            latex_lines.append(r"\textbf{" + content + r"}\\")
            continue

        # Bullet Points
        if line.startswith("- "):
            if in_align:
                latex_lines.append(r"\end{align*}")
                in_align = False
            if not in_itemize:
                latex_lines.append(r"\begin{itemize}")
                in_itemize = True
            latex_lines.append(r"\item " + line[2:])
            continue

        # Zeilen mit Gleichungen oder math. Symbolen
        if line.count("=") >= 1 or line.count("+") >= 3 or any(char in line for char in ["\\", "∫", "Σ", "π", "\u03b1"]):
            if in_itemize:
                latex_lines.append(r"\end{itemize}")
                in_itemize = False
            if not in_align:
                latex_lines.append(r"\begin{align*}")
                in_align = True
            segments = re.split(r"\s{2,}", line)
            for segment in segments:
                latex_lines.append(segment.strip() + r"\\")
            continue

        # Normale Textzeilen
        if in_align:
            latex_lines.append(r"\end{align*}")
            in_align = False
        if in_itemize:
            latex_lines.append(r"\end{itemize}")
            in_itemize = False

        latex_lines.append(line)

    # Offene Umgebungen am Ende schließen
    if in_align:
        latex_lines.append(r"\end{align*}")
    if in_itemize:
        latex_lines.append(r"\end{itemize}")

    return "\n".join(latex_lines)



def ask_gpt(context, user_question):
    system_prompt = (
        "You are a helpful math assistant. "
        "You will extract and list all mathematical formulas from the provided context. "
        "List them in LaTeX, and include any associated titles or explanations if available. "
        "Focus on completeness – do not skip any recognizable mathematical expressions."
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
    match = re.search(r"Seite[n]*\s*(\d+)\s*[-–]\s*(\d+)", frage, re.IGNORECASE)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return range(start, end + 1)
    else:
        return None


def main():
    fragen = [
        "Extrahiere alle Formeln inklusive ihrer Titel aus Seite 51-55",
        "Extrahiere alle Formeln inklusive ihrer Titel aus Seite 56-60",
        "Extrahiere alle Formeln inklusive ihrer Titel aus Seite 61-65",
        "Extrahiere alle Formeln inklusive ihrer Titel aus Seite 66-70"
    ]

    with open(os.path.join(EMBED_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    texts = metadata["texts"]
    pages = metadata.get("pages", list(range(1, len(texts) + 1)))

    for frage in fragen:
        print(f"\nFrage: {frage}")
        keyword = get_main_keyword(frage)
        seitenbereich = extract_page_range(frage)
        seiten_suffix = f"_s{seitenbereich.start}_s{seitenbereich.stop - 1}" if seitenbereich else ""
        filename = f"{keyword}{seiten_suffix}"

        if not seitenbereich:
            print("Konnte Seitenbereich nicht erkennen.")
            continue

        chunks = [text for text, page in zip(texts, pages) if page in seitenbereich]
        if not chunks:
            print("Keine passenden Chunks im gewünschten Seitenbereich gefunden.")
            continue

        context_text = "\n\n---\n\n".join(chunks)
        antwort_roh = ask_gpt(context_text, frage)
        antwort_latex = markdown_to_latex(antwort_roh)

        tex_path = os.path.join(OUTPUT_DIR, f"{filename}.tex")
        pdf_path = os.path.join(OUTPUT_DIR, f"{filename}.pdf")

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(r"\documentclass{article}" + "\n")
            f.write(r"\usepackage[utf8]{inputenc}" + "\n")
            f.write(r"\usepackage{amsmath,amssymb,amsthm}" + "\n")
            f.write(r"\usepackage[a4paper,margin=2.5cm]{geometry}" + "\n")
            f.write(r"\begin{document}" + "\n\n")
            f.write(r"\section*{" + filename.replace("_", " ").title() + "}\n\n")
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