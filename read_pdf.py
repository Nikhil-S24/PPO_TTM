from pypdf import PdfReader

reader = PdfReader("Research_Paper.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n\n"

with open("paper.txt", "w", encoding="utf-8") as f:
    f.write(text)
