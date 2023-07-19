from doctr.io import DocumentFile, Document
from doctr.models import ocr_predictor

doc = DocumentFile.from_images("input/1.png")
model = ocr_predictor(pretrained=True)
result: Document = model(doc)
result.show(doc)
