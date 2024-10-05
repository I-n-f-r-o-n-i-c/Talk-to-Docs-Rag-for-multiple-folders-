import fitz  # PyMuPDF for text and image extraction
import pdfplumber  # For table extraction
from PIL import Image, UnidentifiedImageError
import io
import os
import csv
import json

class PDFExtractor:
    def __init__(self, pdf_path, pdf_output_folder):
        self.pdf_path = pdf_path
        self.pdf_output_folder = pdf_output_folder
        self.document = fitz.open(pdf_path)

    def extract_text(self):
        text_data = []
        for page_num in range(self.document.page_count):
            page = self.document.load_page(page_num)
            text = page.get_text("text")
            text_data.append({"page_number": page_num + 1, "text": text})
        text_output_path = os.path.join(self.pdf_output_folder, "text_data.json")
        with open(text_output_path, "w") as f:
            json.dump(text_data, f, indent=4)
        for page in text_data:
            page_num = page["page_number"]
            page_text = page["text"]
            text_file_path = os.path.join(self.pdf_output_folder, f"text_page_{page_num}.txt")
            with open(text_file_path, "w", encoding='utf-8') as text_file:
                text_file.write(page_text)
        return text_data

    def extract_tables(self):
        tables_data = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_index, table in enumerate(tables):
                    tables_data.append({"page_number": page_num + 1, "table": table})
                    table_file_path = os.path.join(self.pdf_output_folder, f"table_page_{page_num + 1}_{table_index + 1}.csv")
                    with open(table_file_path, "w", newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerows(table)
        tables_output_path = os.path.join(self.pdf_output_folder, "tables_data.json")
        with open(tables_output_path, "w") as f:
            json.dump(tables_data, f, indent=4)
        return tables_data

    def extract_images(self):
        image_data = []
        for page_num in range(self.document.page_count):
            page = self.document.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = self.document.extract_image(xref)
                if "image" in base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        if not image_ext:
                            image_ext = "png"
                        image_path = os.path.join(self.pdf_output_folder, f"image_page_{page_num + 1}_{img_index + 1}.{image_ext}")
                        image.save(image_path)
                        image_data.append({"page_number": page_num + 1, "image_path": image_path})
                    except UnidentifiedImageError:
                        print(f"Skipping non-image data found on page {page_num + 1}")
                else:
                    print(f"No valid image data found for page {page_num + 1}, image index {img_index + 1}")
        images_output_path = os.path.join(self.pdf_output_folder, "images_data.json")
        with open(images_output_path, "w") as f:
            json.dump(image_data, f, indent=4)
        return image_data

    def extract_all_data(self):
        data = {"text": self.extract_text(), "tables": self.extract_tables(), "images": self.extract_images()}
        all_data_output_path = os.path.join(self.pdf_output_folder, "all_data.json")
        with open(all_data_output_path, "w") as f:
            json.dump(data, f, indent=4)
        return data