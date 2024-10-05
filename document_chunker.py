import os
import json

class DocumentChunker:
    def __init__(self, extracted_data_folder, pdf_name):
        self.extracted_data_folder = extracted_data_folder
        self.pdf_name = pdf_name

    def chunk_text(self, max_chunk_size=1500, overlap_ratio=0.25):
        chunked_text = []
        for filename in os.listdir(self.extracted_data_folder):
            if filename.startswith("text_page_") and filename.endswith(".txt"):
                page_num_str = filename.split("_")[2]
                page_num = int(page_num_str.split(".")[0])
                with open(os.path.join(self.extracted_data_folder, filename), "r", encoding='utf-8') as f:
                    text = f.read().strip()
                    overlap_size = int(max_chunk_size * overlap_ratio)
                    start_idx = 0
                    while start_idx < len(text):
                        end_idx = min(start_idx + max_chunk_size, len(text))
                        chunk = text[start_idx:end_idx]
                        chunked_text.append({"page_number": page_num, "chunk_index": len(chunked_text) + 1, "text_chunk": chunk})
                        start_idx += max_chunk_size - overlap_size
        return chunked_text

    def chunk_all_text(self, output_folder, max_chunk_size=1500, overlap_ratio=0.25):
        chunked_text = self.chunk_text(max_chunk_size, overlap_ratio)
        for chunk in chunked_text:
            page_num = chunk["page_number"]
            chunk_index = chunk["chunk_index"]
            text_chunk = chunk["text_chunk"]
            text_file_path = os.path.join(output_folder, f"{self.pdf_name}_page_{page_num}_chunk_{chunk_index}.txt")
            with open(text_file_path, "w", encoding='utf-8') as f:
                f.write(text_chunk)
        return chunked_text
