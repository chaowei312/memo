"""PDF text extraction and processing."""

import os
import re
from typing import List, Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm

import PyPDF2
import pdfplumber


class PDFProcessor:
    """Extract and process text from PDF documents."""
    
    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 1000,
        clean_text: bool = True,
        use_pdfplumber: bool = True
    ):
        """Initialize PDF processor.
        
        Args:
            min_length: Minimum text chunk length
            max_length: Maximum text chunk length
            clean_text: Whether to clean extracted text
            use_pdfplumber: Use pdfplumber (better quality) vs PyPDF2 (faster)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.clean_text = clean_text
        self.use_pdfplumber = use_pdfplumber
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        
        if self.use_pdfplumber:
            # Use pdfplumber for better extraction quality
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                print(f"Error with pdfplumber for {pdf_path}: {e}")
                # Fallback to PyPDF2
                text = self._extract_with_pypdf2(pdf_path)
        else:
            text = self._extract_with_pypdf2(pdf_path)
        
        if self.clean_text:
            text = self._clean_text(text)
        
        return text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"\(\)\[\]]', '', text)
        
        # Fix common OCR errors
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', ' ', text)
        text = re.sub(r'Page \d+', '', text)
        
        # Remove headers/footers (heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip very short lines (likely headers/footers)
            if len(line.strip()) > 10:
                cleaned_lines.append(line)
        
        text = ' '.join(cleaned_lines)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of appropriate size.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            # Check if adding sentence exceeds max length
            if len(current_chunk) + len(sentence) > self.max_length:
                if len(current_chunk) >= self.min_length:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        # Add remaining chunk
        if len(current_chunk) >= self.min_length:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with metadata and text chunks
        """
        # Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Chunk text
        chunks = self.chunk_text(full_text)
        
        # Get metadata
        metadata = {
            'filename': os.path.basename(pdf_path),
            'path': pdf_path,
            'num_chunks': len(chunks),
            'total_chars': len(full_text)
        }
        
        return {
            'metadata': metadata,
            'chunks': chunks,
            'full_text': full_text
        }
    
    def process_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        save_format: str = 'json'
    ) -> List[Dict]:
        """Process all PDFs in a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            output_dir: Directory to save processed data
            save_format: Format to save ('json' or 'txt')
            
        Returns:
            List of processed documents
        """
        folder_path = Path(folder_path)
        pdf_files = list(folder_path.glob('*.pdf'))
        
        print(f"Found {len(pdf_files)} PDF files")
        
        processed_docs = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                doc_data = self.process_pdf(str(pdf_path))
                processed_docs.append(doc_data)
                
                # Save individual file if output_dir specified
                if output_dir:
                    self._save_document(doc_data, output_dir, save_format)
                    
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
        
        # Save combined data
        if output_dir:
            self._save_all_documents(processed_docs, output_dir, save_format)
        
        return processed_docs
    
    def _save_document(
        self,
        doc_data: Dict,
        output_dir: str,
        save_format: str
    ):
        """Save individual document data.
        
        Args:
            doc_data: Document data dictionary
            output_dir: Output directory
            save_format: Save format ('json' or 'txt')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(doc_data['metadata']['filename']).stem
        
        if save_format == 'json':
            output_path = output_dir / f"{base_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        elif save_format == 'txt':
            # Save chunks as separate text files
            chunks_dir = output_dir / base_name
            chunks_dir.mkdir(exist_ok=True)
            
            for i, chunk in enumerate(doc_data['chunks']):
                chunk_path = chunks_dir / f"chunk_{i:04d}.txt"
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)
    
    def _save_all_documents(
        self,
        docs: List[Dict],
        output_dir: str,
        save_format: str
    ):
        """Save all documents to a single file.
        
        Args:
            docs: List of document data
            output_dir: Output directory
            save_format: Save format
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_format == 'json':
            output_path = output_dir / "all_documents.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(docs, f, indent=2, ensure_ascii=False)
        
        elif save_format == 'txt':
            output_path = output_dir / "all_text.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in docs:
                    f.write(f"=== {doc['metadata']['filename']} ===\n\n")
                    for chunk in doc['chunks']:
                        f.write(chunk + "\n\n")
        
        print(f"Saved combined data to {output_path}")


def main():
    """Example usage."""
    processor = PDFProcessor(
        min_length=100,
        max_length=512,
        clean_text=True
    )
    
    # Process single PDF
    # doc_data = processor.process_pdf("path/to/document.pdf")
    
    # Process folder of PDFs
    # docs = processor.process_folder("path/to/pdfs", output_dir="data/processed")


if __name__ == "__main__":
    main()
