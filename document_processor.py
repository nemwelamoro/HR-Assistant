# -*- coding: utf-8 -*-
import tiktoken
import re
from typing import List, Dict
import PyPDF2
from docx import Document
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file with better quality"""
    try:
        import fitz  # PyMuPDF - much better than PyPDF2
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except ImportError:
        # Fallback to PyPDF2 if PyMuPDF not available
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean text"""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 150) -> List[Dict]:  # Increased overlap
    """Enhanced chunking with better context preservation and smaller chunks"""
    if not text:
        return []
    
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        
        # Create smaller, more focused chunks for better search
        # Split by sections first (look for headers/titles)
        sections = re.split(r'\n\s*(?=[A-Z][A-Za-z\s]{10,}:|\d+\.\s+[A-Z])', text)
        
        chunks = []
        
        for section in sections:
            if not section.strip():
                continue
            
            # Further split long sections by paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            
            current_chunk = ""
            current_tokens = 0
            
            for paragraph in paragraphs:
                paragraph_tokens = len(encoding.encode(paragraph))
                
                # If paragraph itself is too long, split by sentences
                if paragraph_tokens > chunk_size * 0.8:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    for sentence in sentences:
                        sentence_tokens = len(encoding.encode(sentence))
                        
                        if current_tokens + sentence_tokens > chunk_size and current_chunk:
                            # Save current chunk
                            chunks.append({
                                "content": current_chunk.strip(),
                                "token_count": current_tokens,
                                "chunk_index": len(chunks)
                            })
                            
                            # Start new chunk with overlap
                            overlap_sentences = current_chunk.split('. ')[-2:] if overlap > 0 else []
                            overlap_text = '. '.join(overlap_sentences)
                            
                            if len(encoding.encode(overlap_text)) <= overlap:
                                current_chunk = overlap_text + ". " + sentence if overlap_text else sentence
                            else:
                                current_chunk = sentence
                            
                            current_tokens = len(encoding.encode(current_chunk))
                        else:
                            current_chunk = (current_chunk + " " + sentence).strip()
                            current_tokens = len(encoding.encode(current_chunk))
                else:
                    # Regular paragraph processing
                    if current_tokens + paragraph_tokens > chunk_size and current_chunk:
                        # Save current chunk
                        chunks.append({
                            "content": current_chunk.strip(),
                            "token_count": current_tokens,
                            "chunk_index": len(chunks)
                        })
                        
                        # Start new chunk with overlap
                        if overlap > 0:
                            overlap_words = current_chunk.split()[-overlap//4:]  # Approximate word overlap
                            overlap_text = ' '.join(overlap_words)
                            if len(encoding.encode(overlap_text)) <= overlap:
                                current_chunk = overlap_text + " " + paragraph
                            else:
                                current_chunk = paragraph
                        else:
                            current_chunk = paragraph
                        
                        current_tokens = len(encoding.encode(current_chunk))
                    else:
                        current_chunk = (current_chunk + "\n\n" + paragraph).strip()
                        current_tokens = len(encoding.encode(current_chunk))
            
            # Add final chunk from this section
            if current_chunk and len(current_chunk.strip()) > 50:
                chunks.append({
                    "content": current_chunk.strip(),
                    "token_count": current_tokens,
                    "chunk_index": len(chunks)
                })
        
        # Filter out very short chunks but be less aggressive
        chunks = [chunk for chunk in chunks if len(chunk["content"].split()) >= 15]  # Reduced from 10
        
        logger.info(f"Created {len(chunks)} chunks with average size {sum(c['token_count'] for c in chunks) / len(chunks):.0f} tokens")
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return []

def process_document(filename: str, file_content: bytes) -> Dict:
    """Enhanced document processing with better metadata"""
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    text_content = ""
    
    # Determine file type and extract text
    try:
        if file_extension == 'pdf':
            # Save bytes to temporary file and extract
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                text_content = extract_text_from_pdf(temp_path)
            finally:
                os.unlink(temp_path)
                
        elif file_extension in ['docx', 'doc']:
            # Save bytes to temporary file and extract
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                text_content = extract_text_from_docx(temp_path)
            finally:
                os.unlink(temp_path)
                
        else:
            # Try to decode as text file
            try:
                text_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = file_content.decode('latin-1')
                except UnicodeDecodeError:
                    logger.error(f"Unable to decode file: {filename}")
                    text_content = ""
        
        if text_content:
            text_content = clean_text(text_content)
            
            # Enhanced title generation
            title = filename.replace('_', ' ').replace('-', ' ')
            if '.' in title:
                title = title.rsplit('.', 1)[0]
            title = ' '.join(word.capitalize() for word in title.split())
            
            # Calculate word count and reading time
            word_count = len(text_content.split()) if text_content else 0
            reading_time = max(1, word_count // 200)  # Assume 200 words per minute
            
            # Enhanced tagging system
            tags = []
            
            # File type tag
            if file_extension:
                tags.append(f"filetype-{file_extension}")
            
            # Content analysis for better tags
            content_lower = text_content.lower()
            
            # HR-specific enhanced keywords
            hr_categories = {
                'policy': {
                    'keywords': ['policy', 'policies', 'procedure', 'guidelines', 'rules', 'regulations'],
                    'weight': 2
                },
                'benefits': {
                    'keywords': ['benefits', 'insurance', 'health', 'dental', 'vision', 'retirement', '401k', 'pension'],
                    'weight': 3
                },
                'leave': {
                    'keywords': ['leave', 'vacation', 'pto', 'sick', 'maternity', 'paternity', 'time off', 'absence'],
                    'weight': 3
                },
                'training': {
                    'keywords': ['training', 'development', 'learning', 'course', 'education', 'skill'],
                    'weight': 2
                },
                'recruitment': {
                    'keywords': ['recruitment', 'hiring', 'interview', 'candidate', 'job', 'position', 'application'],
                    'weight': 2
                },
                'performance': {
                    'keywords': ['performance', 'review', 'evaluation', 'appraisal', 'assessment', 'feedback'],
                    'weight': 2
                },
                'compliance': {
                    'keywords': ['compliance', 'legal', 'regulation', 'law', 'audit', 'requirement'],
                    'weight': 3
                },
                'handbook': {
                    'keywords': ['handbook', 'manual', 'guide', 'reference'],
                    'weight': 2
                }
            }
            
            # Score-based tagging
            for category, info in hr_categories.items():
                matches = sum(info['weight'] for keyword in info['keywords'] if keyword in content_lower)
                if matches >= info['weight']:
                    tags.append(category)
            
            # Document type inference
            if word_count > 1000:
                tags.append('comprehensive')
            elif word_count > 500:
                tags.append('detailed')
            else:
                tags.append('brief')
            
            # Add priority tags for important documents
            priority_indicators = ['policy', 'handbook', 'manual', 'guide', 'procedure']
            if any(indicator in filename.lower() for indicator in priority_indicators):
                tags.append('priority')
            
            return {
                'title': title,
                'content': text_content,
                'word_count': word_count,
                'reading_time': reading_time,
                'file_type': file_extension,
                'tags': tags,
                'quality_score': min(100, max(0, (word_count // 10) + len(tags) * 5))
            }
        else:
            # Return empty document structure if no content
            return {
                'title': filename,
                'content': "",
                'word_count': 0,
                'reading_time': 0,
                'file_type': file_extension,
                'tags': [],
                'quality_score': 0
            }
    
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        # Always return a dictionary, never None
        return {
            'title': filename,
            'content': "",
            'word_count': 0,
            'reading_time': 0,
            'file_type': file_extension,
            'tags': [],
            'quality_score': 0
        }

def should_skip_file(filename: str, content: str) -> bool:
    """Enhanced content filtering"""
    if not content or not content.strip():
        return True
    
    # Skip if content is too short
    word_count = len(content.split())
    if word_count < 30:  # Reduced from 50 to be less aggressive
        return True
    
    # Skip letterhead and template files
    filename_lower = filename.lower()
    skip_patterns = [
        'letterhead', 'letter head', 'header', 
        'placeholder', 'empty', 'logo', 'signature'
    ]
    
    # Don't skip all templates - only letterheads and logos
    if any(pattern in filename_lower for pattern in skip_patterns):
        return True
    
    content_lower = content.lower()
    
    # Skip files that are mostly contact info or addresses
    contact_indicators = [
        'email:', 'phone:', 'tel:', 'fax:', 'address:',
        'p.o. box', 'postal code', 'zip code'
    ]
    
    contact_matches = sum(1 for indicator in contact_indicators if indicator in content_lower)
    if contact_matches > 3 and word_count < 200:  # More lenient threshold
        return True
    
    # Skip if content lacks meaningful sentences
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 5]  # Reduced from 10
    if len(sentences) < 2:  # Reduced from 3
        return True
    
    # Skip if content is mostly formatting or repeated words (more lenient)
    unique_words = set(content.lower().split())
    if len(unique_words) < word_count * 0.2:  # Reduced from 0.3 to be more lenient
        return True
    
    return False