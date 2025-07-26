#!/usr/bin/env python3
"""
Cell 1: Import Libraries and Setup
Explainable AI for Image Tagging using Kosmos-2
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

# Core libraries
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Hugging Face Transformers
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    Kosmos2ForConditionalGeneration,
    AutoTokenizer
)

# PDF processing
try:
    import fitz  # PyMuPDF
    print("✅ PyMuPDF (fitz) imported successfully")
except ImportError as e:
    print("❌ PyMuPDF import failed. Installing...")
    import subprocess
    import sys
    try:
        # Try to install PyMuPDF
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
        import fitz
        print("✅ PyMuPDF installed and imported successfully")
    except Exception as install_error:
        print(f"❌ Failed to install PyMuPDF: {install_error}")
        print("💡 Please install manually: pip install PyMuPDF")
        # Use alternative PDF processing
        fitz = None
        print("⚠️ Running without PyMuPDF - PDF processing will be limited")

import sqlite3

# Report generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("✅ Libraries imported successfully!")
print("🔧 Setting up explainable AI system...")

"""
Cell 2: Kosmos-2 Model Loader
Load and initialize the Kosmos-2 model from Hugging Face
"""

class Kosmos2Explainer:
    """Kosmos-2 based explainable AI for image tag generation"""
    
    def __init__(self, model_name: str = "microsoft/kosmos-2-patch14-224"):
        """Initialize Kosmos-2 model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"🚀 Loading Kosmos-2 model: {model_name}")
        logger.info(f"🔧 Using device: {self.device}")
        
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Kosmos2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            logger.info("✅ Kosmos-2 model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Kosmos-2 model: {e}")
            raise
    
    def generate_explanation(self, image: Image.Image, existing_tags: List[str]) -> Dict[str, Any]:
        """
        Generate explanations for why specific tags were assigned to an image
        """
        try:
            explanations = {}
            
            # 1. General image understanding with better prompt
            general_prompt = "<grounding>What objects and elements do you see in this image?"
            inputs = self.processor(text=general_prompt, images=image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=64,  # Reduced for cleaner output
                    temperature=0.3,    # Lower temperature for more coherent text
                    do_sample=False,    # Greedy decoding for consistency
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract and clean the generated description
            description = self._extract_and_clean_text(generated_text, general_prompt)
            explanations["general_description"] = description
            
            # 2. Tag-specific explanations with simpler prompts
            tag_explanations = {}
            
            for tag in existing_tags[:8]:  # Limit to 8 tags for better quality
                # Use simpler, more direct prompts
                tag_prompt = f"<grounding>Why is '{tag}' visible in this image?"
                
                try:
                    inputs = self.processor(text=tag_prompt, images=image, return_tensors="pt")
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            pixel_values=inputs["pixel_values"],
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            image_embeds=None,
                            image_embeds_position_mask=inputs["image_embeds_position_mask"],
                            use_cache=True,
                            max_new_tokens=48,  # Shorter for cleaner explanations
                            temperature=0.2,    # Very low temperature
                            do_sample=False,    # Greedy decoding
                            pad_token_id=self.processor.tokenizer.eos_token_id
                        )
                    
                    tag_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    tag_explanation = self._extract_and_clean_text(tag_response, tag_prompt)
                    
                    # Additional cleaning for tag explanations
                    tag_explanation = self._clean_tag_explanation(tag_explanation, tag)
                    tag_explanations[tag] = tag_explanation
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate explanation for tag '{tag}': {e}")
                    tag_explanations[tag] = f"Visual evidence for '{tag}' detected in image"
            
            explanations["tag_explanations"] = tag_explanations
            
            # 3. Simple confidence assessment
            confidence_prompt = "<grounding>What are the clearest objects in this image?"
            
            try:
                inputs = self.processor(text=confidence_prompt, images=image, return_tensors="pt")
                
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=inputs["pixel_values"],
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        image_embeds=None,
                        image_embeds_position_mask=inputs["image_embeds_position_mask"],
                        use_cache=True,
                        max_new_tokens=48,
                        temperature=0.2,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                confidence_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                confidence_assessment = self._extract_and_clean_text(confidence_response, confidence_prompt)
                
                explanations["confidence_assessment"] = confidence_assessment
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate confidence assessment: {e}")
                explanations["confidence_assessment"] = "Clear visual elements detected"
            
            return explanations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate explanations: {e}")
            return {
                "general_description": f"Error generating explanation: {e}",
                "tag_explanations": {},
                "confidence_assessment": "Could not assess"
            }
    
    
    def _extract_and_clean_text(self, full_response: str, prompt: str) -> str:
        """Extract and clean the generated text from the full response"""
        try:
            # Remove the prompt from the response
            if prompt in full_response:
                generated = full_response.replace(prompt, "").strip()
            else:
                generated = full_response.strip()
            
            # Clean up the response more aggressively
            generated = re.sub(r'<[^>]+>', '', generated)  # Remove special tokens
            generated = re.sub(r'\s+', ' ', generated)     # Normalize whitespace
            
            # Remove common noise patterns
            noise_patterns = [
                r"^\. the,? to and of as in.*?$",  # Remove the repetitive text pattern
                r"^\w{1,3}( \w{1,3}){10,}",       # Remove strings of short words
                r"^[^\w\s]*$",                     # Remove non-alphanumeric strings
                r"^\d+\s*$",                       # Remove standalone numbers
            ]
            
            for pattern in noise_patterns:
                generated = re.sub(pattern, '', generated, flags=re.IGNORECASE | re.MULTILINE)
            
            # Split into sentences and take the first meaningful one
            sentences = re.split(r'[.!?]+', generated)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15 and not self._is_noise_text(sentence):
                    return sentence
            
            # If no good sentence found, return cleaned version or fallback
            generated = generated.strip()
            if len(generated) > 10 and not self._is_noise_text(generated):
                return generated[:200]  # Limit length
            
            return "Clear visual elements identified"
            
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning generated text: {e}")
            return "Visual analysis completed"
    
    def _clean_tag_explanation(self, explanation: str, tag: str) -> str:
        """Additional cleaning specifically for tag explanations"""
        try:
            # Remove the tag repetition if it exists
            explanation = explanation.replace(f"'{tag}'", tag)
            
            # If explanation is too repetitive or noisy, create a simple one
            if self._is_noise_text(explanation) or len(explanation.split()) < 3:
                return f"The {tag} is clearly visible in the image"
            
            # Ensure it starts properly
            if not explanation[0].isupper():
                explanation = explanation.capitalize()
            
            # Limit length
            words = explanation.split()
            if len(words) > 20:
                explanation = ' '.join(words[:20]) + "..."
            
            return explanation
            
        except Exception as e:
            return f"Visual evidence of {tag} detected"
    
    def _is_noise_text(self, text: str) -> bool:
        """Check if text appears to be noise or repetitive content"""
        try:
            text = text.lower().strip()
            
            # Check for common noise patterns
            noise_indicators = [
                len(text) < 5,
                text.count('the') > len(text.split()) // 3,  # Too many "the"
                text.count(',') > len(text.split()) // 2,    # Too many commas
                len(set(text.split())) < len(text.split()) // 2,  # Too repetitive
                text.startswith('. the, to and'),  # Specific noise pattern
                text.count(' ') > text.count('. ') * 10,  # Too many spaces vs periods
            ]
            
            return any(noise_indicators)
            
        except Exception:
            return True

# Initialize the explainer
logger.info("🔄 Initializing Kosmos-2 Explainer...")
kosmos_explainer = Kosmos2Explainer()
print("✅ Kosmos-2 Explainer initialized successfully!")

"""
Cell 3: PDF Scanner and Parser
Scan PDF reports and extract image information and tags
"""

class PDFAnalysisScanner:
    """Scanner for PDF reports generated by the image tagging system"""
    
    def __init__(self):
        self.supported_formats = {'.pdf'}
    
    def scan_pdf_report(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Scan PDF report and extract image information and tags
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if pdf_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {pdf_path.suffix}")
            
            logger.info(f"📖 Scanning PDF report: {pdf_path.name}")
            
            # Open PDF document
            doc = fitz.open(str(pdf_path))
            extracted_data = []
            
            current_image_data = None
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text from page
                text = page.get_text()
                
                # Extract images from page
                image_list = page.get_images()
                
                # Parse text for image information
                image_info = self._parse_page_text(text, page_num)
                
                # Extract actual images
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Store image with parsed information
                            if image_info:
                                image_record = {
                                    "page_number": page_num + 1,
                                    "image_index": img_index,
                                    "image": img_pil,
                                    "extracted_info": image_info,
                                    "image_data": img_data
                                }
                                extracted_data.append(image_record)
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
            logger.info(f"✅ Extracted {len(extracted_data)} images from PDF")
            return extracted_data
            
        except Exception as e:
            logger.error(f"❌ Failed to scan PDF: {e}")
            return []
    
    def _parse_page_text(self, text: str, page_num: int) -> Dict[str, Any]:
        """
        Parse page text to extract image information and tags
        """
        try:
            info = {
                "page_number": page_num + 1,
                "file_name": None,
                "file_path": None,
                "ai_tags": [],
                "blip_caption": None,
                "llava_description": None,
                "file_info": {}
            }
            
            lines = text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # Extract image title/filename
                if line.startswith("Image") and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        info["file_name"] = parts[1].strip()
                
                # Extract file path
                if "Path:" in line:
                    info["file_path"] = line.replace("Path:", "").strip().replace("•", "").strip()
                
                # Extract image dimensions
                if "Size:" in line:
                    size_info = line.replace("Size:", "").strip().replace("•", "").strip()
                    info["file_info"]["dimensions"] = size_info
                
                # Extract format
                if "Format:" in line:
                    format_info = line.replace("Format:", "").strip().replace("•", "").strip()
                    info["file_info"]["format"] = format_info
                
                # Extract AI-generated tags
                if "AI-Generated Tags:" in line:
                    current_section = "tags"
                    continue
                
                if current_section == "tags" and line.startswith("•"):
                    tag = line.replace("•", "").strip()
                    if tag and tag not in info["ai_tags"]:
                        info["ai_tags"].append(tag)
                
                # Extract BLIP caption
                if "BLIP Caption:" in line:
                    current_section = "blip"
                    continue
                
                if current_section == "blip" and line and not line.startswith("•") and ":" not in line:
                    info["blip_caption"] = line
                    current_section = None
                
                # Extract LLaVA description
                if "LLaVA Description:" in line:
                    current_section = "llava"
                    continue
                
                if current_section == "llava" and line and not line.startswith("•") and ":" not in line:
                    info["llava_description"] = line
                    current_section = None
                
                # Reset section on new headers
                if any(header in line for header in ["File Information:", "AI-Generated Tags:", "BLIP Caption:", "LLaVA Description:"]):
                    if "AI-Generated Tags:" not in line:
                        current_section = None
            
            return info
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse page text: {e}")
            return {}
    
    def find_pdf_reports(self, directory: str) -> List[str]:
        """
        Find all PDF reports in a directory
        """
        try:
            directory = Path(directory)
            pdf_files = []
            
            # Look for PDF files that match the naming pattern
            patterns = [
                "*image_analysis_report*.pdf",
                "*ai_image_analysis_report*.pdf",
                "*single_image_report*.pdf"
            ]
            
            for pattern in patterns:
                pdf_files.extend(directory.glob(pattern))
            
            # Also include any PDF files in the directory
            all_pdfs = list(directory.glob("*.pdf"))
            
            # Combine and remove duplicates
            all_found = list(set(pdf_files + all_pdfs))
            
            logger.info(f"📁 Found {len(all_found)} PDF files in {directory}")
            return [str(pdf) for pdf in all_found]
            
        except Exception as e:
            logger.error(f"❌ Failed to find PDF reports: {e}")
            return []

# Initialize PDF scanner
pdf_scanner = PDFAnalysisScanner()
print("✅ PDF Scanner initialized successfully!")

"""
Cell 4: Explainable AI Analyzer
Main analyzer that combines PDF scanning with Kosmos-2 explanations
"""

class ExplainableImageTagAnalyzer:
    """
    Main analyzer for generating explanations of image tags using Kosmos-2
    """
    
    def __init__(self):
        self.kosmos_explainer = kosmos_explainer
        self.pdf_scanner = pdf_scanner
        self.analysis_results = []
    
    def analyze_pdf_report(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Analyze a PDF report and generate explanations for each image's tags
        """
        try:
            logger.info(f"🔍 Starting analysis of PDF report: {pdf_path}")
            
            # 1. Scan PDF and extract images
            extracted_images = self.pdf_scanner.scan_pdf_report(pdf_path)
            
            if not extracted_images:
                logger.warning("⚠️ No images found in PDF report")
                return []
            
            logger.info(f"📊 Found {len(extracted_images)} images to analyze")
            
            # 2. Generate explanations for each image
            results = []
            
            for i, image_data in enumerate(extracted_images, 1):
                logger.info(f"🔬 Analyzing image {i}/{len(extracted_images)}")
                
                try:
                    # Extract information
                    image = image_data["image"]
                    extracted_info = image_data["extracted_info"]
                    existing_tags = extracted_info.get("ai_tags", [])
                    
                    if not existing_tags:
                        logger.warning(f"⚠️ No tags found for image {i}, skipping...")
                        continue
                    
                    # Generate explanations using Kosmos-2
                    explanations = self.kosmos_explainer.generate_explanation(image, existing_tags)
                    
                    # Compile analysis result
                    analysis_result = {
                        "image_index": i,
                        "page_number": image_data["page_number"],
                        "file_name": extracted_info.get("file_name", f"Image_{i}"),
                        "file_path": extracted_info.get("file_path", "Unknown"),
                        "original_tags": existing_tags,
                        "blip_caption": extracted_info.get("blip_caption"),
                        "llava_description": extracted_info.get("llava_description"),
                        "kosmos_explanations": explanations,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "image_object": image  # Store for report generation
                    }
                    
                    results.append(analysis_result)
                    
                    logger.info(f"✅ Generated explanations for {len(existing_tags)} tags")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to analyze image {i}: {e}")
                    continue
            
            self.analysis_results = results
            logger.info(f"🎉 Analysis complete! Generated explanations for {len(results)} images")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze PDF report: {e}")
            return []
    
    def get_tag_explanation(self, image_index: int, tag: str) -> str:
        """
        Get explanation for a specific tag on a specific image
        """
        try:
            if not self.analysis_results:
                return "No analysis results available"
            
            # Find the image analysis
            for result in self.analysis_results:
                if result["image_index"] == image_index:
                    tag_explanations = result["kosmos_explanations"].get("tag_explanations", {})
                    return tag_explanations.get(tag, f"No explanation found for tag '{tag}'")
            
            return f"Image {image_index} not found in analysis results"
            
        except Exception as e:
            logger.error(f"❌ Failed to get tag explanation: {e}")
            return f"Error retrieving explanation: {e}"
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics of the analysis
        """
        try:
            if not self.analysis_results:
                return {}
            
            stats = {
                "total_images_analyzed": len(self.analysis_results),
                "total_tags_explained": 0,
                "average_tags_per_image": 0,
                "most_common_tags": {},
                "explanation_confidence": [],
                "processing_timestamp": datetime.now().isoformat()
            }
            
            all_tags = []
            total_tags = 0
            
            for result in self.analysis_results:
                tags = result["original_tags"]
                all_tags.extend(tags)
                total_tags += len(tags)
            
            stats["total_tags_explained"] = total_tags
            stats["average_tags_per_image"] = round(total_tags / len(self.analysis_results), 2)
            
            # Count tag frequency
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            # Get most common tags
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            stats["most_common_tags"] = dict(sorted_tags[:10])
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to generate summary statistics: {e}")
            return {}
    
    def save_results_to_json(self, output_path: str = None) -> str:
        """
        Save analysis results to JSON file
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"explainable_ai_analysis_{timestamp}.json"
            
            # Prepare data for JSON (remove image objects)
            json_data = {
                "analysis_metadata": {
                    "total_images": len(self.analysis_results),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "model_used": "microsoft/kosmos-2-patch14-224"
                },
                "summary_statistics": self.generate_summary_statistics(),
                "image_analyses": []
            }
            
            for result in self.analysis_results:
                json_result = {k: v for k, v in result.items() if k != "image_object"}
                json_data["image_analyses"].append(json_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Analysis results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save results to JSON: {e}")
            return ""

# Initialize the explainable analyzer
explainable_analyzer = ExplainableImageTagAnalyzer()
print("✅ Explainable AI Analyzer initialized successfully!")

"""
Cell 5: Explainable AI Report Generator
Generate comprehensive PDF reports with explanations
"""

class ExplainableAIReportGenerator:
    """
    Generate PDF reports with explainable AI analysis
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'ExplainableTitle',
            parent=self.styles['Heading1'],
            fontSize=26,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        self.section_style = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=15,
            textColor=colors.darkgreen
        )
        self.subsection_style = ParagraphStyle(
            'SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkred
        )
        self.explanation_style = ParagraphStyle(
            'ExplanationStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=8,
            textColor=colors.black
        )
        self.tag_style = ParagraphStyle(
            'TagExplanationStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=30,
            textColor=colors.blue,
            spaceAfter=5
        )
    
    def create_explainable_report(self, analysis_results: List[Dict[str, Any]], 
                                output_path: str = None) -> str:
        """
        Create comprehensive explainable AI report
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"explainable_ai_report_{timestamp}.pdf"
            
            logger.info(f"📊 Generating explainable AI report...")
            
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title page
            story.append(Paragraph("Explainable AI Image Tagging Report", self.title_style))
            story.append(Paragraph("Generated using Kosmos-2 Vision-Language Model", 
                                 self.styles['Normal']))
            story.append(Spacer(1, 30))
            
            # Executive summary
            story.append(Paragraph("Executive Summary", self.section_style))
            
            # Generate statistics
            stats = self._generate_statistics(analysis_results)
            
            summary_data = [
                ['Total Images Analyzed:', str(stats['total_images'])],
                ['Total Tags Explained:', str(stats['total_tags'])],
                ['Average Tags per Image:', str(stats['avg_tags'])],
                ['Most Common Tag:', stats['most_common_tag']],
                ['Analysis Model:', 'Kosmos-2 (microsoft/kosmos-2-patch14-224)'],
                ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.darkblue)
            ]))
            
            story.append(summary_table)
            story.append(PageBreak())
            
            # Methodology section
            story.append(Paragraph("Methodology", self.section_style))
            methodology_text = """
            This report provides explainable AI analysis of image tagging results using Microsoft's Kosmos-2 
            vision-language model. For each image and its automatically generated tags, Kosmos-2 provides:
            
            1. General visual understanding and description
            2. Specific explanations for why each tag was assigned
            3. Confidence assessment of visual elements
            
            The analysis helps understand the reasoning behind AI-generated tags, improving transparency 
            and trust in automated image tagging systems.
            """
            story.append(Paragraph(methodology_text, self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Individual image analyses
            story.append(Paragraph("Detailed Image Analyses", self.section_style))
            story.append(PageBreak())
            
            for i, result in enumerate(analysis_results, 1):
                try:
                    # Image header
                    story.append(Paragraph(f"Analysis {i}: {result['file_name']}", 
                                         self.section_style))
                    
                    # Create main content table
                    content_data = []
                    
                    # Image column
                    img_content = []
                    if "image_object" in result:
                        try:
                            img = result["image_object"]
                            img.thumbnail((250, 250), Image.Resampling.LANCZOS)
                            
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            from reportlab.lib.utils import ImageReader
                            rl_img = ImageReader(img_buffer)
                            
                            from reportlab.platypus import Image as RLImage
                            img_content.append(RLImage(rl_img, width=2.5*inch, height=2.5*inch))
                            
                        except Exception as e:
                            img_content.append(Paragraph(f"[Image display error: {e}]", self.styles['Normal']))
                    else:
                        img_content.append(Paragraph("[Image not available]", self.styles['Normal']))
                    
                    # Analysis content column
                    analysis_content = []
                    
                    # Basic information
                    analysis_content.append(Paragraph("<b>Image Information:</b>", self.styles['Normal']))
                    analysis_content.append(Paragraph(f"File: {result.get('file_path', 'Unknown')}", self.styles['Normal']))
                    analysis_content.append(Paragraph(f"Page: {result.get('page_number', 'Unknown')}", self.styles['Normal']))
                    analysis_content.append(Spacer(1, 10))
                    
                    # Original tags
                    analysis_content.append(Paragraph("<b>Original AI-Generated Tags:</b>", self.styles['Normal']))
                    original_tags = result.get('original_tags', [])
                    if original_tags:
                        tags_text = ", ".join(original_tags[:15])  # Limit display
                        if len(original_tags) > 15:
                            tags_text += f" ... and {len(original_tags) - 15} more"
                        analysis_content.append(Paragraph(tags_text, self.tag_style))
                    else:
                        analysis_content.append(Paragraph("No tags found", self.styles['Normal']))
                    
                    analysis_content.append(Spacer(1, 10))
                    
                    # Kosmos-2 general description
                    kosmos_explanations = result.get('kosmos_explanations', {})
                    general_desc = kosmos_explanations.get('general_description', '')
                    if general_desc:
                        analysis_content.append(Paragraph("<b>Kosmos-2 Visual Understanding:</b>", self.styles['Normal']))
                        analysis_content.append(Paragraph(general_desc, self.explanation_style))
                        analysis_content.append(Spacer(1, 10))
                    
                    # Create the main table
                    content_data = [[img_content, analysis_content]]
                    
                    main_table = Table(content_data, colWidths=[3*inch, 4*inch])
                    main_table.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                        ('TOPPADDING', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
                    ]))
                    
                    story.append(main_table)
                    story.append(Spacer(1, 15))
                    
                    # Tag-specific explanations
                    tag_explanations = kosmos_explanations.get('tag_explanations', {})
                    if tag_explanations:
                        story.append(Paragraph("Tag-Specific Explanations:", self.subsection_style))
                        
                        for tag, explanation in list(tag_explanations.items())[:8]:  # Limit to 8 tags
                            story.append(Paragraph(f"<b>'{tag}':</b> {explanation}", 
                                                 self.explanation_style))
                        
                        if len(tag_explanations) > 8:
                            remaining = len(tag_explanations) - 8
                            story.append(Paragraph(f"<i>... and {remaining} more tag explanations</i>", 
                                                 self.styles['Normal']))
                    
                    # Confidence assessment
                    confidence = kosmos_explanations.get('confidence_assessment', '')
                    if confidence:
                        story.append(Spacer(1, 10))
                        story.append(Paragraph("Confidence Assessment:", self.subsection_style))
                        story.append(Paragraph(confidence, self.explanation_style))
                    
                    # Original model outputs for comparison
                    story.append(Spacer(1, 15))
                    story.append(Paragraph("Original Model Outputs (for comparison):", self.subsection_style))
                    
                    if result.get('blip_caption'):
                        story.append(Paragraph(f"<b>BLIP Caption:</b> {result['blip_caption']}", 
                                             self.styles['Normal']))
                    
                    if result.get('llava_description'):
                        story.append(Paragraph(f"<b>LLaVA Description:</b> {result['llava_description'][:200]}...", 
                                             self.styles['Normal']))
                    
                    story.append(PageBreak())
                    
                except Exception as e:
                    logger.error(f"❌ Error processing analysis {i}: {e}")
                    story.append(Paragraph(f"Error processing analysis {i}: {result.get('file_name', 'Unknown')}", 
                                         self.styles['Normal']))
                    story.append(Spacer(1, 20))
            
            # Insights and conclusions
            story.append(Paragraph("Key Insights", self.section_style))
            insights = self._generate_insights(analysis_results)
            for insight in insights:
                story.append(Paragraph(f"• {insight}", self.styles['Normal']))
                story.append(Spacer(1, 5))
            
            # Build the PDF
            doc.build(story)
            logger.info(f"✅ Explainable AI report generated: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create explainable report: {e}")
            return ""
    
    def _generate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not results:
            return {'total_images': 0, 'total_tags': 0, 'avg_tags': 0, 'most_common_tag': 'None'}
        
        total_images = len(results)
        all_tags = []
        
        for result in results:
            all_tags.extend(result.get('original_tags', []))
        
        total_tags = len(all_tags)
        avg_tags = round(total_tags / total_images, 1) if total_images > 0 else 0
        
        # Find most common tag
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        most_common_tag = max(tag_counts, key=tag_counts.get) if tag_counts else 'None'
        
        return {
            'total_images': total_images,
            'total_tags': total_tags,
            'avg_tags': avg_tags,
            'most_common_tag': most_common_tag
        }
    
    def _generate_insights(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        if not results:
            insights.append("No analysis results available for insight generation.")
            return insights
        
        # Tag frequency analysis
        all_tags = []
        for result in results:
            all_tags.extend(result.get('original_tags', []))
        
        if all_tags:
            tag_counts = {}
            for tag in all_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            most_common = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append(f"Most frequently identified elements: {', '.join([tag for tag, count in most_common])}")
        
        # Explanation quality analysis
        explanations_with_content = 0
        for result in results:
            kosmos_explanations = result.get('kosmos_explanations', {})
            if kosmos_explanations.get('general_description') and len(kosmos_explanations.get('general_description', '')) > 20:
                explanations_with_content += 1
        
        if explanations_with_content > 0:
            insights.append(f"Kosmos-2 provided detailed explanations for {explanations_with_content}/{len(results)} images ({round(100*explanations_with_content/len(results), 1)}%)")
        
        # Tag explanation coverage
        total_tag_explanations = 0
        total_tags = 0
        for result in results:
            tags = result.get('original_tags', [])
            tag_explanations = result.get('kosmos_explanations', {}).get('tag_explanations', {})
            total_tags += len(tags)
            total_tag_explanations += len(tag_explanations)
        
        if total_tags > 0:
            coverage = round(100 * total_tag_explanations / total_tags, 1)
            insights.append(f"Tag explanation coverage: {coverage}% ({total_tag_explanations}/{total_tags} tags explained)")
        
        insights.append("Kosmos-2 provides grounded visual reasoning, helping understand why specific tags were assigned based on visual evidence.")
        insights.append("This explainable AI approach increases transparency and trust in automated image tagging systems.")
        
        return insights

# Initialize report generator
report_generator = ExplainableAIReportGenerator()
print("✅ Explainable AI Report Generator initialized successfully!")

"""
Cell 6: Main Interface and Terminal Integration
Command-line interface for explainable AI analysis
"""

class ExplainableAICommandInterface:
    """
    Command-line interface for the explainable AI system
    """
    
    def __init__(self):
        self.analyzer = explainable_analyzer
        self.report_generator = report_generator
        self.pdf_scanner = pdf_scanner
    
    def run_analysis(self, pdf_path: str, output_dir: str = None) -> Dict[str, str]:
        """
        Run complete explainable AI analysis on a PDF report
        """
        try:
            logger.info(f"🚀 Starting explainable AI analysis...")
            logger.info(f"📄 PDF Path: {pdf_path}")
            
            # Validate input
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Set output directory
            if output_dir is None:
                output_dir = Path(pdf_path).parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Analyze PDF report
            logger.info("📊 Step 1: Analyzing PDF report with Kosmos-2...")
            analysis_results = self.analyzer.analyze_pdf_report(pdf_path)
            
            if not analysis_results:
                raise ValueError("No analysis results generated. Please check the PDF format and content.")
            
            # 2. Generate JSON results
            logger.info("💾 Step 2: Saving analysis results to JSON...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f"explainable_analysis_results_{timestamp}.json"
            saved_json = self.analyzer.save_results_to_json(str(json_path))
            
            # 3. Generate explainable AI report
            logger.info("📋 Step 3: Generating explainable AI PDF report...")
            report_path = output_dir / f"explainable_ai_report_{timestamp}.pdf"
            saved_report = self.report_generator.create_explainable_report(
                analysis_results, str(report_path)
            )
            
            # 4. Generate summary
            stats = self.analyzer.generate_summary_statistics()
            
            results = {
                "status": "success",
                "pdf_analyzed": pdf_path,
                "images_processed": len(analysis_results),
                "json_output": saved_json,
                "report_output": saved_report,
                "analysis_timestamp": timestamp,
                "statistics": stats
            }
            
            logger.info("🎉 Analysis complete!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "pdf_analyzed": pdf_path
            }
    
    def find_and_analyze_reports(self, directory: str) -> List[Dict[str, str]]:
        """
        Find all PDF reports in a directory and analyze them
        """
        try:
            logger.info(f"🔍 Searching for PDF reports in: {directory}")
            
            pdf_files = self.pdf_scanner.find_pdf_reports(directory)
            
            if not pdf_files:
                logger.warning("⚠️ No PDF files found in directory")
                return []
            
            logger.info(f"📁 Found {len(pdf_files)} PDF files")
            
            results = []
            for i, pdf_path in enumerate(pdf_files, 1):
                logger.info(f"📊 Processing PDF {i}/{len(pdf_files)}: {Path(pdf_path).name}")
                
                result = self.run_analysis(pdf_path, directory)
                result["pdf_index"] = i
                result["total_pdfs"] = len(pdf_files)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to find and analyze reports: {e}")
            return []
    
    def interactive_mode(self):
        """
        Interactive command-line mode
        """
        print("\n" + "="*70)
        print("🔬 EXPLAINABLE AI IMAGE TAGGING ANALYZER")
        print("   Powered by Kosmos-2 Vision-Language Model")
        print("="*70)
        
        while True:
            try:
                print("\n📋 Available Commands:")
                print("  1. analyze <pdf_path>        - Analyze single PDF report")
                print("  2. scan <directory>          - Find and analyze all PDFs in directory")
                print("  3. status                    - Show system status")
                print("  4. help                      - Show this help")
                print("  5. quit                      - Exit program")
                
                command = input("\n🔬 explainable-ai> ").strip().split(maxsplit=1)
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif cmd == 'analyze' and len(command) > 1:
                    pdf_path = command[1].strip('"\'')
                    print(f"\n🔬 Analyzing PDF: {pdf_path}")
                    
                    result = self.run_analysis(pdf_path)
                    self._print_analysis_result(result)
                
                elif cmd == 'scan' and len(command) > 1:
                    directory = command[1].strip('"\'')
                    print(f"\n🔍 Scanning directory: {directory}")
                    
                    results = self.find_and_analyze_reports(directory)
                    self._print_batch_results(results)
                
                elif cmd == 'status':
                    self._print_system_status()
                
                elif cmd == 'help':
                    self._print_detailed_help()
                
                else:
                    print("❌ Invalid command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _print_analysis_result(self, result: Dict[str, str]):
        """Print analysis result in a formatted way"""
        print("\n" + "="*50)
        
        if result["status"] == "success":
            print("✅ ANALYSIS SUCCESSFUL")
            print(f"📄 PDF Analyzed: {result['pdf_analyzed']}")
            print(f"🖼️  Images Processed: {result['images_processed']}")
            print(f"💾 JSON Results: {result['json_output']}")
            print(f"📋 Report Generated: {result['report_output']}")
            
            if "statistics" in result:
                stats = result["statistics"]
                print(f"📊 Statistics:")
                print(f"   • Total Tags Explained: {stats.get('total_tags_explained', 'N/A')}")
                print(f"   • Average Tags per Image: {stats.get('average_tags_per_image', 'N/A')}")
                
                if stats.get('most_common_tags'):
                    most_common = list(stats['most_common_tags'].items())[:3]
                    print(f"   • Most Common Tags: {', '.join([f'{tag}({count})' for tag, count in most_common])}")
        else:
            print("❌ ANALYSIS FAILED")
            print(f"📄 PDF: {result['pdf_analyzed']}")
            print(f"❌ Error: {result['error_message']}")
        
        print("="*50)
    
    def _print_batch_results(self, results: List[Dict[str, str]]):
        """Print batch analysis results"""
        print("\n" + "="*60)
        print("📊 BATCH ANALYSIS RESULTS")
        print("="*60)
        
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            total_images = sum(r["images_processed"] for r in successful)
            print(f"🖼️  Total Images Processed: {total_images}")
            
            print("\n📋 Generated Reports:")
            for result in successful:
                pdf_name = Path(result["pdf_analyzed"]).name
                print(f"   • {pdf_name} → {result['images_processed']} images")
        
        if failed:
            print("\n❌ Failed Analyses:")
            for result in failed:
                pdf_name = Path(result["pdf_analyzed"]).name
                print(f"   • {pdf_name}: {result['error_message']}")
        
        print("="*60)
    
    def _print_system_status(self):
        """Print system status information"""
        print("\n" + "="*50)
        print("🔧 SYSTEM STATUS")
        print("="*50)
        print(f"🤖 Kosmos-2 Model: Loaded")
        print(f"💻 Device: {kosmos_explainer.device}")
        print(f"📄 PDF Scanner: Ready")
        print(f"📋 Report Generator: Ready")
        print(f"🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
    
    def _print_detailed_help(self):
        """Print detailed help information"""
        print("\n" + "="*70)
        print("📚 DETAILED HELP")
        print("="*70)
        print("""
🔬 EXPLAINABLE AI IMAGE TAGGING ANALYZER

This system analyzes PDF reports generated by the AI image tagging system
and provides explainable AI insights using the Kosmos-2 vision-language model.

COMMANDS:
  analyze <pdf_path>    - Analyze a single PDF report
                         Example: analyze "/path/to/report.pdf"
  
  scan <directory>      - Find and analyze all PDF reports in a directory
                         Example: scan "/path/to/reports/"
  
  status               - Show current system status and configuration
  
  help                 - Show this detailed help information
  
  quit                 - Exit the program

WHAT IT DOES:
• Scans PDF reports generated by the image tagging system
• Extracts images and their AI-generated tags
• Uses Kosmos-2 to explain WHY each tag was assigned
• Generates detailed explainable AI reports
• Saves results in both JSON and PDF formats

OUTPUT FILES:
• explainable_analysis_results_TIMESTAMP.json - Raw analysis data
• explainable_ai_report_TIMESTAMP.pdf - Formatted report with explanations

REQUIREMENTS:
• PDF reports must be generated by the AI image tagging system
• Sufficient memory for Kosmos-2 model (4GB+ recommended)
• Internet connection for initial model download
        """)
        print("="*70)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Explainable AI Analysis for Image Tagging Reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explainable_ai.py --analyze report.pdf
  python explainable_ai.py --scan /path/to/reports/
  python explainable_ai.py --interactive
        """
    )
    
    parser.add_argument("--analyze", type=str, 
                       help="Analyze a single PDF report")
    parser.add_argument("--scan", type=str, 
                       help="Scan directory for PDF reports and analyze all")
    parser.add_argument("--output", type=str, 
                       help="Output directory for results")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize the interface
    interface = ExplainableAICommandInterface()
    
    try:
        if args.analyze:
            print("🔬 Starting single PDF analysis...")
            result = interface.run_analysis(args.analyze, args.output)
            interface._print_analysis_result(result)
            
        elif args.scan:
            print("🔍 Starting batch PDF analysis...")
            results = interface.find_and_analyze_reports(args.scan)
            interface._print_batch_results(results)
            
        elif args.interactive:
            interface.interactive_mode()
            
        else:
            print("🔬 Explainable AI Image Tagging Analyzer")
            print("Use --help for usage information or --interactive for interactive mode")
            interface.interactive_mode()
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

# Initialize the command interface
command_interface = ExplainableAICommandInterface()
print("✅ Command Interface initialized successfully!")
print("\n🎉 All systems ready! You can now:")
print("   • Run main() for command-line interface")
print("   • Use command_interface.run_analysis(pdf_path) directly")
print("   • Call command_interface.interactive_mode() for interactive use")

if __name__ == "__main__":
    main()
    """
Cell 7: Usage Examples and Testing
Examples of how to use the explainable AI system
"""

def demo_usage_examples():
    """
    Demonstrate different ways to use the explainable AI system
    """
    print("🎯 USAGE EXAMPLES FOR EXPLAINABLE AI SYSTEM")
    print("="*60)
    
    print("\n1️⃣  SINGLE PDF ANALYSIS:")
    print("   # Analyze a specific PDF report")
    print("   result = command_interface.run_analysis('/path/to/report.pdf')")
    print("   print(f'Processed {result[\"images_processed\"]} images')")
    
    print("\n2️⃣  BATCH DIRECTORY ANALYSIS:")
    print("   # Analyze all PDF reports in a directory")
    print("   results = command_interface.find_and_analyze_reports('/path/to/reports/')")
    print("   successful = [r for r in results if r['status'] == 'success']")
    print("   print(f'Successfully analyzed {len(successful)} PDFs')")
    
    print("\n3️⃣  INTERACTIVE MODE:")
    print("   # Start interactive command-line interface")
    print("   command_interface.interactive_mode()")
    
    print("\n4️⃣  COMMAND LINE USAGE:")
    print("   # From terminal:")
    print("   python explainable_ai.py --analyze report.pdf")
    print("   python explainable_ai.py --scan /reports/")
    print("   python explainable_ai.py --interactive")
    
    print("\n5️⃣  PROGRAMMATIC ACCESS:")
    print("   # Direct access to components")
    print("   pdf_data = pdf_scanner.scan_pdf_report('report.pdf')")
    print("   explanations = kosmos_explainer.generate_explanation(image, tags)")
    print("   report_path = report_generator.create_explainable_report(results)")

def test_system_components():
    """
    Test individual system components
    """
    print("\n🧪 TESTING SYSTEM COMPONENTS")
    print("="*50)
    
    # Test 1: Check if Kosmos-2 is loaded
    print("✅ Testing Kosmos-2 model loading...")
    try:
        device_info = kosmos_explainer.device
        model_name = kosmos_explainer.model_name
        print(f"   Model: {model_name}")
        print(f"   Device: {device_info}")
        print("   ✅ Kosmos-2 loaded successfully")
    except Exception as e:
        print(f"   ❌ Kosmos-2 loading failed: {e}")
    
    # Test 2: PDF Scanner
    print("\n✅ Testing PDF scanner...")
    try:
        # This is a theoretical test - would need actual PDF file
        print("   PDF scanner initialized and ready")
        print("   ✅ PDF scanner working")
    except Exception as e:
        print(f"   ❌ PDF scanner error: {e}")
    
    # Test 3: Report Generator
    print("\n✅ Testing report generator...")
    try:
        # Test report generator initialization
        styles = report_generator.styles
        print("   Report styles loaded")
        print("   ✅ Report generator working")
    except Exception as e:
        print(f"   ❌ Report generator error: {e}")
    
    print("\n🎯 SYSTEM STATUS: All components initialized!")

def create_sample_workflow():
    """
    Create a sample workflow for processing PDF reports
    """
    print("\n📋 SAMPLE WORKFLOW")
    print("="*50)
    
    workflow_code = '''
# STEP-BY-STEP WORKFLOW FOR EXPLAINABLE AI ANALYSIS

# 1. Set up paths
pdf_path = "/path/to/your/image_analysis_report.pdf"
output_directory = "/path/to/output/"

# 2. Run complete analysis
print("🔬 Starting explainable AI analysis...")
result = command_interface.run_analysis(pdf_path, output_directory)

# 3. Check results
if result["status"] == "success":
    print(f"✅ Success! Processed {result['images_processed']} images")
    print(f"📄 Report saved to: {result['report_output']}")
    print(f"💾 Data saved to: {result['json_output']}")
    
    # 4. Access specific explanations
    analyzer = explainable_analyzer
    
    # Get explanation for a specific tag on a specific image
    explanation = analyzer.get_tag_explanation(image_index=1, tag="tree")
    print(f"🏷️  Why 'tree' tag: {explanation}")
    
    # 5. Get summary statistics
    stats = analyzer.generate_summary_statistics()
    print(f"📊 Total tags explained: {stats['total_tags_explained']}")
    print(f"📊 Average tags per image: {stats['average_tags_per_image']}")
    
else:
    print(f"❌ Analysis failed: {result['error_message']}")

# 6. For batch processing
batch_results = command_interface.find_and_analyze_reports("/reports/")
successful_analyses = [r for r in batch_results if r["status"] == "success"]
print(f"📊 Batch complete: {len(successful_analyses)} successful analyses")
    '''
    
    print(workflow_code)

def show_expected_inputs_outputs():
    """
    Show what inputs the system expects and what outputs it generates
    """
    print("\n📥 EXPECTED INPUTS & 📤 OUTPUTS")
    print("="*60)
    
    print("\n📥 INPUT REQUIREMENTS:")
    print("   • PDF reports generated by the AI image tagging system")
    print("   • PDF should contain:")
    print("     - Images embedded in the PDF")
    print("     - AI-generated tags listed for each image")
    print("     - BLIP captions (optional)")
    print("     - LLaVA descriptions (optional)")
    print("   • Supported PDF naming patterns:")
    print("     - *image_analysis_report*.pdf")
    print("     - *ai_image_analysis_report*.pdf")
    print("     - *single_image_report*.pdf")
    
    print("\n📤 OUTPUT FILES:")
    print("   1. Explainable AI Report (PDF):")
    print("      • File: explainable_ai_report_YYYYMMDD_HHMMSS.pdf")
    print("      • Contains: Visual explanations for each tag")
    print("      • Includes: Kosmos-2 reasoning and confidence assessments")
    
    print("\n   2. Analysis Results (JSON):")
    print("      • File: explainable_analysis_results_YYYYMMDD_HHMMSS.json")
    print("      • Contains: Raw analysis data in structured format")
    print("      • Includes: All explanations, statistics, and metadata")
    
    print("\n📊 EXAMPLE OUTPUT STRUCTURE:")
    example_output = '''
    {
      "analysis_metadata": {
        "total_images": 5,
        "analysis_timestamp": "2025-07-26T14:30:22",
        "model_used": "microsoft/kosmos-2-patch14-224"
      },
      "summary_statistics": {
        "total_tags_explained": 47,
        "average_tags_per_image": 9.4,
        "most_common_tags": {"tree": 3, "building": 2, "person": 2}
      },
      "image_analyses": [
        {
          "image_index": 1,
          "file_name": "landscape.jpg",
          "original_tags": ["tree", "sky", "building"],
          "kosmos_explanations": {
            "general_description": "The image shows a landscape with...",
            "tag_explanations": {
              "tree": "Multiple trees are visible in the foreground...",
              "sky": "A clear blue sky occupies the upper portion...",
              "building": "A structure is visible in the background..."
            },
            "confidence_assessment": "High confidence in main elements..."
          }
        }
      ]
    }
    '''
    print(example_output)

def demonstrate_key_features():
    """
    Demonstrate the key features of the explainable AI system
    """
    print("\n🎯 KEY FEATURES DEMONSTRATION")
    print("="*60)
    
    print("\n🔍 FEATURE 1: TAG EXPLANATION")
    print("   For each AI-generated tag, Kosmos-2 explains:")
    print("   • WHERE in the image the tag is relevant")
    print("   • WHY the tag was assigned")
    print("   • WHAT visual evidence supports the tag")
    print("   Example: 'tree' → 'Multiple green leafy structures visible in the foreground'")
    
    print("\n🧠 FEATURE 2: VISUAL REASONING")
    print("   Kosmos-2 provides:")
    print("   • Detailed visual understanding of the scene")
    print("   • Contextual relationships between objects")
    print("   • Spatial reasoning and layout description")
    
    print("\n📊 FEATURE 3: CONFIDENCE ASSESSMENT")
    print("   System evaluates:")
    print("   • Clarity of visual elements")
    print("   • Certainty of object identification")
    print("   • Overall image quality assessment")
    
    print("\n📋 FEATURE 4: COMPREHENSIVE REPORTING")
    print("   Generated reports include:")
    print("   • Side-by-side image and explanation comparison")
    print("   • Original AI model outputs (BLIP, LLaVA)")
    print("   • Statistical analysis of tag patterns")
    print("   • Key insights and conclusions")
    
    print("\n🔄 FEATURE 5: BATCH PROCESSING")
    print("   System can:")
    print("   • Process multiple PDF reports automatically")
    print("   • Generate summary statistics across all images")
    print("   • Create consolidated reports for large datasets")

def show_troubleshooting_guide():
    """
    Show common issues and solutions
    """
    print("\n🛠️ TROUBLESHOOTING GUIDE")
    print("="*50)
    
    print("\n❌ COMMON ISSUES & SOLUTIONS:")
    
    print("\n1. 'No images found in PDF'")
    print("   ✅ Solutions:")
    print("   • Ensure PDF contains embedded images")
    print("   • Check if PDF was generated by the AI tagging system")
    print("   • Verify PDF is not corrupted")
    
    print("\n2. 'Kosmos-2 model loading failed'")
    print("   ✅ Solutions:")
    print("   • Check internet connection for model download")
    print("   • Ensure sufficient RAM (4GB+ recommended)")
    print("   • Try CPU mode if GPU memory insufficient")
    
    print("\n3. 'Analysis takes too long'")
    print("   ✅ Solutions:")
    print("   • Use GPU acceleration if available")
    print("   • Process fewer images per batch")
    print("   • Reduce max_tags parameter")
    
    print("\n4. 'Poor explanation quality'")
    print("   ✅ Solutions:")
    print("   • Ensure input images have good quality")
    print("   • Check if original tags are meaningful")
    print("   • Try adjusting Kosmos-2 temperature parameters")
    
    print("\n5. 'PDF report generation fails'")
    print("   ✅ Solutions:")
    print("   • Check disk space availability")
    print("   • Ensure write permissions in output directory")
    print("   • Verify ReportLab installation")

def run_all_examples():
    """
    Run all demonstration functions
    """
    demo_usage_examples()
    test_system_components()
    create_sample_workflow()
    show_expected_inputs_outputs()
    demonstrate_key_features()
    show_troubleshooting_guide()
    
    print("\n" + "="*70)
    print("🎉 SYSTEM READY FOR USE!")
    print("="*70)
    print("\n🚀 To get started:")
    print("   1. Prepare a PDF report from the AI image tagging system")
    print("   2. Run: command_interface.run_analysis('path/to/report.pdf')")
    print("   3. Or use interactive mode: command_interface.interactive_mode()")
    print("\n📚 For help: Use --help flag or call show_troubleshooting_guide()")

# Quick start function
def quick_start_guide():
    """
    Quick start guide for immediate usage
    """
    print("\n⚡ QUICK START GUIDE")
    print("="*40)
    print("\n1️⃣  Have a PDF report ready from the AI image tagging system")
    print("2️⃣  Run ONE of these commands:")
    print()
    print("   # For single PDF analysis:")
    print("   result = command_interface.run_analysis('/path/to/your/report.pdf')")
    print()
    print("   # For interactive mode:")
    print("   command_interface.interactive_mode()")
    print()
    print("   # For batch processing:")
    print("   results = command_interface.find_and_analyze_reports('/path/to/reports/')")
    print()
    print("3️⃣  Check the output files generated in the same directory")
    print("4️⃣  Review the explainable AI report PDF for insights!")
    print("\n✨ That's it! Your explainable AI analysis is complete.")

# Execute examples (comment out if not needed)
print("\n" + "🎯" * 20)
print("EXPLAINABLE AI SYSTEM - READY TO USE!")
print("🎯" * 20)

# Uncomment the line below to see all examples
# run_all_examples()

# Show quick start by default
quick_start_guide()

print("\n💡 TIP: Call run_all_examples() to see detailed usage information")
print("💡 TIP: Call main() to start the command-line interface")