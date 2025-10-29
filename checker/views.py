from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pymupdf as fitz
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from docx import Document
from .utils import (
    plagiarism_score,
    generate_solution_with_ai,
    cached_solution_path,
    fetch_similar_urls,      
    scrape_multiple_urls,
    analyze_research_paper,
)

def research_paper_check(request):
    """
    Research Paper Check: Upload PDF/DOCX, fetch similar URLs, 
    scrape web content, and perform similarity analysis.
    """
    result = None
    error = None
    processing_status = []
    
    if request.method == "POST" and request.FILES.get("research_paper"):
        research_paper = request.FILES["research_paper"]
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        
        # Step 1: Save and read the research paper
        processing_status.append("✓ Uploaded research paper")
        paper_path = fs.save(f"research_papers/{research_paper.name}", research_paper)
        paper_text = read_file(os.path.join(settings.MEDIA_ROOT, paper_path))
        
        if not paper_text:
            error = "Research paper is empty or contains invalid text."
            return render(request, "checker/research_paper_check.html", {
                "error": error,
                "processing_status": processing_status
            })
        
        processing_status.append(f"✓ Extracted {len(paper_text)} characters from paper")
        
        # Step 2: Fetch similar URLs using Gemini AI
        processing_status.append("⏳ Fetching similar sources from web...")
        urls = fetch_similar_urls(paper_text, max_chars=3000)
        
        if not urls:
            error = "Could not fetch similar URLs. API rate limit may be exceeded."
            return render(request, "checker/research_paper_check.html", {
                "error": error,
                "processing_status": processing_status
            })
        
        processing_status.append(f"✓ Found {len(urls)} potential sources")
        
        # Step 3: Scrape web content from URLs
        processing_status.append("⏳ Scraping web content from sources...")
        scraped_sources = scrape_multiple_urls(urls)
        
        if not scraped_sources:
            error = "Could not extract content from any URLs."
            return render(request, "checker/research_paper_check.html", {
                "error": error,
                "processing_status": processing_status,
                "urls_found": urls
            })
        
        processing_status.append(f"✓ Successfully scraped {len(scraped_sources)} sources")
        
        # Step 4: Analyze similarity with all sources
        processing_status.append("⏳ Analyzing similarity with sources...")
        analysis = analyze_research_paper(paper_text, scraped_sources)
        processing_status.append("✓ Analysis complete!")
        
        result = {
            'analysis': analysis,
            'paper_name': research_paper.name,
            'paper_length': len(paper_text),
            'processing_status': processing_status
        }
    
    return render(request, "checker/research_paper_check.html", {
        "result": result,
        "error": error,
        "processing_status": processing_status
    })


def multi_file_check(request):
    """
    Compare one student file against multiple reference files
    """
    results = []
    error = None
    
    if request.method == "POST" and request.FILES.get("student_file"):
        student_file = request.FILES["student_file"]
        reference_files = request.FILES.getlist("reference_files")  # Multiple files
        
        fs = FileSystemStorage(location="media/")
        
        # Save and read student file
        student_path = fs.save(f"students/{student_file.name}", student_file)
        student_text = read_file(os.path.join("media", student_path))
        
        if not student_text:
            error = "Student file is empty or contains invalid text."
        
        # Compare against each reference file
        for ref_file in reference_files:
            ref_path = fs.save(f"references/{ref_file.name}", ref_file)
            ref_text = read_file(os.path.join("media", ref_path))
            
            if not ref_text:
                results.append({
                    "reference_file": ref_file.name,
                    "similarity": 0.0,
                    "confidence": 0.0,
                    "error": "Reference file is empty or contains invalid text."
                })
                continue
            
            similarity, confidence = plagiarism_score(student_text, ref_text)
            
            results.append({
                "reference_file": ref_file.name,
                "similarity": similarity,
                "confidence": confidence
            })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return render(request, "checker/multi_file_check.html", {"results": results, "error": error})


# Raw text check
def check_plagiarism(request):
    result = None
    if request.method == "POST":
        text1 = request.POST.get("text1")
        text2 = request.POST.get("text2")

        if text1 and text2:
            similarity, confidence = plagiarism_score(text1, text2)
            result = {"similarity": similarity, "confidence": confidence}

    return render(request, "checker/index.html", {"result": result})

# Helper to read docx files

def read_file(file_path):
    """
    Extracts text from PDF, DOCX, or TXT files
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        # Extract text from PDF using PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()  # Ensure no empty text
        
    elif ext == ".docx":
        doc = Document(file_path)
        text = " ".join([para.text for para in doc.paragraphs])
        return text.strip()  # Ensure no empty text
        
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return text.strip()  # Ensure no empty text
    else:
        return ""  # Unsupported format

# File check
def file_check(request):
    result = None
    if request.method == "POST" and request.FILES.get("file1") and request.FILES.get("file2"):
        f1 = request.FILES["file1"]
        f2 = request.FILES["file2"]

        fs = FileSystemStorage(location="media/")
        path1 = fs.save(f1.name, f1)
        path2 = fs.save(f2.name, f2)

        text1 = read_file(os.path.join("media", path1))
        text2 = read_file(os.path.join("media", path2))

        similarity, confidence = plagiarism_score(text1, text2)
        result = {"similarity": similarity, "confidence": confidence}

    return render(request, "checker/file_checker.html", {"result": result})


def assignment_check(request):
    result = None
    saved_paths = {}
    error = None

    if request.method == "POST":
        # Teacher-provided solution or assignment
        assignment_text = request.POST.get("assignment_text", "").strip()
        teacher_file = request.FILES.get("teacher_file", None)

        # Student submission
        student_file = request.FILES.get("student_file", None)
        student_text_input = request.POST.get("student_text", "").strip()

        # Optional max tokens
        try:
            max_tokens = int(request.POST.get("max_tokens", 1000))
        except ValueError:
            max_tokens = 1000

        fs = FileSystemStorage(location=settings.MEDIA_ROOT)

        # 1) Determine reference (teacher solution or generated)
        reference_text = ""
        if teacher_file:
            p = fs.save(f"teacher_refs/{teacher_file.name}", teacher_file)
            saved_paths["teacher_file"] = fs.url(p)
            reference_text = read_file(os.path.join(settings.MEDIA_ROOT, p))
        elif assignment_text:
            try:
                reference_text = generate_solution_with_ai(assignment_text, max_tokens=max_tokens)
                path = cached_solution_path(assignment_text)
                saved_paths["generated_solution"] = os.path.relpath(path, settings.MEDIA_ROOT)
            except Exception as e:
                error = f"AI generation failed: {str(e)}"
        else:
            error = "Please provide an assignment text or upload a teacher reference file."

        # 2) Get student text
        student_text = ""
        if student_text_input:
            student_text = student_text_input
        elif student_file:
            p2 = fs.save(f"student_uploads/{student_file.name}", student_file)
            saved_paths["student_file"] = fs.url(p2)
            student_text = read_file(os.path.join(settings.MEDIA_ROOT, p2))
        else:
            error = (error or "") + " Please provide a student file or paste student text."

        # 3) If no error, run plagiarism check
        if not error and reference_text and student_text:
            similarity, confidence = plagiarism_score(reference_text, student_text)
            result = {
                "similarity": similarity,
                "confidence": confidence,
                "saved_paths": saved_paths,
            }

    return render(
        request,
        "checker/assignment_check.html",
        {"result": result, "error": error},
    )

