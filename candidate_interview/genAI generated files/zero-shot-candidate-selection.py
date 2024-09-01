import json
from transformers import pipeline
from typing import Dict, Any

def parse_transcript(transcript: str) -> str:
    """Parse the transcript JSON string and return a formatted string."""
    transcript_data = json.loads(transcript)
    formatted_transcript = ""
    for pair in transcript_data["pairs"]:
        formatted_transcript += f"Interviewer: {pair[0]}\nInterviewee: {pair[1]}\n\n"
    return formatted_transcript.strip()

def parse_resume(resume: Dict[str, Any]) -> str:
    """Parse the resume dictionary and return a formatted string."""
    data = resume["data"]
    formatted_resume = ""
    
    if data["education"]:
        edu = data["education"][0]
        formatted_resume += f"Education: {edu['degree']} from {edu['school']}, GPA: {edu['GPA']}\n"
    
    if data["projects"]:
        proj = data["projects"][0]
        formatted_resume += f"Project: {proj['projectName']} - {proj['projectDescription']}\n"
    
    if data["skills"]:
        formatted_resume += f"Skills: {', '.join(data['skills'])}\n"
    
    if data["workExperience"]:
        work = data["workExperience"][0]
        formatted_resume += f"Work Experience: {work.get('jobTitle', 'N/A')} at {work.get('company', 'N/A')}\n"
    
    return formatted_resume.strip()

def prepare_candidate_info(transcript: str, resume: Dict[str, Any]) -> str:
    """Combine transcript and resume information into a single string."""
    parsed_transcript = parse_transcript(transcript)
    parsed_resume = parse_resume(resume)
    return f"Interview Transcript:\n{parsed_transcript}\n\nResume:\n{parsed_resume}"

def select_candidate(candidate_a_info: str, candidate_b_info: str, role: str, classifier) -> str:
    """Use zero-shot classification to select the best candidate for the role."""
    combined_text = f"Role: {role}\n\nCandidate A:\n{candidate_a_info}\n\nCandidate B:\n{candidate_b_info}\n\nWhich candidate is a better fit for the role?"
    
    result = classifier(combined_text, candidate_labels=["Candidate A", "Candidate B"], hypothesis_template="This candidate is a good fit for the role.")
    
    return result["labels"][0]

def main():
    # Load your data here (this is a mock-up, replace with your actual data loading)
    candidate_a_transcript = '{"pairs": [["Interviewer: Hello and welcome to the AI interview! This interview will consist of basic questions about your background as well as some high-level questions about the skills you listed on your application. Ensure that you minimize long pauses during your responses, otherwise you may be cut off prematurely. Are you ready to start the interview?", "Interviewee: Yes, I am ready to begin the interview. Thank you for the introduction and guidelines."]]}'
    candidate_b_transcript = '{"pairs": [["Interviewer: Hello and welcome to the AI interview! This interview will consist of basic questions about your background as well as some high-level questions about the skills you listed on your application. Ensure that you minimize long pauses during your responses, otherwise you may be cut off prematurely. Are you ready to start the interview?", "Interviewee: Absolutely, I\'m prepared and looking forward to discussing my background and skills with you."]]}'
    
    candidate_a_resume = {
        "data": {
            "education": [{"degree": "Bachelor of Technology", "school": "Chameli Devi Group of Institutions", "GPA": "7.77"}],
            "projects": [{"projectName": "Merchant Data Capabilities (AMEX)", "projectDescription": "Migrated code from Java to Spark (Scala), leading to faster job executions and significant cost savings."}],
            "skills": ["Java", "Scala", "Spark", "Data Analysis"],
            "workExperience": [{"jobTitle": "Software Engineer", "company": "Tech Solutions Inc."}]
        },
        "status": "success"
    }
    
    candidate_b_resume = {
        "data": {
            "education": [{"degree": "Master of Science in Computer Science", "school": "State University", "GPA": "3.9"}],
            "projects": [{"projectName": "AI-Driven Customer Service Bot", "projectDescription": "Developed an AI chatbot that improved customer service efficiency by 40%."}],
            "skills": ["Python", "Machine Learning", "Natural Language Processing", "AI"],
            "workExperience": [{"jobTitle": "Data Scientist", "company": "AI Innovations LLC"}]
        },
        "status": "success"
    }
    
    role = "Data Scientist with strong communication skills"

    # Prepare candidate information
    candidate_a_info = prepare_candidate_info(candidate_a_transcript, candidate_a_resume)
    candidate_b_info = prepare_candidate_info(candidate_b_transcript, candidate_b_resume)

    # Initialize the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Select the best candidate
    selected_candidate = select_candidate(candidate_a_info, candidate_b_info, role, classifier)

    print(f"The selected candidate for the role of {role} is: {selected_candidate}")

if __name__ == "__main__":
    main()
