Grading model for the American Board of Anesthesiology

Scenario: You are working for a medical certification board exploring the use of short-answer questions for continuing certification exams. You are asked to build a prototype grading model that can automatically score short, free-text responses (1-3 sentences) given a reference “ideal” answer.
Objective: Please share and explain your approach to reach the goal.

Example ideal and candidate answers:

Testing Automated Grading Prototype  

Question: What is the first-line treatment for acute ischemic stroke within 4.5 hours? Ideal Answer: Intravenous alteplase is the first-line treatment for acute ischemic stroke within 4.5 hours, provided no contraindications exist. Candidate Response: IV tissue plasminogen activator within 4.5 hours if eligible. Expert Score: 4 Model Score: 4 Feedback: Score: 4/5. Missing keywords: contraindications. -------------------------------------------------------------------------------- 

Question: What is the first-line treatment for acute ischemic stroke within 4.5 hours? Ideal Answer: Intravenous alteplase is the first-line treatment for acute ischemic stroke within 4.5 hours, provided no contraindications exist. Candidate Response: Aspirin is the best treatment for stroke. Expert Score: 2 Model Score: 2 Feedback: Score: 2/5. Missing keywords: alteplase, hours, contraindications. -------------------------------------------------------------------------------- 

Question: What is the primary cause of type 2 diabetes? Ideal Answer: Type 2 diabetes is primarily caused by insulin resistance and relative insulin deficiency. Candidate Response: Insulin resistance causes type 2 diabetes. Expert Score: 3 Model Score: 3 Feedback: Score: 3/5. Missing keywords: deficiency. 
-------------------------------------------------------------------------------- 

Question: What is the recommended initial test for suspected pulmonary embolism? Ideal Answer: A CT pulmonary angiography is the recommended initial test for suspected pulmonary embolism. Candidate Response: D-dimer test is the first step for pulmonary embolism. Expert Score: 3 Model Score: 2 Feedback: Score: 2/5. Missing keywords: angiography, ct. --------------------------------------------------------------------------------