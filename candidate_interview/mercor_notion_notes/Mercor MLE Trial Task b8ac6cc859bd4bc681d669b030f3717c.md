# Mercor MLE Trial Task:

### **Overview & Objective:**

Mercor’s goal is to make hiring more efficient, meritocratic, and accessible with AI. When a company hires a candidate through our platform, they navigate to [team.mercor.com](http://team.mercor.com) and enter a query in natural language. Shortly after, they see a ranked list of candidates that would be the best fit for the given query. 

Ranking candidates who might be the best fit for a role is a complex task. In this trial, you will leveraging LLMs to compare two candidates and decide which is a better fit for the query. You will be given access to the resumes and AI interview transcripts of two candidates and your model should determine which candidate should be preferred for a given query (role). 

You will evaluate the model's performance using the provided test data, which is real preference data from expert hiring managers. Additionally, **complete trial submissions will be paid $50 / ~4,000 INR to cover costs of experiments and to compensate you for your time.** You will receive information regarding the reimbursement to your email.

### **Task Details:**

- **Data Provided:** Each row in the CSV contains the resume and interview transcript data of two candidates who were compared against each other.
    - candidateAId - ID of Candidate A
    - candidateBId - ID of Candidate B
    - winnerId (ground truth) - The ID of the preferred candidate between the two. The model should correctly predict the preferred candidate.
    - candidateATranscript - Interview transcript of Candidate A
    - candidateBTranscript - Interview transcript of Candidate B
    - role - Input query specifying the role for which the two candidates are being considered
    - candidateAResumeData - Resume data of Candidate A
    - candidateBResumeData - Resume data of Candidate B
- **Main Goal:**
    - Using the training data, achieve the highest generalizable performance on the test data.
    - Explore training, fine-tuning, reasoning chains, few-shot prompting, or other techniques to achieve consistent results on the test dataset. Although the train dataset is small, you should be able to go quite far leveraging it for LLM fine-tuning or few-shot prompting.
    - As with most ML datasets, there is inherent bias. You should be aware of this bias and report on a) inherent biases in the dataset and b) how you would control for them in your model.
- **Deliverables:**
    - A minimum 1-page writeup (PDF) that includes an explanation of the experiments you conducted, code to run the experiments, and their performance. You should also include an explicit analysis of a) inherent biases in the dataset and b) how you would control for them in your writeup, since this is a crucial part of any MLE role. It would also be okay to submit a Jupyter Notebook or something comparable.
    - Final self-reported accuracy on the test set. Just the number is fine, but we will be examining your code and replicating your results. Any dishonesty will not be tolerated and will result in immediate disqualification from the role.
- **Next steps:** we will be reaching out to the top 5 candidates with onboarding information and job offers!

---

### **Note:**

1. Only use the training dataset for training, fine-tuning, or in-context learning techniques.

### **Submission:**

[1 page Write-up - Mercor take away assign](https://www.notion.so/1-page-Write-up-Mercor-take-away-assign-fe99456fda9f452d93aa84fb37485976?pvs=21)

Use the following Google form link for your final submission:
[https://forms.gle/pMNyPJNttAHRaevA8](https://forms.gle/pMNyPJNttAHRaevA8)

### Dataset Files:

[train_dataset.csv](Mercor%20MLE%20Trial%20Task%20b8ac6cc859bd4bc681d669b030f3717c/train_dataset.csv)

[test_dataset.csv](Mercor%20MLE%20Trial%20Task%20b8ac6cc859bd4bc681d669b030f3717c/test_dataset.csv)

Talks with GenAI

[claude ](https://www.notion.so/claude-e1f5c238eb0f4892b709087c963027e1?pvs=21)

[phind](https://www.notion.so/phind-2d26a0a2b4a0427482855133f3ad7e46?pvs=21)