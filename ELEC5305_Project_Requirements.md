# ELEC5305 Speech and Audio Processing
## Project Requirements Document

**Course:** ELEC5305 - Speech and Audio Processing
**Institution:** The University of Sydney
**School:** Electrical and Information Engineering

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [General Description](#general-description)
3. [Project Scope and Assessment](#project-scope-and-assessment)
4. [Project Ideas and Resources](#project-ideas-and-resources)
5. [Submission Requirements](#submission-requirements)
6. [Marking Criteria](#marking-criteria)
7. [Academic Integrity](#academic-integrity)

---

## Project Overview

A major part of the class will be the projects undertaken by students in some area of speech and audio processing and recognition. Students are encouraged to start thinking about their project from the earliest possible date and to discuss ideas with the instructor to develop the best project plans. Various resources, such as corpora of sound files or access to existing software tools, will be provided where possible.

### Project Components

Each project will consist of THREE main deliverables:

1. **Written Research Report**
2. **Working Code (via GitHub)**
3. **Video Presentation**

The emphasis is on communication and sharing ideas and knowledge. Demonstrations and sound examples are particularly encouraged.

---

## General Description

### Written Report Format

The written report should follow the broad format of a research publication, including:

- **Introduction** - Describing the problem
- **Approach Description** - Methodology and methods
- **Results Presentation** - Data, figures, and analysis
- **Discussion** - Interpretation of results (optional but recommended)
- **Conclusions** - Final summary and findings

**Report Length:** Typically 10 pages in length, including figures.

**Format Options:**
- Single document (printed or online)
- Set of web pages (with linked examples)

### Video Presentation

Each project will be described by a video presentation that includes:

- Description of the signal processing code
- Demonstration of the code in action
- Explanation of results and findings

---

## Project Scope and Assessment

### Expected Workload

The project accounts for **30% of the marks** in the class, which corresponds to approximately **half a day per week** over the semester.

### Project Philosophy

**Important:** The best projects have **simple and clear concepts** at their core, rather than ballooning into vast investigations.

### Recommended Project Structure

Here's a recipe for one possible project 'shape':

#### 1. Identify the Area
Choose a specific area of investigation within speech and audio processing.

#### 2. Define a Concrete Task
For instance:
- **Sound classification:** Define the set of target classes and a corpus for testing
- **Voice transformation:** Set a specific goal (e.g., making a male speaker sound female)
- Task should be explicit, concrete, and have an identified target domain

#### 3. Define Evaluation Metrics
This is crucial for quality research. Without measures of progress, it's easy to get lost.

**Examples:**
- **Classification problems:** Error rates on a test set
- **Transformation tasks:** Subjective success measures (e.g., "does it sound like a male or female?")
- Consider both objective metrics (measurable) and subjective testing (when relevant)

#### 4. Identify Your Approach
- Types of features you plan to extract
- Basic signal processing sequence
- Overall methodology

#### 5. Make an Implementation
Create and debug your implementation.

#### 6. Measure Performance
- Use your evaluation metrics
- Conduct qualitative investigation into shortcomings
- Analyze: What differs from what you hoped or intended?
- Consider: How might it be improved?

#### 7. Iterate and Improve
Based on analysis:
- Modify your implementation or create a new one
- Address shortcomings identified in step 6
- Assess the new iteration
- Compare to the original
- Analyze: Were you able to improve? How has performance changed?

#### 8. Repeat (if time permits)
Continue the cycle of improvement.

#### 9. Reflect
Finally, step back and look at the whole path:
- If starting over, how would you do it differently?
- What have you learned about the nature of the problem?
- What are the most promising avenues for future work?

### Key Principles

1. **Well-defined goals** - Be clear about objectives
2. **Evaluation standards** - Know what's relevant and what isn't
3. **Iterative development** - Analyze performance and improve
4. **Systematic approach** - Follow a structured methodology

---

## Project Ideas and Resources

### Some Project Ideas

This list is meant to stimulate ideas, not to define a limited domain. Interesting ideas outside these categories are also encouraged.

#### 1. Speech Recognition Variants
- Use existing speech recognition frameworks
- Focus on modifying a specific aspect:
  - Feature representations
  - Model structures
  - Training procedures
- Make quantitative measures of impact on recognizer performance

#### 2. Audio Compression Variants
- Investigate different ideas for audio signal compression
- Start from scratch or modify available packages
- Measure bitrate reductions and quality

#### 3. Nonspeech Signal Recognition
- Apply speech recognition techniques to other sounds:
  - Alarm sounds
  - Acoustic events in movie soundtracks
  - Animal calls
- Build recognizers with suitable corpora and well-defined experiments

#### 4. Speaker Identification and Characterization
- Extract information about the speaker rather than words:
  - Gender identification
  - Age estimation
  - Country of origin detection
- Find appropriate features and train recognizers

#### 5. Spatial Location Analysis and Synthesis
- Use auditory spatial perception models
- Recognize sound origin (azimuth, elevation, range)
- Synthesize sounds appearing to come from specific points in space

#### 6. Prosody Detection
- Investigate variable aspects of speech apart from phonetic content:
  - Pitch (melody)
  - Timing
  - Stress
- Extract reliable correlates from speech signals

#### 7. Music Synthesis
- Investigate algorithms used in computer and electronic music
- Compare and extend existing approaches

#### 8. Music Analysis
- Automatic transcription (challenging but interesting)
- Extract other information:
  - Rhythm analysis
  - Genre classification
  - Instrument identification
  - Chord progressions
  - Bass lines

#### 9. Audio and Music Retrieval
- Define similarity between sounds
- Build search engines working purely on sound
- Examine or develop approaches for audio retrieval

#### 10. Temporal Structure Recovery
- Analyze soundtrack to infer what's happening
- Recognize sound effects or respond to soundtrack music
- Recover coarse-time structural information from multimedia content

### Project Code Starters

#### From MATLAB

The following MATLAB examples can serve as starting points:

1. Speech Command Recognition Using Deep Learning
2. Cocktail Party Source Separation Using Deep Learning Networks
3. Keyword Spotting in Noise Using MFCC and LSTM Networks
4. Denoise Speech Using Deep Learning Networks
5. Train Generative Adversarial Network (GAN) for Sound Synthesis
6. Voice Activity Detection in Noise Using Deep Learning
7. Classify Gender Using LSTM Networks
8. Spoken Digit Recognition with Wavelet Scattering and Deep Learning
9. Speech Emotion Recognition
10. Acoustic Scene Recognition Using Late Fusion

#### From the Web

**Open Source Speech Technology:**
- Mozilla Voice: https://voice.mozilla.org/
- Open source speech-to-text engine
- Open source text-to-speech engine

**Music Processing:**
- Music Genre Classification: GTZAN dataset
- Cover Song Identification
- Librosa library projects (GitHub)
- Music emotion recognition
- Music video generation
- Speech to face applications

**Additional Resources:**
- Audio-related speech and language: https://huggingface.co/
- Music processing tutorials and examples

### Areas for Further Exploration

Several resources are provided for deeper investigation:

1. **Text-to-Speech Systems**
2. **Deep Learning with Audio Courses**
   - Aalto University course materials
   - Course GitHub repositories
3. **Music Processing Books and Materials**
   - Fundamentals of Music Processing
   - Jupyter Notebooks for music processing
4. **Audio Programming**
   - Pure Data
   - Music Signal Processing via Pure Data
5. **Deep Learning Resources**
   - Coursera Deep Learning Specialisation
   - Deep Learning for Music
   - Valerio Valardo tutorials
6. **Creative Audio**
   - Magenta Art and Music

---

## Submission Requirements

### Deadline
**End of Week 13**

### Three Required Components

#### 1. Working Code (GitHub Repository)

**Requirements:**
- All code must be downloadable via GitHub
- Code must be WORKING and executable
- Include clear instructions to run the code
- Provide a clickable link to the GitHub site

**What to Include:**
- Source code files
- README with setup instructions
- Example data or instructions to obtain data
- Any necessary configuration files

**GitHub Site Should Contain:**
- Project description
- Installation instructions
- Usage examples
- Link to your report
- Link to your video demonstration

#### 2. Written Project Report

**Format:** Research paper style

**Required Sections:**

##### Introduction (5%)
**Must include:**
- Structure with specific statements about the research/design field
- Introduction of key authors
- Link between aim and existing research/design work
- Analysis of literature indicating gap in existing work
- Outline of scope and rationale for the project

##### Literature Review (15%)
**Must include:**
- Comprehensive and analytical examination of the topic
- Current state of the art or understanding with regards to research question
- Identification of methods or approaches to answer the research question
- Justification for selected methods with advantages and disadvantages
- Explanation of unknown concepts learned during research
- Clear structure grouping literature into themes relevant to research topic
- Links to your own project

##### Methodology (15%)
**Must include:**
- Complete description of the method
- Justification/explanation of how approach answers research question
- Clarification of new tools or techniques learned
- Description of approach/method taken to address research question
- Research methods and materials described so they could be repeated
- Methods showing structure that might yield appropriate data
- Rationale drawn from published research

##### Results (10%)
**Must include:**
- Presentation of results in intuitive and clear form:
  - Performance graphs
  - Confusion matrices (for classifiers)
  - Accuracy tables
  - Computational efficiency metrics
  - Training time
  - Model complexity comparisons
- Sufficient accompanying text for reader understanding
- Reproducible data in logical order reflecting research aim
- Figures and tables integrated with clear written legends
- Complete, precision data with appropriate analytical techniques
- Links to research aim/question
- Discussion of sources of error

##### Discussion/Conclusion (25%)
**Must include:**
- Clear interpretation of results
- Links to theoretical understanding from literature
- Substantiation of research claims with references
- Comparison and explanation of (un)expected results with published results
- Anticipation of criticism
- Identification of limitations and potential resolutions
- Suggestions for further work related to topic
- Description of what worked well and what did not
- Comparison in context of previous work

##### Presentation Format & References (10%)
**Requirements:**
- Clear writing without grammatical errors
- Consistent and clear style
- Sections and subsections with contents page
- Correct departmental formatting
- Proper referencing following academic standards
- All sections integrated into cohesive document

##### Originality & Personal Contribution (10%)
**Requirements:**
- Work must be the student's own
- Acknowledgment of external sources
- Compliance with University Academic Integrity Policy
- Places results in credible research context
- Makes a contribution to the topic

##### Command of Subject (10%)
**Requirements:**
- Links theory to research
- Uses theory to inform research/design question
- Demonstrates understanding of topic
- Uses models to inform research/design aim
- Compares and contrasts several theories
- Reveals strengths and weaknesses of complex theoretical models

**Report Length:** Approximately 10 pages (but comprehensive work may be longer)

**Submission Format:**
- Upload report to Canvas
- Provide link in GitHub repository

#### 3. Video Demonstration

**Content Requirements:**
- Brief demonstration highlighting accomplishments
- Description of signal processing code
- Demonstration of code in action
- Show-and-tell format

**Length:** No strict limit, but keep it concise and focused

**What to Include:**
- Introduction to the project
- Explanation of the approach
- Live demonstration of code running
- Explanation of results
- Key findings and conclusions

**Submission:**
- Upload video to Canvas
- Provide link on GitHub site
- Include link in report

### Submission Checklist

Before submitting, ensure you have:

- [ ] GitHub repository created and code uploaded
- [ ] GitHub repository is PUBLIC and accessible
- [ ] README.md in GitHub with clear instructions
- [ ] Written report completed (all sections)
- [ ] Report uploaded to Canvas
- [ ] Report includes GitHub link
- [ ] Video recorded and demonstrates code
- [ ] Video uploaded/linked
- [ ] Video link included in report and GitHub
- [ ] All three components cross-reference each other

---

## Marking Criteria

### Grading Rubric

Projects will be graded on several dimensions according to the following criteria:

#### 1. Project Structure
How well the basic investigation is defined, how systematically it is pursued, and how well the effort was balanced between different areas.

#### 2. Technical Content
The breadth and depth of understanding of audio processing-related ideas displayed within the project.

#### 3. Presentation
How well the ideas and results of the project are communicated.

### Detailed Marking Scheme

| Criteria | Fail (<50%) | Pass (50-74%) | Credit (75-84%) | Distinction (85%+) |
|----------|-------------|---------------|-----------------|-------------------|
| **Originality & Personal Contribution (10%)** | Does not meet Academic Board Policy | Work is student's own; acknowledgments present; meets policy | Places results in credible research context | Makes a contribution to the topic |
| **Command of Subject (10%)** | Does not link theory to research | Describes and uses theory to inform research question; uses readings | Demonstrates understanding; uses models to inform aim | Compares/contrasts theories; reveals strengths/weaknesses |
| **Introduction (5%)** | Absent, poorly structured, or lacks elements | Contains structure; describes project generally | Makes specific statements; introduces key authors; links to existing work | Analyzes literature; indicates gap; outlines scope with rationale |
| **Literature Review (15%)** | Too short; lacks detail/analysis; doesn't cite important work | Reports literature; quotes/paraphrases appropriately; grasps key issues | Clear structure; groups into themes; clear link to own project | Comprehensive analytical examination; links to methodology; sound understanding |
| **Methodology (15%)** | Uses inappropriate methods; lacks structure/argument | Describes methods so they could be repeated; shows structure; might yield data | Provides rationale from published research; logical link to results | Derives from analysis of existing work; sound rationale for project |
| **Results (10%)** | Insufficient data or doesn't fulfill purpose | Sufficient reliable data; supported by figures/tables | Reproducible data in logical order; integrated figures/tables with legends | Complete precision data; appropriate analysis; links to aim; discusses errors |
| **Discussion/Conclusion (25%)** | Cannot reasonably explain results | Makes links with basic reasoning; states usefulness | Substantiates claims with references; compares/explains results; suggests further work | Clearly interprets; links to theory; anticipates criticism; identifies limitations |
| **Presentation & References (10%)** | Writing doesn't clearly communicate | Writes well; contains sections; correct formatting/referencing | Consistently clear style without grammatical errors | Writes analytically; cohesive document |

### Important Notes on Grading

**Conciseness is a virtue:** Blindly generating vast volumes of results is a warning sign. Step back and refocus on objectives.

**Quality over Quantity:** The emphasis is on well-defined investigations pursued systematically, not on the volume of output.

---

## Academic Integrity

### Requirements

You are required to take part in your education in an honest and ethical manner. Failure to comply with assessment rules and University policies could result in investigation and penalties for academic integrity breaches.

### Turnitin

The university uses Turnitin to help detect potential academic integrity breaches. In some cases, you may be permitted multiple attempts and can view the Turnitin report immediately to help revise your submission.

### Compliance Statement

In submitting this work, I (or we, in the case of a group submission) acknowledge that:

1. I have read and understood the Academic Integrity Policy, and where relevant, the Research Code of Conduct
2. This assessment submission complies with these policies
3. I have complied with all rules and referencing requirements set for this assessment task
4. I have correctly acknowledged the use of any generative AI tools or assistance from others such as copyediting or proofreading
5. The work has not previously been submitted in part or in full for assessment in another unit unless permission has been given
6. Engaging another person to complete part or all of the submitted work will, if detected, lead to proceedings for potential student misconduct

### Important Reminders

- Keep copies of your assignment submission, drafts, AI outputs, and research materials for one year
- Any use of AI tools must be acknowledged
- Assistance from others (copyediting, proofreading) must be acknowledged
- Previous submissions cannot be reused without permission

---

## Tips for Success

### Getting Started

1. **Start Early:** Begin thinking about your project as soon as possible
2. **Discuss Ideas:** Talk with the instructor to develop the best project plans
3. **Keep It Simple:** The best projects have simple, clear concepts at their core
4. **Define Clear Goals:** Know what you're trying to achieve and how you'll measure success

### During Development

1. **Stay Systematic:** Follow a structured approach to your investigation
2. **Evaluate Regularly:** Use your evaluation metrics to track progress
3. **Iterate Thoughtfully:** Analyze results and make informed improvements
4. **Document Everything:** Keep good records of your process and findings

### Before Submission

1. **Test Your Code:** Ensure everything runs correctly
2. **Write Clearly:** Make your report easy to understand
3. **Show Your Work:** Demonstrate what you've accomplished in your video
4. **Cross-Check:** Ensure all three components are complete and connected
5. **Review Requirements:** Use the submission checklist

### Resources and Support

- **Office Hours:** Use instructor office hours to discuss ideas and get feedback
- **Course Materials:** Leverage provided resources, corpora, and software tools
- **Online Resources:** Explore the suggested project starters and code examples
- **Peer Discussion:** Learn from classmates (while maintaining academic integrity)

---

## Contact Information

For questions about project requirements, clarifications, or to discuss project ideas, please contact the course instructor through the official course communication channels.

---

**Document Version:** 1.0
**Last Updated:** November 2025
**Course:** ELEC5305 - Speech and Audio Processing
**Institution:** The University of Sydney

---

*This document summarizes the project requirements for ELEC5305. Students should also refer to the official course materials and announcements for any updates or additional information.*
