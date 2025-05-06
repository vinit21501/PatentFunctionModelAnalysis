# Patent Component Interaction Analyzer

A Streamlit-based application that analyzes patent documents to identify and visualize component interactions using advanced NLP and graph visualization techniques.

## Overview

We have built a working prototype that takes real-world patent documents and applies RAG (Retrieval-Augmented Generation) techniques to extract functional models—a key step in invention analysis and ideation. The system:

- Processes patent documents to identify key components and their interactions
- Generates a component interaction matrix and visual graph representation
- Provides an interactive Q&A interface for detailed patent analysis
- Implements a robust retry mechanism to ensure accurate component extraction
- Uses advanced NLP techniques to understand complex technical relationships
- Supports both text and image extraction from patent documents
- Utilizes Google's Generative AI (Gemini) for enhanced understanding

This tool is particularly useful for:
- Patent analysts and researchers
- Inventors and R&D teams
- Technology transfer specialists
- IP professionals
- Anyone interested in understanding complex technical systems

## Features

- PDF Patent Document Analysis with Image Extraction
- Component Interaction Matrix Generation
- Interactive Graph Visualization
- Question-Answering Interface
- Retry Mechanism for Robust Analysis
- Customizable Chunking and Clustering Settings
- Google Gemini AI Integration
- Image Processing and Analysis

## Prerequisites

- Python 3.8 or higher
- Graphviz (system dependency)
- Google Cloud API Key (for Gemini model)

### Installing Graphviz

- **Windows**: Download and install from [Graphviz Download Page](https://graphviz.org/download/)
- **Linux**: `sudo apt-get install graphviz`
- **macOS**: `brew install graphviz`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd streamlitApp
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a patent PDF document using the file uploader

4. Adjust settings in the sidebar if needed:
   - Chunk Size: Size of text chunks for analysis
   - Chunk Overlap: Overlap between chunks
   - Number of Clusters: For semantic clustering

5. View the analysis results:
   - Component Interaction Matrix
   - Visual Graph Representation
   - Key Components List
   - Component Interactions
   - Extracted Images (if any)

6. Use the Q&A interface to ask questions about the patent

## How It Works

1. **Document Processing**:
   - PDF text and image extraction using PyMuPDF
   - Semantic chunking with LangChain
   - Embedding generation using Google's Generative AI
   - Image processing with Pillow

2. **Analysis**:
   - Component identification using Google Gemini
   - Interaction detection
   - Matrix generation
   - Graph visualization with Graphviz

3. **Interactive Features**:
   - Real-time Q&A with retry mechanism
   - Dynamic graph updates
   - Google Gemini integration

## Project Structure

```
streamlitApp/
├── app.py              # Main Streamlit application
├── pipeline.py         # Core processing pipeline
├── process_data.py     # PDF and image processing
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── README.md          # This file
```

## Dependencies

### Core Dependencies
- streamlit==1.32.0
- python-dotenv==1.0.1
- numpy==1.26.4
- pandas==2.2.1
- scikit-learn==1.4.0
- graphviz==0.20.1

### PDF and Image Processing
- PyPDF2==3.0.1
- PyMuPDF==1.25.5
- Pillow==10.2.0

### AI and Language Models
- langchain==0.1.12
- langchain-google-genai==0.0.11
- langchain-experimental==0.0.50
- google-generativeai==0.4.1

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google for the Gemini model
- Streamlit for the web application framework
- Graphviz for visualization capabilities
- LangChain for the RAG framework 