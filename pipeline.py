import logging
from process_data import extract_pdf_content, NumpyEncoder
import json
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from sklearn.cluster import KMeans
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_and_process_pdf(pdf_path):
    """Load and process PDF content."""
    try:
        logger.info(f"Loading PDF from: {pdf_path}")
        pdfdata = extract_pdf_content(pdf_path)
        pdfdataString = json.dumps(pdfdata, cls=NumpyEncoder)
        pdfdataString = " ".join(pdfdataString.split())
        pdfdataString = pdfdataString.replace('\t', ' ')
        return pdfdataString
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def create_text_chunks(text, max_tokens=1024):
    """Create semantic text chunks using SemanticChunker."""
    try:
        logger.info("Creating semantic text chunks with SemanticChunker")
        # Initialize Google Generative AI Embeddings
        embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Initialize SemanticChunker
        semantic_chunker = SemanticChunker(embed_model)
        documents = [Document(page_content=text)]
        semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])
        return [doc.page_content for doc in semantic_chunks]
    except Exception as e:
        logger.error(f"Error creating semantic text chunks: {str(e)}")
        raise

def get_embeddings(chunks):
    """Generate embeddings for the chunks using GoogleGenerativeAIEmbeddings."""
    try:
        logger.info("Generating embeddings with GoogleGenerativeAIEmbeddings")
        if not chunks:
            raise ValueError("No chunks provided for embedding")
        valid_chunks = [chunk for chunk in chunks if chunk and len(chunk.strip()) > 0]
        if not valid_chunks:
            raise ValueError("No valid chunks found after filtering")
        logger.info(f"Processing {len(valid_chunks)} valid chunks for embedding")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        vectors = embeddings.embed_documents(valid_chunks)
        vectors = np.array(vectors)
        logger.info("Generated embeddings")
        return valid_chunks, vectors
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def perform_clustering(vectors, num_clusters=10):
    """Perform K-means clustering and select best chunks."""
    try:
        logger.info(f"Performing K-means clustering with {num_clusters} clusters")
        
        # Validate input vectors
        if len(vectors) == 0:
            raise ValueError("No vectors provided for clustering")
            
        # Ensure num_clusters is not larger than number of vectors
        num_clusters = min(num_clusters, len(vectors))
        logger.info(f"Adjusted number of clusters to {num_clusters}")
        
        # Initialize and fit K-means
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10
        ).fit(vectors)

        # Find closest chunks to cluster centers
        closest_indices = []
        for i in range(num_clusters):
            try:
                distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
                closest_index = np.argmin(distances)
                closest_indices.append(closest_index)
                logger.info(f"Found closest chunk for cluster {i+1}")
            except Exception as e:
                logger.error(f"Error finding closest chunk for cluster {i+1}: {str(e)}")
                raise

        selected_indices = sorted(closest_indices)
        logger.info(f"Selected {len(selected_indices)} representative chunks")
        return selected_indices
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        raise

def initialize_llm():
    """Initialize the language model."""
    try:
        logger.info("Initializing language model")
        
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        
        logger.info("Language model initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

def create_summaries(chunks, selected_indices, llm, prompt=None):
    """Create summaries for selected chunks."""
    try:
        logger.info("Creating summaries for selected chunks")
        
        # Validate input
        if not chunks or not selected_indices:
            raise ValueError("No chunks or indices provided")
            
        # Use default prompt if none provided
        if prompt is None:
            prompt = """
            Summarize the following patent section (including any referenced images) with a focus on:
            Key technical details (only essential information, no repetition).
            List all components mentioned, along with their specific functions or interactions with other components.
            Clearly describe how each component is connected, used, or functions in relation to others—this should support the creation of a component interaction matrix.
            Be precise and structured. Do not include background info, claims language, or redundant descriptions.
            ```{text}```
            """
            
        map_prompt_template = ChatPromptTemplate.from_template(prompt)
        
        # Process selected chunks
        summary_list = []
        
        for i, idx in enumerate(selected_indices):
            if idx >= len(chunks):
                continue
                
            chunk = chunks[idx]
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                # Clean and prepare the chunk
                chunk = " ".join(chunk.split())
                if not chunk:
                    logger.warning(f"Skipping empty chunk {i+1}")
                    break
                
                # Create a new document for the chunk
                chunk_doc = Document(page_content=chunk)
                
                try:
                    # Generate summary using the LLM
                    result = llm.invoke(map_prompt_template.format(text=chunk))
                    
                    # Extract the summary text
                    summary = result.content if hasattr(result, 'content') else str(result)
                    
                    # Clean prompt from summary
                    summary = summary.replace(prompt, "").strip()
                    
                    print(f"Generated summary for chunk {i+1}:")
                    # print(summary)
                    
                    # Validate the summary
                    if summary and len(summary.strip()) > 0:
                        summary_list.append(summary)
                        logger.info(f"Successfully processed chunk {i+1}/{len(selected_indices)}")
                        break
                    else:
                        raise ValueError("Generated empty summary")
                        
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Attempt {retry_count} failed for chunk {i+1}: {str(e)}")
                    if retry_count == max_retries:
                        logger.error(f"Failed to process chunk {i+1} after {max_retries} attempts")
                        break
        
        if not summary_list:
            raise ValueError("No valid summaries were generated")
            
        # Combine all summaries
        combined_summary = "\n\n".join(summary_list)
        return Document(page_content=combined_summary)
        
    except Exception as e:
        logger.error(f"Error creating summaries: {str(e)}")
        raise

def generate_final_summary(summaries, llm):
    """Generate the final summary with interaction matrix."""
    try:
        logger.info("Generating final summary with interaction matrix")
        
        # Validate input
        if not summaries or not summaries.page_content:
            raise ValueError("No summary content provided")

        combine_prompt = """
        Given the patent summary, do the following:
        Extract all key components.
        Identify how each component interacts with the others (function or relation).
        Create an interaction matrix:
            Rows and columns = components.
            Cells = interaction/function (e.g., "contains", "couples with").
            Use - if unknown or not mentioned.
        List:
            Independent claims and their components.
            Dependent claims and which independent claim they refer to.
        Format the output clearly.
        ```{text}```
        RESPONSE:
        """

        combine_prompt_template = ChatPromptTemplate.from_template(combine_prompt)
        
        try:
            # Generate final summary using the LLM
            result = llm.invoke(combine_prompt_template.format(text=summaries.page_content))
            
            # Extract the summary text
            summary = result.content if hasattr(result, 'content') else str(result)
            
            # Clean prompt from summary
            summary = summary.replace(combine_prompt, "").strip()
            
            logger.info("Final summary generated successfully")
            return summary
                
        except Exception as e:
            logger.error(f"Error generating final summary: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in final summary generation: {str(e)}")
        raise

def main():
    """Main function to run the RAG pipeline."""
    try:
        # Input PDF path
        pdf_path = "data/patents/US20130058155A1_SRAM DIMIENSIONED TO PROVIDE BETA RATO SUPPORTING READ STABILITY AND REDUCED WRITE TIME.pdf"
        
        # Process PDF
        pdf_text = load_and_process_pdf(pdf_path)
        
        # Initialize LLM
        llm = initialize_llm()
        
        # Create semantic chunks
        chunks = create_text_chunks(pdf_text)
        
        # Generate embeddings and create vector store
        vector_store, valid_chunks, vectors = get_embeddings(chunks)
        
        # Perform clustering to select best chunks
        selected_indices = perform_clustering(vectors, 2)
        
        # Create summaries for selected chunks
        summaries = create_summaries(valid_chunks, selected_indices, llm)
        
        # Generate final summary
        final_output = generate_final_summary(summaries, llm)
        
        logger.info("RAG pipeline completed successfully")
        print(final_output)
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()