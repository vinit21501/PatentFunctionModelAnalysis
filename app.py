import streamlit as st
import os
from pipeline import (
    load_and_process_pdf,
    create_text_chunks,
    get_embeddings,
    perform_clustering,
    initialize_llm,
    create_summaries,
    generate_final_summary
)
import tempfile
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
import re
import hashlib
from graphviz import Digraph
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Patent Analysis RAG Pipeline",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .settings-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("Patent Component Interaction Analyzer")
    st.markdown("""
    Welcome to the Patent Component Interaction Analyzer!
    
    **How it works:**
    1. Upload a patent PDF.
    2. The app will extract, semantically chunk, and analyze the document to identify key components and their interactions.
    3. View a clear, interactive matrix of component relationships.
    4. Ask questions about the analyzed patent in a chat-like interface.
    
    _Upload a new PDF at any time to analyze a different patent._
    """)

    # Settings in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Chunk settings
        st.markdown("#### Text Chunk Settings")
        chunk_size = st.slider("Chunk Size (tokens)", min_value=256, max_value=10240, value=1024, step=128)
        chunk_overlap = st.slider("Chunk Overlap (tokens)", min_value=32, max_value=1280, value=128, step=32)
        
        # Clustering settings
        st.markdown("#### Clustering Settings")
        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=2, step=1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses:
        - Model for text generation
        - Semantic Chunk for embeddings
        - K-means clustering for content selection
        """)

    # File uploader
    uploaded_file = st.file_uploader("Upload Patent PDF", type=['pdf'])

    # Helper to get a unique file id
    def get_file_id(uploaded_file):
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        return hashlib.md5(file_bytes).hexdigest()

    file_id = None
    if uploaded_file is not None:
        file_id = get_file_id(uploaded_file)

    # Only process if new file or not cached
    if file_id and (st.session_state.get('file_id') != file_id):
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        try:
            with st.spinner("Processing patent document..."):
                progress = st.progress(0, text="Extracting text from PDF...")
                start_time = time.time()
                pdf_text = load_and_process_pdf(pdf_path)
                progress.progress(20, text="Initializing language model...")
                llm = initialize_llm()
                progress.progress(30, text="Chunking text...")
                chunks = create_text_chunks(pdf_text, max_tokens=chunk_size)
                progress.progress(40, text="Generating embeddings...")
                valid_chunks, vectors = get_embeddings(chunks)
                progress.progress(60, text="Clustering chunks...")
                selected_indices = perform_clustering(vectors, num_clusters)
                progress.progress(70, text="Summarizing clusters...")
                summaries = create_summaries(valid_chunks, selected_indices, llm)
                progress.progress(85, text="Generating interaction matrix...")
                final_output = generate_final_summary(summaries, llm)
                elapsed = time.time() - start_time
                progress.progress(100, text=f"Done! (Processed in {elapsed:.1f} seconds)")
                time.sleep(0.5)
                progress.empty()

            # Store for QA reuse and caching
            st.session_state['file_id'] = file_id
            st.session_state['valid_chunks'] = valid_chunks
            st.session_state['vectors'] = vectors
            st.session_state['summaries'] = summaries
            st.session_state['llm'] = llm
            st.session_state['final_output'] = final_output
            st.session_state['qa_history'] = []

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    # --- Post-process and display the interaction matrix ---
    if 'final_output' in st.session_state:
        def readNodes(dfm):
            # Get the first column name (which should be the components column)
            dfm.reset_index(inplace=True)
            components_col = dfm.columns[0]
            obj = list(dfm[components_col].str.replace("*", ""))
            d = []
            # Iterate through all columns except the first one
            for col in dfm.columns[1:]:
                for i, f in enumerate(dfm[col]):
                    if pd.notna(f) and str(f).strip() != "-":
                        d.append([obj[i], str(f).strip(), col])
            return d

        def makeM(data, filename):
            dot = Digraph(engine='dot')
            dot.attr(dpi='300', overlap='false', rankdir='LR', splines='polyline')
            dot.attr('node', shape='Mrecord', color='lightblue', style='filled', fontsize='12', width='1.5')
            dot.attr('edge', fontsize='10', color='black', style='solid', dir='forward')

            for node1, label, node2 in data:
                dot.node(node1, label=node1)
                dot.node(node2, label=node2)
                dot.edge(node1, node2, label=label)
            dot.render(filename.removesuffix('.csv'), format='png', cleanup=True)

        def extract_first_markdown_table(text):
            lines = text.splitlines()
            table_lines = []
            in_table = False
            header_cols = 0

            for i, line in enumerate(lines):
                # Detect header
                if not in_table and line.strip().startswith('|') and line.strip().endswith('|'):
                    # Check for at least 2 columns
                    header = [c.strip() for c in line.strip().split('|') if c.strip()]
                    if len(header) < 2:
                        continue
                    # Check next line is a separator
                    if i + 1 < len(lines):
                        sep = lines[i + 1].strip()
                        if sep.startswith('|') and sep.endswith('|') and set(sep.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')) == set():
                            in_table = True
                            header_cols = len(header)
                            table_lines.append(line)
                            table_lines.append(lines[i + 1])
                            continue
                elif in_table:
                    # Table rows must match header columns
                    row = [c.strip() for c in line.strip().split('|') if c.strip()]
                    if line.strip().startswith('|') and line.strip().endswith('|') and len(row) == header_cols:
                        table_lines.append(line)
                    else:
                        break
            if len(table_lines) >= 2:
                return '\n'.join(table_lines)
            return None

        def clean_component_name(name):
            # Remove **, extra spaces, and colons at the end
            name = re.sub(r'\\*\\*', '', name)
            name = name.strip()
            name = re.sub(r':$', '', name)
            return name

        def is_valid_component(name):
            # Heuristic: component names are not empty, not just dashes, and not relationship words
            if not name or set(name) <= set('-:'):
                return False
            # Add more rules if needed, e.g., filter out known relationship words
            relationship_words = {'applied to', 'connect', 'form', 'represents', 'evaluates', 'categorizes', 'used within', 'generated by', 'represents', 'forms', 'involved'}
            if name.lower() in relationship_words:
                return False
            return True

        def parse_matrix(llm_output, max_attempts=2):
            sections = {}
            for attempt in range(max_attempts):
                try:
                    # Extract key components
                    key_comp = re.search(r'\*\*Key Components:\*\*([\s\S]+?)(\*\*|$)', llm_output)
                    if key_comp:
                        comps = [line.strip(' .') for line in key_comp.group(1).split('\n') if line.strip() and not line.strip().startswith('**')]
                        sections['components'] = comps
                    
                    # Extract component interactions
                    inter = re.search(r'\*\*Component Interactions:\*\*([\s\S]+?)(\*\*|$)', llm_output)
                    if inter:
                        interactions = [line.strip(' .') for line in inter.group(1).split('\n') if line.strip() and not line.strip().startswith('**')]
                        sections['interactions'] = interactions
                    
                    # Extract and validate matrix (standard way)
                    matrix = re.search(r'\*\*Interaction Matrix:\*\*([\s\S]+?)(\*\*|$)', llm_output)
                    matrix_text = None
                    if matrix:
                        matrix_text = matrix.group(1).strip()
                        sections['matrix'] = matrix_text
                        df = markdown_table_to_df(matrix_text)
                        if not df.empty and len(df) > 0:
                            data = readNodes(df)
                            makeM(data, "interaction_matrix")
                            sections['graph_image'] = "interaction_matrix.png"
                            # st.markdown('<div style="color:#fff;background:#222;padding:10px;border-radius:8px;margin-bottom:10px;overflow-x:auto;">', unsafe_allow_html=True)
                            # st.table(df)
                            # st.markdown('</div>', unsafe_allow_html=True)
                            return sections
                    # Fallback: search for any markdown table in the output
                    if (not matrix_text or df.empty or len(df) == 0):
                        table_text = extract_first_markdown_table(llm_output)
                        if table_text:
                            df_fallback = markdown_table_to_df(table_text)
                            if not df_fallback.empty and len(df_fallback) > 0:
                                sections['matrix'] = table_text
                                data = readNodes(df_fallback)
                                makeM(data, "interaction_matrix")
                                sections['graph_image'] = "interaction_matrix.png"
                                # st.markdown('<div style="color:#fff;background:#222;padding:10px;border-radius:8px;margin-bottom:10px;overflow-x:auto;">', unsafe_allow_html=True)
                                # st.table(df_fallback)
                                # st.markdown('</div>', unsafe_allow_html=True)
                                return sections
                    # Extract claims
                    claims = re.search(r'\*\*Patent Claims:\*\*([\s\S]+)', llm_output)
                    if claims:
                        claims_text = claims.group(1).strip()
                        sections['claims'] = claims_text
                    # If we don't have a valid matrix and haven't reached max attempts, try again
                    if attempt < max_attempts - 1:
                        st.warning(f"Attempt {attempt + 1}/{max_attempts}: No valid matrix found. Trying with different chunk...")
                        valid_chunks = st.session_state['valid_chunks']
                        vectors = np.array(st.session_state['vectors'])
                        used_indices = st.session_state.get('used_chunk_indices', [])
                        available_indices = [i for i in range(len(valid_chunks)) if i not in used_indices]
                        if not available_indices:
                            st.error("No more unique chunks available")
                            return sections
                        chunk_idx = available_indices[attempt % len(available_indices)]
                        used_indices.append(chunk_idx)
                        st.session_state['used_chunk_indices'] = used_indices
                        chunk = valid_chunks[chunk_idx]
                        llm = st.session_state['llm']
                        summaries = st.session_state['summaries']
                        previous_summaries = st.session_state.get('previous_summaries', [])
                        new_summary = create_summaries([chunk], [0], llm)
                        combined_summary_text = f"{summaries.page_content}\n\n"
                        for prev_summary in previous_summaries:
                            combined_summary_text += f"{prev_summary.page_content}\n\n"
                        combined_summary_text += f"{new_summary.page_content}"
                        previous_summaries.append(new_summary)
                        st.session_state['previous_summaries'] = previous_summaries
                        combined_summary = Document(page_content=combined_summary_text)
                        final_output = generate_final_summary(combined_summary, llm)
                        llm_output = final_output
                        continue
                    else:
                        st.error("Failed to generate valid matrix after maximum attempts")
                        return sections
                except Exception as e:
                    st.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        st.error("Failed to generate valid matrix after maximum attempts")
                        return sections
            return sections

        def markdown_table_to_df(md):
            try:
                lines = [l for l in md.split('\n') if l.strip()]
                # Remove the separator row (second line)
                if len(lines) > 2 and all(set(c.strip()) <= set('-:| ') for c in lines[1]):
                    lines.pop(1)
                header = [h.strip() for h in lines[0].split('|') if h.strip()]
                data = []
                for l in lines[1:]:
                    row = [c.strip() for c in l.split('|')[1:-1]]
                    if row and len(row) == len(header):
                        data.append(row)
                df = pd.DataFrame(data, columns=header)
                # Clean the first column (component names)
                df.iloc[:, 0] = df.iloc[:, 0].apply(clean_component_name)
                # Remove rows where the first column is not a valid component
                df = df[df.iloc[:, 0].apply(is_valid_component)]
                # Optionally, set the first column as index if it is 'Component' or similar
                if df.columns[0].lower() in ['component', 'components']:
                    df.set_index(df.columns[0], inplace=True)
                    # Clean index as well
                    df.index = df.index.map(clean_component_name)
                # Clean columns
                df.columns = [clean_component_name(c) for c in df.columns]
                return df
            except Exception as e:
                st.error(f"Error converting markdown to DataFrame: {str(e)}")
                return pd.DataFrame()

        parsed = parse_matrix(st.session_state['final_output'])

        st.markdown("### 📊 Patent Analysis Results")
        st.markdown("#### Component Interaction Matrix")
        # Key Components
        if 'components' in parsed:
            st.markdown('<div style="color:#fff;background:#222;padding:10px;border-radius:8px;margin-bottom:10px;"><b>Key Components:</b><ul>' + ''.join([f'<li>{c}</li>' for c in parsed['components']]) + '</ul></div>', unsafe_allow_html=True)
        # Component Interactions
        if 'interactions' in parsed:
            st.markdown('<div style="color:#fff;background:#222;padding:10px;border-radius:8px;margin-bottom:10px;"><b>Component Interactions:</b><ul>' + ''.join([f'<li>{c}</li>' for c in parsed['interactions']]) + '</ul></div>', unsafe_allow_html=True)
        # Matrix
        if 'matrix' in parsed:
            try:
                df = markdown_table_to_df(parsed['matrix'])
                df.index = df.index.map(clean_component_name)
                st.markdown('<div style="color:#fff;background:#222;padding:10px;border-radius:8px;margin-bottom:10px;overflow-x:auto;">', unsafe_allow_html=True)
                st.table(df)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display graph visualization if available
                if 'graph_image' in parsed and os.path.exists(parsed['graph_image']):
                    st.markdown("### 🔄 Component Interaction Graph")
                    st.image(parsed['graph_image'], use_column_width=True)
            except Exception:
                st.markdown('<div style="color:#fff;background:#222;padding:10px;border-radius:8px;margin-bottom:10px;overflow-x:auto;">' + parsed['matrix'].replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
        # Claims
        if 'claims' in parsed:
            st.markdown('<div style="color:#fff;background:#222;padding:10px;border-radius:8px;margin-bottom:10px;"><b>Patent Claims:</b><br>' + parsed['claims'].replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
        st.markdown("<hr style='margin: 2em 0; border: 1px solid #444;'>", unsafe_allow_html=True)
        chat_anchor = st.empty()

    # --- Ask prompt and chat below the matrix ---
    if 'valid_chunks' in st.session_state and 'vectors' in st.session_state and 'summaries' in st.session_state and 'llm' in st.session_state:
        st.markdown("---")
        st.markdown("### ❓ Ask Questions About the Patent")
        if 'qa_history' not in st.session_state:
            st.session_state['qa_history'] = []
        # Display chat history first
        for entry in st.session_state['qa_history']:
            q, a = entry[:2]
            st.markdown(f'<div style="background:#222;color:#fff;padding:10px;border-radius:8px;margin-bottom:5px;"><b>Q:</b> {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:#222;color:#fff;padding:10px;border-radius:8px;margin-bottom:15px;"><b>A:</b> {a}</div>', unsafe_allow_html=True)
        # Ask prompt always below all chats, use st.form for immediate response
        with st.form(key="qa_form", clear_on_submit=True):
            question = st.text_input("Enter your question about the patent:", key="qa_input")
            submitted = st.form_submit_button("Ask")
            if submitted and question:
                with st.spinner("Generating answer..."):
                    valid_chunks = st.session_state['valid_chunks']
                    vectors = np.array(st.session_state['vectors'])
                    llm = st.session_state['llm']
                    summaries = st.session_state['summaries']
                    q_emb = get_embeddings([question])[1][0]
                    sims = np.dot(vectors, q_emb) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_emb) + 1e-8)
                    top_idx = int(np.argmax(sims))
                    best_chunk = valid_chunks[top_idx]
                    best_chunk_clean = re.sub(r'"base64":\s*"[A-Za-z0-9+/=]+"', '[IMAGE OMITTED]', best_chunk)
                    # Merge with previous summary
                    merged_text = f"Summary:\n{summaries.page_content}\n\nRelevant Chunk:\n{best_chunk_clean}"
                    # Summarize the merged text
                    summarization_prompt = f"""
                    Summarize the following information, focusing on the most relevant details for answering a user question. Be concise and clear.
                    
                    {merged_text}
                    """
                    summary_result = llm.invoke(summarization_prompt)
                    merged_summary = summary_result.content if hasattr(summary_result, 'content') else str(summary_result)
                    # Use the new summary as context for the answer
                    qa_prompt = f"""
                    You are a patent assistant. Use the following summary to answer the user's question. Be concise and specific.
                    
                    Summary:
                    {merged_summary}
                    
                    Question: {question}
                    """
                    result = llm.invoke(qa_prompt)
                    answer = result.content if hasattr(result, 'content') else str(result)
                    st.session_state['qa_history'].append((question, answer))
                chat_anchor.markdown("<a name='chat_area'></a>", unsafe_allow_html=True)
                st.rerun()

if __name__ == "__main__":
    main() 