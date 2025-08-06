import logging
import sys
import requests
import json
from config import Config
from flask import Flask, request, jsonify, render_template, session
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from data_ingestion import validate_file, get_vectorstore, process_file
import uuid
from langchain import hub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize environment variables
groq_api_key = Config.GROQ_API_KEY
app.secret_key = Config.FLASK_SECRET_KEY
serp_api_key = Config.SERPAPI_API_KEY
COOKIE_SIZE_LIMIT = 4093

# Initialize vectorstore and user memories
vectorstore = None
user_memories = {}  

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the service is running."""
    return jsonify(status='healthy'), 200  # Return a healthy status

@app.route('/')
def home():
    """Render the homepage for the chatbot application."""
    return render_template('index.html')  # Render the main HTML template

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads for processing and validation.
    
    Accepts multiple files, validates them, processes them, and returns
    a response with the processed files and any errors encountered.
    """
    files = request.files.getlist('files')  # Handle multiple files
    
    if not files:
        logger.error("No files provided for bulk upload.")   # Log error if no files
        return jsonify({"error": "No files provided"}), 400

    processed_files = []  # List to store successfully processed files
    errors = []  # List to store any errors encountered

    try:
        for file in files:
            # Validate the file
            filename, error = validate_file(file)
            if error:
                logger.error(error)
                errors.append({"file": filename, "error": error})
                continue

            # Process the file
            filename, error = process_file(file, vectorstore)
            if error:
                errors.append({"file": filename, "error": error})
                continue

            processed_files.append(filename)

        # Prepare response
        response = {"processed_files": processed_files}
        if errors:
            response["errors"] = errors
            logger.warning(f"Bulk upload completed with errors: {errors}")
        
        return jsonify(response), 200 if not errors else 207

    except Exception as e:
        logger.error(f"Error during bulk upload: {str(e)}")
        return jsonify({"error": f"{str(e)}"}), 500

def manage_session():
    """Check and manage the session data.
    
    Ensures that a unique user ID is created for each session and
    initializes a memory buffer for storing conversation history.
    """
    # Ensure the user ID exists in the session
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Generate a unique user ID
        user_memories[session['user_id']] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True  # Initialize memory for user
        )
    else:
        # Retrieve the memory associated with the user ID
        user_id = session['user_id']
        if user_id not in user_memories:
            user_memories[user_id] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True  # Initialize memory if not present
            )

@app.route('/ask', methods=['POST'])
def ask():
    """Endpoint to handle user questions.
    
    Manages user sessions, retrieves the user's question from the request,
    and invokes the agent executor to get an answer. Returns the answer
    or an error message if something goes wrong.
    """
    manage_session()
    data = request.json
    question = data.get("question")
    
    # Get user's memory
    memory = user_memories[session['user_id']]
    
    try:
        # Use intelligent agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            memory=memory
        )
        
        # Invoke the agent with the question
        res = agent_executor.invoke({'input': question})
        
        logger.info(f"User asked: {question}")
        return jsonify({'answer': res['output']})
        
    except Exception as e:
        logger.error(f"Error during question processing: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route('/debug', methods=['GET'])
def debug_endpoint():
    """Debug endpoint to show what PDFs are stored."""
    try:
        documents = local_memory.list_documents()
        document_details = {}
        
        for doc in documents:
            metadata = local_memory.get_document(doc)
            chunks = local_memory.get_chunks(doc)
            document_details[doc] = {
                'metadata': metadata,
                'chunk_count': len(chunks),
                'first_chunk_preview': chunks[0][:200] + "..." if chunks else "No content"
            }
        
        return jsonify({
            'total_documents': len(documents),
            'documents': documents,
            'document_details': document_details
        })
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    """Main entry point for the application.
    
    Initializes the necessary components, including embeddings, vectorstore,
    tools, and the agent, and starts the Flask application.
    """

    # Langtrace removed for simplified setup

    # Initialize embeddings and vectorstore using Groq
    from data_ingestion import GroqEmbeddings
    embeddings = GroqEmbeddings(groq_api_key)
    vectorstore = get_vectorstore(embeddings)
    logger.info("Initialized Groq-based embeddings and ChromaDB vectorstore")

    # Import enhanced functions
    from data_ingestion import local_memory, get_pdf_summary, search_pdf_content
    
    def search_arxiv_papers(query: str, max_results: int = 5):
        """Search for research papers on ArXiv."""
        try:
            # ArXiv API endpoint
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:"{query}"',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('.//{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('.//{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('.//{http://www.w3.org/2005/Atom}summary').text.strip()
                authors = [author.find('.//{http://www.w3.org/2005/Atom}name').text for author in entry.findall('.//{http://www.w3.org/2005/Atom}author')]
                published = entry.find('.//{http://www.w3.org/2005/Atom}published').text
                pdf_url = entry.find('.//{http://www.w3.org/2005/Atom}link[@title="pdf"]').get('href')
                
                papers.append({
                    'title': title,
                    'summary': summary,
                    'authors': authors,
                    'published': published,
                    'pdf_url': pdf_url
                })
            
            if papers:
                result = f"Found {len(papers)} papers on ArXiv for '{query}':\n\n"
                for i, paper in enumerate(papers, 1):
                    result += f"{i}. **{paper['title']}**\n"
                    result += f"   Authors: {', '.join(paper['authors'])}\n"
                    result += f"   Published: {paper['published'][:10]}\n"
                    result += f"   Summary: {paper['summary'][:200]}...\n"
                    result += f"   PDF: {paper['pdf_url']}\n\n"
                return result
            else:
                return f"No papers found on ArXiv for '{query}'"
                
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return f"Error searching ArXiv: {str(e)}"
    

    
    def enhanced_retriever(query: str):
        """Enhanced retriever that uses both local memory and vectorstore for intelligent content retrieval."""
        try:
            # First check what documents are available
            available_docs = local_memory.list_documents()
            if not available_docs:
                return ["No PDFs have been uploaded yet. Please upload PDF files first."]
            
            # Get local memory results for fast retrieval
            local_results = search_pdf_content(query)
            
            # Get vectorstore results for semantic search
            vector_results = vectorstore.as_retriever(search_kwargs={"k": 6}).get_relevant_documents(query)
            
            # Combine and prioritize results intelligently
            combined_results = []
            
            # Add local memory results with context
            for result in local_results:
                filename = result['filename']
                content = result['content']
                chunk_index = result['chunk_index']
                
                # Provide more context about where this content comes from
                combined_results.append(f"📄 **{filename}** (Section {chunk_index + 1}):\n{content[:600]}...")
            
            # Add vectorstore results with source information
            for i, doc in enumerate(vector_results):
                # Try to extract source information from metadata
                source = doc.metadata.get('source', 'Unknown source') if hasattr(doc, 'metadata') else 'Vector search'
                combined_results.append(f"🔍 **Semantic Match {i+1}** ({source}):\n{doc.page_content[:600]}...")
            
            if combined_results:
                # Add a summary of what was found
                summary = f"Found {len(combined_results)} relevant sections across {len(set([r['filename'] for r in local_results]))} document(s)."
                combined_results.insert(0, summary)
                return combined_results
            else:
                # Provide helpful guidance when no content is found
                return [
                    f"❌ No relevant content found for '{query}' in the uploaded documents.",
                    f"📚 Available documents: {', '.join(available_docs)}",
                    "💡 Try rephrasing your question or ask about specific aspects of the papers."
                ]
                
        except Exception as e:
            logger.error(f"Error in enhanced retriever: {e}")
            return [f"Error retrieving content: {str(e)}"]
    
    def pdf_summarizer(filename: str):
        """Intelligent tool to get comprehensive PDF summaries."""
        try:
            # Get the basic summary
            basic_summary = get_pdf_summary(filename)
            
            # Get additional context from the document
            metadata = local_memory.get_document(filename)
            chunks = local_memory.get_chunks(filename)
            
            if not chunks:
                return f"❌ No content found for {filename}"
            
            # Create a more comprehensive summary
            summary_parts = []
            summary_parts.append(f"📄 **{filename}** - Comprehensive Summary")
            summary_parts.append("=" * 50)
            
            # Add metadata information
            if metadata:
                summary_parts.append(f"📊 **Document Info:**")
                summary_parts.append(f"   • Upload Time: {metadata.get('upload_time', 'Unknown')}")
                summary_parts.append(f"   • File Size: {metadata.get('file_size', 'Unknown')} bytes")
                summary_parts.append(f"   • Total Sections: {metadata.get('chunk_count', len(chunks))}")
                summary_parts.append("")
            
            # Add the main summary
            summary_parts.append("📝 **Content Summary:**")
            summary_parts.append(basic_summary)
            summary_parts.append("")
            
            # Add key sections preview
            summary_parts.append("🔍 **Key Sections Preview:**")
            for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                preview = chunk[:150].replace('\n', ' ').strip()
                summary_parts.append(f"   {i+1}. {preview}...")
            
            if len(chunks) > 5:
                summary_parts.append(f"   ... and {len(chunks) - 5} more sections")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            return f"Error getting summary: {str(e)}"
    
    def list_uploaded_pdfs(query=""):
        """Tool to list all uploaded PDFs."""
        try:
            documents = local_memory.list_documents()
            if documents:
                return f"Uploaded PDFs: {', '.join(documents)}"
            else:
                return "No PDFs have been uploaded yet."
        except Exception as e:
            return f"Error listing documents: {str(e)}"
    
    # Create enhanced tools
    retriever_tool = Tool(
        name="Research_Paper_Retriever",
        func=enhanced_retriever,
        description="Use this tool to search and retrieve relevant research paper content from uploaded documents. Provide a clear query about the research topic, methodology, results, or any specific aspect you want to find in the papers."
    )
    
    summary_tool = Tool(
        name="PDF_Summarizer",
        func=pdf_summarizer,
        description="Use this tool to get a summary of a specific uploaded PDF. Provide the filename of the PDF you want to summarize."
    )
    
    list_tool = Tool(
        name="List_Uploaded_PDFs",
        func=list_uploaded_pdfs,
        description="Use this tool to list all uploaded PDF documents. No input needed."
    )
    
    arxiv_tool = Tool(
        name="ArXiv_Search",
        func=search_arxiv_papers,
        description="Search for research papers on ArXiv. Use this to find recent papers on any research topic. Provide a clear search query."
    )
    


    # Initialize the search tool (optional)
    serpapi_tool = None
    if serp_api_key and serp_api_key != "your_serpapi_api_key_here":
        try:
            search = SerpAPIWrapper()
            serpapi_tool = Tool(
                name="Search",
                description="A search engine. Use this to answer questions about current events. Input should be a search query.",
                func=search.run
            )
            logger.info("SerpAPI search tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SerpAPI: {e}")
    else:
        logger.info("SerpAPI not configured - web search disabled")

    # Initialize chat components with Groq
    chat = ChatGroq(
        api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Updated to your preferred model
        temperature=0.1  # Slightly higher temperature for more creative responses
    )
    base_prompt = hub.pull("hwchase17/react")
    template = """
        You are an exceptionally intelligent Research Paper AI Assistant with advanced reasoning capabilities, similar to Cursor IDE's intelligent code analysis. You excel at deep understanding, critical analysis, and providing insightful, comprehensive responses.

        **Core Intelligence Principles:**
        - **Contextual Understanding**: Always understand the full context before responding
        - **Multi-layered Analysis**: Look beyond surface-level information to extract deeper insights
        - **Intelligent Tool Selection**: Choose the most appropriate tools based on the user's intent
        - **Proactive Problem Solving**: Anticipate user needs and provide comprehensive solutions
        - **Clear Communication**: Express complex ideas in an accessible, well-structured manner

        **Available Tools:**
        1. **Research_Paper_Retriever**: Searches through uploaded PDF documents for relevant content. Use this for questions about specific topics, methodologies, results, or any aspect of the research papers.

        2. **PDF_Summarizer**: Provides summaries of specific uploaded PDF files. Use this when users ask for summaries of particular documents.

        3. **List_Uploaded_PDFs**: Lists all uploaded PDF documents. Use this when users want to know what papers are available.

        4. **ArXiv_Search**: Search for research papers on ArXiv. Use this to find recent papers on any research topic.

        5. **Search** (if available): Provides web search results for additional context.

        **Intelligent Analysis Strategy:**
        1. **Context Assessment**: First, understand what documents are available and the user's specific request
        2. **Tool Orchestration**: Intelligently combine multiple tools to provide comprehensive answers
        3. **Deep Analysis**: Go beyond simple retrieval to provide insights, connections, and implications
        4. **Proactive Enhancement**: Suggest related questions or areas for further exploration

        **Response Quality Standards:**
        - **Comprehensive**: Cover all aspects of the user's question thoroughly
        - **Insightful**: Provide analysis that goes beyond what's explicitly stated
        - **Well-structured**: Organize information logically with clear sections
        - **Actionable**: Include practical implications and next steps where relevant
        - **Educational**: Help users understand the broader context and significance

        **Smart Workflow Examples:**
        
        **For Summarization Requests:**
        1. Check available documents
        2. Get comprehensive summary using PDF_Summarizer
        3. Use Research_Paper_Retriever to find key sections
        4. Provide structured summary with: Overview, Key Findings, Methodology, Implications, Future Directions

        **For Analysis Requests:**
        1. Identify relevant documents
        2. Retrieve specific content sections
        3. Cross-reference with ArXiv for broader context
        4. Provide critical analysis with: Strengths, Limitations, Novel Contributions, Practical Applications

        **For Comparison Requests:**
        1. Analyze each document individually
        2. Identify common themes and differences
        3. Provide comparative analysis with: Similarities, Differences, Complementary Aspects, Synthesis

        **Critical Guidelines:**
        - **Always verify document availability** before attempting analysis
        - **Provide evidence-based responses** with specific references to document content
        - **Offer multiple perspectives** when analyzing complex topics
        - **Highlight novel contributions** and their significance
        - **Suggest follow-up questions** to deepen understanding
        - **Maintain academic rigor** while being accessible

        **Error Handling:**
        - If no PDFs are uploaded: Clearly explain the requirement and guide the user
        - If content is not found: Suggest alternative approaches or related topics
        - If tools fail: Provide helpful guidance and alternative solutions

        Remember: You are an intelligent research assistant. Think like a senior researcher who can see connections, implications, and opportunities that others might miss. Provide responses that demonstrate deep understanding and offer genuine value to the user's research needs.
        """
        
    # Create the prompt with instructions
    prompt = base_prompt.partial(instructions=template)
    
    # Prepare tools list (include all enhanced tools)
    tools = [retriever_tool, summary_tool, list_tool, arxiv_tool]
    if serpapi_tool:
        tools.append(serpapi_tool)
    
    # Initialize the agent
    agent = create_react_agent(
        tools=tools,
        llm=chat,
        prompt=prompt,
    )

    logger.info("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=8000)  # Start the Flask app