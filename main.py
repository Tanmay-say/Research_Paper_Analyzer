import logging
import sys
from config import Config
from flask import Flask, request, jsonify, render_template, session
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import create_react_agent, Tool, AgentExecutor
# Removed langtrace import
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
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory
    )

    try:
        # Invoke the agent with the question
        res = agent_executor.invoke({'input':question})

        logger.info(f"User asked: {question}")
        return jsonify({'answer':res['output']})  # Return the answer
    except Exception as e:
        logger.error(f"Error during question processing: {str(e)}")
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
    
    def enhanced_retriever(query: str):
        """Enhanced retriever that uses both local memory and vectorstore."""
        try:
            # First try local memory for fast retrieval
            local_results = search_pdf_content(query)
            
            # Also get vectorstore results
            vector_results = vectorstore.as_retriever(search_kwargs={"k": 4}).get_relevant_documents(query)
            
            # Combine results
            combined_results = []
            
            # Add local memory results
            for result in local_results:
                combined_results.append(f"From {result['filename']}: {result['content'][:500]}...")
            
            # Add vectorstore results
            for doc in vector_results:
                combined_results.append(f"Vector search result: {doc.page_content[:500]}...")
            
            if combined_results:
                return combined_results
            else:
                return ["No relevant content found in uploaded documents."]
                
        except Exception as e:
            logger.error(f"Error in enhanced retriever: {e}")
            return [f"Error retrieving content: {str(e)}"]
    
    def pdf_summarizer(filename: str):
        """Tool to get PDF summaries."""
        try:
            return get_pdf_summary(filename)
        except Exception as e:
            return f"Error getting summary: {str(e)}"
    
    def list_uploaded_pdfs():
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
        temperature=0
    )
    base_prompt = hub.pull("hwchase17/react")
    template = """
        You are an intelligent Research Paper AI Assistant designed to help users analyze and understand research papers. Your primary goal is to answer user queries accurately and efficiently using the tools available to you. You have access to:

        **Available Tools:**
        1. **Research_Paper_Retriever**: Searches through uploaded PDF documents for relevant content. Use this for questions about specific topics, methodologies, results, or any aspect of the research papers.

        2. **PDF_Summarizer**: Provides summaries of specific uploaded PDF files. Use this when users ask for summaries of particular documents.

        3. **List_Uploaded_PDFs**: Lists all uploaded PDF documents. Use this when users want to know what papers are available.

        4. **Search** (if available): Provides web search results for additional context.

        **Your Strategy for PDF Analysis:**
        1. **For general questions about uploaded papers**: Use Research_Paper_Retriever to find relevant content
        2. **For specific PDF summaries**: Use PDF_Summarizer with the filename
        3. **For listing available papers**: Use List_Uploaded_PDFs
        4. **For questions not covered by uploaded papers**: Use Search tool if available

        **Important Guidelines:**
        - Always check if PDFs are uploaded before attempting analysis
        - Provide specific, detailed answers based on the actual content of the papers
        - If no relevant content is found, clearly state this
        - When summarizing, focus on key findings, methodology, and conclusions
        - Be helpful and educational in your responses

        **Example Workflow:**
        User: "Analyze the uploaded PDF and summarize it"
        1. Use List_Uploaded_PDFs to see available documents
        2. Use PDF_Summarizer to get a summary of the document
        3. Use Research_Paper_Retriever for specific analysis questions
        4. Provide a comprehensive response based on the actual content

        Remember: You can only analyze PDFs that have been uploaded to the system. If no PDFs are uploaded, inform the user to upload documents first.
        """
        
    # Create the prompt with instructions
    prompt = base_prompt.partial(instructions=template)
    
    # Prepare tools list (include all enhanced tools)
    tools = [retriever_tool, summary_tool, list_tool]
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