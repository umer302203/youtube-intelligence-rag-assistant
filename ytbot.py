# Import necessary libraries for the YouTube bot
import gradio as gr
import re  # For extracting video id
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For chunking
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes  
from ibm_watsonx_ai import APIClient, Credentials  
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain.chains import LLMChain  
from langchain.prompts import PromptTemplate  

# ==========================================
# 1. CORE TRANSCRIPT PROCESSING FUNCTIONS
# ==========================================

def get_video_id(url):    
    if not url:
        return None
    # Enhanced Regex: Robust matching for standard, short, embed, shorts & removing tracking tokens
    pattern = r'(?:v=|\/([0-9A-Za-z_-]{11})(?:\?|&|$)|shorts\/|youtu\.be\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url.strip())
    if match:
        return match.group(2) if match.group(2) else match.group(1)
    return None

def get_transcript(url):
    video_id = get_video_id(url)
    print(f"\n[DEBUG] Input URL: '{url}'")
    print(f"[DEBUG] Extracted Video ID: '{video_id}'")
    
    if not video_id:
        print("[DEBUG ERROR] Invalid Video ID extracted.")
        return None
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcripts = ytt_api.list(video_id)
        
        # Priority 1: Search for English ('en') transcript first
        for t in transcripts:
            if 'en' in t.language_code.lower():
                print(f"[DEBUG SUCCESS] Found English transcript: '{t.language_code}'")
                return t.fetch()
        
        # Priority 2: Multi-Language Fallback (Hindi, Urdu, etc.)
        for t in transcripts:
            print(f"[DEBUG FALLBACK] Fetching non-English transcript: '{t.language_code}'")
            return t.fetch()
            
        return None
    except Exception as e:
        print(f"[DEBUG EXCEPTION] YouTube API Error: {e}")
        return None

def process(transcript):
    if not transcript:
        return ""
    txt = ""
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except (KeyError, AttributeError):
            try:
                txt += f"Text: {i['text']} Start: {i['start']}\n"
            except Exception:
                pass
    return txt

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(processed_transcript)

# ==========================================
# 2. WATSONX & EMBEDDING SETUP
# ==========================================

def setup_credentials():
    # LLM Model updated to officially supported Mistral Model
    model_id = "mistralai/mistral-small-3-1-24b-instruct-2503"
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
    client = APIClient(credentials)
    project_id = "skills-network"
    return model_id, credentials, client, project_id

def define_parameters():
    return {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 900,
    }

def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    return WatsonxLLM(
        model_id=model_id,
        url=credentials.get("url"),
        project_id=project_id,
        params=parameters
    )

def setup_embedding_model(credentials, project_id):
    # Embedding Model updated to officially supported Granite Multilingual Model
    return WatsonxEmbeddings(
        model_id='ibm/granite-embedding-278m-multilingual',
        url=credentials["url"],
        project_id=project_id
    )

def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

def retrieve(query, faiss_index, k=7):
    return faiss_index.similarity_search(query, k=k)

# ==========================================
# 3. PROMPT & CHAIN INITIALIZATION
# ==========================================

def create_summary_prompt():
    # Clean generic prompt for universal compatibility
    template = """You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Ignore any timestamps in your summary.
3. Focus on the spoken content (Text) of the video.

Transcript to summarize:
{transcript}

Summary:"""
    return PromptTemplate(input_variables=["transcript"], template=template)

def create_summary_chain(llm, prompt, verbose=True):
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)

def create_qa_prompt_template():
    qa_template = """You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
1. Precise and free from repetition
2. Consistent with the information provided in the video
3. Well-organized and easy to understand
4. Focused on addressing the user's question directly

Relevant Video Context:
{context}

Question: {question}

Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=qa_template)

def create_qa_chain(llm, prompt_template, verbose=True):
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)

def generate_answer(question, faiss_index, qa_chain, k=7):
    relevant_context = retrieve(question, faiss_index, k=k)
    # Convert FAISS Document objects to plain text string to avoid warning issues
    context_str = "\n".join([doc.page_content for doc in relevant_context])
    
    # Modern .invoke() method
    response = qa_chain.invoke({"context": context_str, "question": question})
    return response["text"] if isinstance(response, dict) else response

# ==========================================
# 4. GRADIO APP STATE & LOGIC FUNCTIONS
# ==========================================

fetched_transcript = None
processed_transcript = ""

def summarize_video(video_url):
    global fetched_transcript, processed_transcript
    
    if not video_url or len(video_url.strip()) == 0:
        return "❌ Please provide a valid YouTube URL."
    
    fetched_transcript = get_transcript(video_url)
    if not fetched_transcript:
        return "❌ Unable to fetch transcript. Check if the URL is correct or if subtitles exist for this video."

    processed_transcript = process(fetched_transcript)

    if processed_transcript:
        model_id, credentials, client, project_id = setup_credentials()
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)
        
        # Modern .invoke() method
        response = summary_chain.invoke({"transcript": processed_transcript})
        summary = response["text"] if isinstance(response, dict) else response
        return summary
    else:
        return "❌ Transcript is empty or could not be processed."

def answer_question(video_url, user_question):
    global fetched_transcript, processed_transcript

    if not user_question or len(user_question.strip()) == 0:
        return "❌ Please enter a question to ask."

    if not processed_transcript:
        if video_url and len(video_url.strip()) > 0:
            fetched_transcript = get_transcript(video_url)
            if not fetched_transcript:
                return "❌ Unable to fetch transcript for this URL."
            processed_transcript = process(fetched_transcript)
        else:
            return "❌ Please provide a valid YouTube URL first."

    if processed_transcript:
        chunks = chunk_transcript(processed_transcript)
        model_id, credentials, client, project_id = setup_credentials()
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
        
        # Setup updated embedding model
        embedding_model = setup_embedding_model(credentials, project_id)
        faiss_index = create_faiss_index(chunks, embedding_model)
        
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)
        
        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "❌ No processed transcript available to answer questions."

# ==========================================
# 5. BEAUTIFIED GRADIO INTERFACE (RED THEME)
# ==========================================

custom_css = """
.container { max-width: 900px; margin: auto; padding-top: 20px; }
.title-header { text-align: center; color: #DC2626; margin-bottom: 20px; }
.btn-primary { background: linear-gradient(90deg, #DC2626 0%, #991B1B 100%) !important; color: white !important; border: none !important; font-weight: bold !important; }
"""

with gr.Blocks() as interface:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            # 🎬 YouTube Intelligence & RAG Assistant
            ### Extract Summaries and Ask Questions directly from any YouTube Video
            ---
            """
        )

        with gr.Row():
            video_url = gr.Textbox(
                label="📺 YouTube Video URL", 
                placeholder="Paste link here (e.g., https://www.youtube.com/watch?v=... or https://youtu.be/...)", 
                scale=4
            )

        with gr.Tabs():
            with gr.TabItem("📝 Video Summarizer"):
                summarize_btn = gr.Button("Generate Video Summary", elem_classes="btn-primary")
                summary_output = gr.Textbox(label="AI Summary", lines=6, placeholder="Summary will appear here...")
                summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)

            with gr.TabItem("❓ Video Q&A (RAG System)"):
                question_input = gr.Textbox(
                    label="Ask a Question", 
                    placeholder="e.g., What did the speaker say about machine learning?"
                )
                question_btn = gr.Button("Get Answer from Video", elem_classes="btn-primary")
                answer_output = gr.Textbox(label="AI Answer", lines=6, placeholder="Answer will appear here...")
                question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

# Launch with theme & css parameters inside launch()
interface.launch(
    server_name="0.0.0.0", 
    server_port=7860, 
    share=True,
    theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"),
    css=custom_css
)