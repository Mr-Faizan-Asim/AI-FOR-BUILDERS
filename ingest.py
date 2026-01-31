import os
import sys
from dotenv import load_dotenv

# Load .env file if present (create one with your API keys!)
load_dotenv()

# ================== CONFIG ==================
# SECURITY: Set these in environment variables instead!
# Example .env file:
# GEMINI_API_KEY=your_key_here
# PINECONE_API_KEY=your_key_here

# ============================================

# Check API keys
if not GEMINI_API_KEY or not PINECONE_API_KEY:
    print("❌ Missing API keys! Set them in .env file or environment variables.")
    sys.exit(1)

try:
    from pinecone import Pinecone
    from google import genai
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install pinecone google-genai python-dotenv")
    sys.exit(1)

# Initialize clients
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"❌ Pinecone connection failed: {e}")
    sys.exit(1)

genai_client = genai.Client(api_key=GEMINI_API_KEY)

def embed_text(text):
    """Generate embeddings using Google GenAI"""
    try:
        response = genai_client.models.embed_content(
            model="text-embedding-004",  # Current model name for google-genai
            contents=[text]
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"⚠️  Embedding error: {e}")
        raise

# Process knowledge base
knowledge_folder = "knowledge"
doc_id = 0

if not os.path.exists(knowledge_folder):
    print(f"❌ Folder '{knowledge_folder}' not found!")
    sys.exit(1)

print(f"📁 Processing markdown files from '{knowledge_folder}'...")

for root, dirs, files in os.walk(knowledge_folder):
    for file in files:
        if file.endswith(".md"):
            path = os.path.join(root, file)
            
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Smarter chunking: split by paragraphs first, then by size
                paragraphs = content.split('\n\n')
                chunks = []
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk) + len(para) < 500:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Fallback to fixed size if no paragraph breaks
                if not chunks:
                    chunks = [content[i:i+500] for i in range(0, len(content), 450)]

                print(f"📄 {file}: {len(chunks)} chunks")
                
                for chunk in chunks:
                    if not chunk.strip() or len(chunk) < 10:
                        continue
                    
                    vector = embed_text(chunk)
                    
                    index.upsert(
                        vectors=[{
                            "id": f"doc-{doc_id}",
                            "values": vector,
                            "metadata": {
                                "source_file": file,
                                "folder": os.path.basename(root),
                                "text": chunk[:1500]  # Pinecone metadata limit
                            }
                        }]
                    )
                    doc_id += 1
                    
                print(f"   ✅ Upserted {len(chunks)} chunks")
                        
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")

print(f"\n✅ Knowledge base ingested! Total chunks: {doc_id}")