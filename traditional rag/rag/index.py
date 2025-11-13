from chunk import split_into_chunks
from embedding import embed_chunk
from vectordatabase import save_embeddings

def build_index():
    chunks = split_into_chunks("doc.md")
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    save_embeddings(chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks.")

if __name__ == "__main__":
    build_index()