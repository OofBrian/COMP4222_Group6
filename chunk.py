from typing import List

def split_into_chunks(doc_file: str) -> List[str]:
    return [chunk for chunk in doc_file.split("\n\n")]
'''
def split_into_chunks(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
'''