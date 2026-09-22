# scratch/s1_arxiv.py
from backend.services.arxiv_client import search, extract_arxiv_id

print("--- ID lookup ---")
for p in search(id_list="1706.03762"):
    print(p.model_dump_json(indent=2))

print("\n--- Topic search ---")
for p in search(query="kv cache compression", max_results=5):
    print(p.arxiv_id, "|", p.published, "|", p.title)

print("\n--- ID extraction ---")
for s in ["2401.12345", "2401.12345v2", "https://arxiv.org/abs/2401.12345", "no id here"]:
    print(repr(s), "->", extract_arxiv_id(s))

print("\n--- Failure cases ---")
print("Nonsense topic:", search(query="asdkjhqwezzz"))
print("Invalid ID:", search(id_list="9999.99999"))