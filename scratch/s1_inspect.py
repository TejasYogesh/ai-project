# scratch/s1_inspect.py
import feedparser, httpx

r = httpx.get("https://export.arxiv.org/api/query", params={"id_list": "1706.03762"})
entry = feedparser.parse(r.text).entries[0]

print("keys:", list(entry.keys()))
print("id:", entry.id)
print("title:", repr(entry.title))
print("authors:", entry.authors[:3])
print("published:", entry.published)
print("links:", entry.links)
print("tags:", entry.tags)