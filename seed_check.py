from vector_store.guidelines_store import GuidelinesStore

store = GuidelinesStore()
# Verify query returns data
rules = store.query_guidelines("Java Spring Boot best practices", top_k=1)
print("Rules found:", len(rules))