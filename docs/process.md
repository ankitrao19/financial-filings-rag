### Vectorization
building vectors from corpus we have 2 apporches --> sentence transformers or use openai sdk and use models from it
what is a sentence transformer:
a transformer architecture that converts sentenses into dense numeric vectors(embeddings)
sentence transformer uses pooling method to compress a full sentence into a single fixed size vector --> do research on this
popular sentence transformers:
all-MiniLM-L6-v2
all-mpnet-base-v2 --> high accuracy, dedicated gpu, english only supported, ideal for deep semantic search
paraphrase-multilingual-MiniLM-L12-v2 


### Storing vectors
we will be using chromadb
we have 2 clients mainly to store evctors:
PersistentClient --> this lets us store data into a dedicated folder inside my hard drive and is on slower end
EphemeralClient --> this lets us store data into ram which can be faster to access and once we close our terminal or our service goes down we don't have anything at hand we need to recreate the collection


### Fetching corpus (build_chunks.py --fetch)
sec edgar needs a real name + email as identity, else requests get blocked
filing.markdown() keeps headings/tables intact, filing.text() is only the fallback
sleep between requests to stay under sec.gov rate limit (<10 req/s)
wrap each ticker/form/filing in its own try so one bad filing doesn't kill the whole run

### Chunking
fixed size chunks (1200 chars ~300 tokens) with 200 char overlap so facts on chunk borders aren't cut in half
drop tiny tail fragments (<50 chars) since they add noise but no meaning
every chunk carries full metadata (ticker, form, filing_date, accession, chunk_index) --> needed later for filtering + scoring

### Indexing into chromadb (build_chunks.py --index)
id = accession + chunk_index --> unique and stable per chunk across reruns
encode in batches (64) for embeddings and add to chroma in batches (500) to avoid memory/request limits
use upsert not add --> reruns are idempotent instead of erroring on duplicate ids
--reset drops the collection first, needed when chunking/embedding model changes (old vectors become incompatible)

### Retrieval (evaluate_rag.py)
query must be embedded with the SAME model used at index time or distances are meaningless
metadata filter (where={"ticker": ...}) narrows search space --> big boost when question is about one company
top_k is a tradeoff: higher k = better hit rate but more noise + tokens in the llm context

### Scoring retrieval
hit rate@k --> did the tagged filing show up in top k at all
MRR (mean reciprocal rank) --> rewards right filing appearing higher up (1/rank)
filing-level hit is only an upper bound, right filing can still be the wrong section
check if the ground truth filing is even indexed, else accuracy blame lands on model unfairly

### Generation
prompt says answer ONLY from context + say "don't know" otherwise --> reduces hallucination
temperature=0 for repeatable answers during evaluation
ask the model to cite ticker + filing date so answers are traceable

### Grading answers
llm-as-judge with fixed verdicts (correct / partial / incorrect / refused) handles unit rewrites like $61,761M == $61.8B
response_format json_object + validate verdict, fall back to "unparsed" instead of crashing
numeric recall is a provider-free sanity check on the judge, strip years so they don't inflate matches
refusal phrase matching catches "i don't know" answers separately from wrong answers

### Metrics / aggregation
strict success (correct only) vs lenient success (correct + partial) tells how close the misses are
break down by difficulty and ticker to find where the pipeline actually fails
ground truth health (missing answers, unverified, not indexed) --> eval set itself can be the bug
write results after every question so a crash midway doesn't lose progress
--no-filter / --k flags let you measure the effect of one change at a time
