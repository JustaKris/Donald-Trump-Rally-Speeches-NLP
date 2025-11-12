# AI-Powered Topic Analysis

This guide explains how to use the AI-powered topic analysis feature with semantic clustering and LLM-generated insights.

## Overview

The topic analysis system goes beyond simple word frequency by:

1. **Semantic Clustering** — Groups related keywords using embeddings
2. **AI-Generated Labels** — Creates meaningful topic names using Gemini
3. **Contextual Snippets** — Shows keywords in actual use with highlighting
4. **AI Summaries** — Provides interpretive analysis of main themes
5. **Smart Filtering** — Excludes common verbs and weak clusters

## Basic vs Enhanced Topic Extraction

### Legacy Approach (Removed)

The old frequency-based extraction simply listed keywords by count.

### Current: AI-Powered Topic Analysis (`/analyze/topics`)

Returns semantically clustered topics with context:

```json
{
  "clustered_topics": [
    {
      "label": "National Pride",
      "keywords": [
        {"word": "great", "count": 40, "relevance": 1.0},
        {"word": "country", "count": 35, "relevance": 0.875}
      ],
      "avg_relevance": 0.9375,
      "total_mentions": 75
    }
  ],
  "snippets": [
    {
      "label": "National Pride",
      "snippets": [
        "We're going to make America **great** again. This **country** deserves better.",
        "Our **country** is the greatest nation on Earth..."
      ]
    }
  ],
  "summary": "The speech emphasizes themes of national pride and American exceptionalism, with recurring references to restoring greatness...",
  "metadata": {
    "total_keywords": 30,
    "num_clusters": 5,
    "has_ai_summary": true
  }
}
```

## API Usage

### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze/topics" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "They're burning Minneapolis. You don't think of Minneapolis that way, right? You don't think of it. The city is burning down. You have this fake CNN reporter, what's his name? Nobody, the nice shaved head. Maybe I should try that! By that! No, I don't think... Donald Trump went down substantially in the polls, like about 40%. He showed up with a new haircut. It's called the shave head. Ah, hello. And remember he said, No, this is a friendly protest. It's a mostly genteel. And it's really quite nice. Now people are shooting bullets at him. He's being hit with tear gas. This is a friendly protest."
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze/topics",
    json={"text": "Your text here..."}
)

result = response.json()

# Access clustered topics
for cluster in result["clustered_topics"]:
    print(f"\n{cluster['label']} ({cluster['total_mentions']} mentions)")
    print(f"Keywords: {', '.join([kw['word'] for kw in cluster['keywords'][:3]])}")

# Access snippets
for snippet_group in result["snippets"]:
    print(f"\n{snippet_group['label']} examples:")
    for snippet in snippet_group["snippets"]:
        print(f"  - {snippet}")

# Access AI summary
if result["summary"]:
    print(f"\nSummary: {result['summary']}")
```

### JavaScript/Frontend Example

```javascript
async function analyzeTopics(text) {
  const response = await fetch('/analyze/topics', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  
  const data = await response.json();
  
  // Display AI summary
  if (data.summary) {
    console.log('Summary:', data.summary);
  }
  
  // Display clustered topics
  data.clustered_topics.forEach(cluster => {
    console.log(`${cluster.label}: ${cluster.total_mentions} mentions`);
  });
  
  return data;
}
```

## Parameters

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to analyze |
| `top_n` | integer | 10 | Number of topic clusters to return |
| `num_clusters` | integer | auto | Number of clusters (3-6 auto-determined) |
| `snippets_per_topic` | integer | 3 | Number of example snippets per cluster |

### Parameter Examples

**Get more clusters:**
```bash
curl -X POST "http://localhost:8000/analyze/topics?top_n=8&num_clusters=6" \
  -H "Content-Type: application/json" \
  -d '{"text": "..."}'
```

**Get more snippets per topic:**
```bash
curl -X POST "http://localhost:8000/analyze/topics?snippets_per_topic=5" \
  -H "Content-Type: application/json" \
  -d '{"text": "..."}'
```

## Response Structure

### Clustered Topics

Each topic cluster contains:

- `label` — AI-generated semantic label (e.g., "Border Security")
- `keywords` — List of related keywords with counts and relevance scores
- `avg_relevance` — Average relevance score for the cluster (0-1)
- `total_mentions` — Total mentions of all keywords in cluster

### Snippets

Each snippet group contains:

- `label` — Matches the topic cluster label
- `snippets` — Array of text excerpts with keywords highlighted using `**bold**` markdown
- `keyword_count` — Number of distinct keywords in this cluster

### Summary

- AI-generated 2-3 sentence interpretation of main themes
- Identifies dominant topics and patterns
- Provides objective, analytical perspective
- Only present if Gemini LLM is configured

### Metadata

- `total_keywords` — Number of keywords extracted before clustering
- `num_clusters` — Number of clusters created
- `text_length` — Length of input text in characters
- `has_ai_summary` — Whether AI summary was generated

## Use Cases

### 1. Political Speech Analysis

Analyze political speeches to identify main themes:

```python
speech_text = load_speech("path/to/speech.txt")
result = analyze_topics_enhanced(speech_text, top_n=6)

print(f"Main themes: {', '.join([c['label'] for c in result['clustered_topics']])}")
print(f"Summary: {result['summary']}")
```

### 2. Document Summarization

Extract key topics from long documents:

```python
for cluster in result['clustered_topics'][:3]:
    print(f"\n{cluster['label']}:")
    for snippet in result['snippets'][idx]['snippets']:
        print(f"  '{snippet}'")
```

### 3. Content Classification

Categorize documents by topic clusters:

```python
def categorize_document(text):
    result = analyze_topics_enhanced(text, top_n=3)
    return [cluster['label'] for cluster in result['clustered_topics']]
```

### 4. Sentiment + Topic Analysis

Combine with sentiment analysis for deeper insights:

```python
# Get topics
topics = analyze_topics_enhanced(text)

# Get sentiment
sentiment = analyze_sentiment(text)

# Combine insights
print(f"Document sentiment: {sentiment['sentiment']}")
print(f"Main topics: {[c['label'] for c in topics['clustered_topics'][:3]]}")
```

## Technical Details

### Clustering Algorithm

The system uses **KMeans clustering** on **MPNet embeddings** (768-dimensional vectors):

1. Extract top keywords using TF-IDF style scoring
2. Generate embeddings for each keyword using `all-mpnet-base-v2`
3. Cluster embeddings into 3-6 groups (auto-determined based on keyword count)
4. Sort clusters by total mentions (most important first)

### Label Generation

Cluster labels are generated using **Gemini LLM** with a specialized prompt:

```
Given these related keywords: economy, jobs, employment, market
Generate a concise 2-4 word label that captures the main theme.
```

If LLM is not available, falls back to using the top keyword as the label.

### Snippet Extraction

Snippets are extracted with context windows around keyword occurrences:

1. Find all positions of cluster keywords in text
2. Deduplicate nearby positions (min 200 chars apart)
3. Extract ±100 character context around each keyword
4. Clean up to sentence boundaries when possible
5. Highlight keywords with `**bold**` markdown

### AI Summary Generation

The summary is generated by providing Gemini with:

1. List of topic clusters with keywords and mention counts
2. Sample of the input text (first 2000 chars)
3. Prompt requesting 2-3 sentence analytical summary

## Performance Considerations

### Response Time

- **Without LLM:** ~1-2 seconds for typical documents
- **With LLM:** ~3-5 seconds (includes label + summary generation)

### Optimal Text Length

- **Minimum:** 100+ words for meaningful clustering
- **Optimal:** 500-2000 words
- **Maximum:** No hard limit, but longer texts increase processing time

### Caching

The embedding model and LLM service are initialized once at startup and reused across requests for optimal performance.

## Troubleshooting

### No Clusters Generated

**Problem:** Empty `clustered_topics` array

**Solutions:**
- Ensure text has at least 50-100 words
- Check that text contains meaningful content (not just stopwords)
- Try increasing `top_n` parameter

### Missing AI Summary

**Problem:** `summary` field is `null`

**Solutions:**
- Ensure `GEMINI_API_KEY` is configured in `.env`
- Check API logs for LLM errors
- Verify Gemini API quota/limits

### Service Unavailable

**Problem:** 503 error with "Topic service not initialized"

**Solutions:**
- Ensure application started successfully
- Check startup logs for topic service initialization errors
- Verify all dependencies are installed (`scikit-learn`, `sentence-transformers`)

## See Also

- [Architecture Documentation](../reference/architecture.md#enhanced-topic-extraction)
- [API Reference](https://trump-speeches-nlp-chatbot.azurewebsites.net/docs#/nlp/analyze_topics_enhanced_analyze_topics_enhanced_post)
- [RAG Features](rag-features.md) — Learn about the RAG system that also uses embeddings
