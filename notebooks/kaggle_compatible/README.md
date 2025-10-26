# Kaggle-Compatible Notebooks

This folder contains optimized versions of project notebooks specifically prepared for uploading to Kaggle.

## 📁 Contents

### 1. `trump-rally-speeches-nlp-analysis.ipynb`
**Word Frequency & Topics Analysis** - Beginner-friendly text analysis

**Features:**
- Word clouds and n-gram analysis (unigrams, bigrams, trigrams)
- Temporal linguistic patterns (2019 vs 2020)
- Topic extraction and keyword frequency
- Interactive Plotly visualizations

**Runtime:** ~2-3 minutes (CPU) | **Size:** ~70 KB

---

### 2. `trump-rally-sentiment-analysis.ipynb`
**Advanced Sentiment Analysis with FinBERT** - Deep learning showcase

**Features:**
- BERT-based transformer model for sentiment classification
- Text chunking strategy for handling long documents
- Chunk-level and speech-level sentiment aggregation
- Temporal sentiment trends and year-over-year comparison
- Interactive dashboards and heatmaps

**Runtime:** ~5-10 minutes (CPU), ~2-3 minutes (GPU) | **Size:** ~85 KB

**Note:** This notebook demonstrates advanced NLP/ML skills with state-of-the-art transformers!

---

**Key Changes for Kaggle:**
- ✅ Added comprehensive headers with project links and tech stack
- ✅ Adjusted data loading path for Kaggle's `/kaggle/input/` structure
- ✅ Removed all output cells to minimize file size (< 1MB requirement)
- ✅ Enhanced documentation and context for Kaggle audience
- ✅ Added "Next Steps" sections linking to full project
- ✅ Professional formatting with clear sections
- ✅ Self-contained (no `%store` dependencies between notebooks)

## 📤 How to Upload to Kaggle

1. **Go to the dataset page**: [Donald Trump Rally Speeches](https://www.kaggle.com/datasets/christianlillelund/donald-trumps-rallies)

2. **Click "New Notebook"**

3. **Upload this notebook**:
   - Click "+ Add data" → "Upload"
   - Select the `.ipynb` file you want to upload

4. **Run the notebook** to generate outputs

5. **Make it public** and share!

## 📊 File Size Check

Before uploading, verify each file is under 1MB:
```powershell
# In PowerShell
(Get-Item "trump-rally-speeches-nlp-analysis.ipynb").Length / 1MB
(Get-Item "trump-rally-sentiment-analysis.ipynb").Length / 1MB
# Both should be < 1.0
```

## ✏️ Customization Before Upload

**Update these placeholders in the notebook header:**
- Replace `yourusername` with your GitHub username in all URLs
- Update API deployment URL once you deploy
- Add your contact information

**Search for:** `JustaKris` and replace with your username
**Search for:** `https://trump-speeches-nlp-api.onrender.com` and update with your deployed URL

## 🎯 What Makes This Kaggle-Friendly?

### Professional Header
- Clear project description
- Links to GitHub, API, and documentation
- Attribution to original dataset
- Tech stack overview

### Optimized for Kaggle Environment
- Data loading works with `/kaggle/input/` path
- Fallback to local path for development
- No large embedded outputs
- Clean, runnable cells

### Engagement Features
- Clear section headers
- Comprehensive final summary
- "Next Steps" section promoting full project
- Call-to-action for upvotes and feedback

## 💡 Tips for Success on Kaggle

### Kernel Settings

**For Word Frequency Notebook:**
- Language: Python
- Accelerator: None (CPU is fine)
- Internet: ON (required for NLTK data)

**For Sentiment Analysis Notebook:**
- Language: Python
- Accelerator: **GPU T4 x2** (recommended for FinBERT model)
- Internet: ON (required for downloading model)

### Best Practices
1. **Run all cells before publishing** - Kaggle shows outputs in the preview
2. **Add markdown explanations** - Help readers understand your analysis
3. **Use interactive visualizations** - Plotly charts are engaging
4. **Respond to comments** - Build community engagement
5. **Update regularly** - Show you're actively maintaining
6. **Cross-reference notebooks** - Link the word frequency ↔ sentiment analysis notebooks

## 🔗 Expected Links to Update

Current placeholders (replace with actual URLs):
- GitHub repo: `https://github.com/JustaKris/Donald-Trump-Rally-Speeches-NLP`
- Live API: `https://trump-speeches-nlp-api.onrender.com/docs`
- Author GitHub: `https://github.com/JustaKris`

## 🏆 Kaggle Medals

Quality notebooks can earn Kaggle medals:
- **Bronze**: 5+ upvotes
- **Silver**: 20+ upvotes
- **Gold**: 50+ upvotes

Good documentation, clear analysis, and community engagement are key!

---

**Note:** This folder is committed to Git. The notebooks here are presentation versions meant for public sharing, not working versions with outputs.
