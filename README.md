# 🌍📖 Semantic & Emotion-Aware Book Recommendation System

> *“Not just what you want to read, but how you want to feel.”*
> *Redefining the future of reading using AI, emotions, and intent.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Streamlit-red)
----------------------------------------------------------------

## 🚀 About the Project

This project is an **intelligent book recommendation system** that combines **semantic search** and **emotion analysis** to recommend books based on **what users describe in natural language** and **how they want to feel**.

Unlike traditional recommenders that rely only on ratings or collaborative filtering, this system uses **vector embeddings**, **zero-shot classification**, and **emotion-aware ranking** to make discovery more meaningful and personalized.

---

## 🔍 Key Highlights

### 🧠 AI-Powered Semantic Search

* Uses **Google Generative AI embeddings** to capture the **semantic meaning** of book descriptions and user queries.
* Retrieves books that are conceptually similar, even if keywords don’t match.

### 🎭 Emotion-Aware Recommendations

* Analyzes each book’s description with a **DistilRoBERTa-based Hugging Face emotion classifier**.
* Scores emotional tones like Joy, Sadness, Fear, Anger, Surprise, Neutral.
* Lets users filter or rank results by mood (e.g., “happy romance” or “suspenseful thriller”).

### 🧩 Genre & Category Control

* Uses **zero-shot classification (BART-large-MNLI)** to fill missing categories.
* Enables filtering by genres like Mystery, Romance, Fantasy, Thriller, Sci-Fi, etc.

### 🖥️ Interactive Frontend (Streamlit App)

* A clean **Streamlit dashboard** where users type queries, select categories and moods, and instantly get personalized book recommendations.

---

## 📸 Screenshot

<img src="https://github.com/raahulmaurya1/Semantic-Book-recommendation/blob/e08b766e8bdda2271cd275f86acc576583db228b/Picture1.png" width="500" height="500"/>

---

## 🧪 Technical Stack

| Layer                  | Technology Used                                                      |
| ---------------------- | -------------------------------------------------------------------- |
| 🧠 Semantic Embeddings | [Google Generative AI Embeddings](https://ai.google.dev/)            |
| 🎭 Emotion Detection   | [Hugging Face Transformers](https://huggingface.co/) (DistilRoBERTa) |
| 🔗 Semantic Retrieval  | [ChromaDB](https://www.trychroma.com/) / [FAISS](https://faiss.ai/)  |
| 🧩 Category Enrichment | Hugging Face Zero-Shot (BART-large-MNLI)                             |
| 💻 UI Layer            | [Streamlit](https://streamlit.io/)                                   |
| 📊 Data Processing     | Python (Pandas, NumPy)                                               |

---

## 📂 Project Structure

```
📁 book-recommendation-system/
├── app.py                        # Streamlit frontend + backend pipeline
├── .env                          # Environment variables (API keys)
├── requirements.txt              # Python dependencies
├── 📁 data/
│   ├── books_with_emotions.csv    # Metadata + emotion scores
│   ├── books_with_categories.csv  # Metadata + inferred genres
│   ├── books_cleaned.csv          # Base cleaned dataset
│   └── tagged_description.txt     # Preprocessed text for embeddings
├── 📁 notebooks/
│   ├── data_exploration.ipynb     # EDA & cleaning steps
│   ├── sentiment-analysis.ipynb   # Emotion classification workflow
│   ├── text-classification.ipynb  # Zero-shot category classification
│   └── vector-search.ipynb        # Embedding + semantic search logic
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_API_KEY=your_hugging_face_api_key
```

### 4️⃣ Run the App

```bash
streamlit run app.py
```

### 5️⃣ Access the System

Open `http://localhost:8501/` in your browser and start exploring.

---

## 📊 Datasets Used

* **books\_cleaned.csv** – Base metadata after preprocessing
* **books\_with\_categories.csv** – Enriched with inferred categories
* **books\_with\_emotions.csv** – Enriched with emotion probabilities
* **tagged\_description.txt** – Input corpus for embedding generation

---

## 🔬 Example Use Cases

| User Input                                          | System Output                             |
| --------------------------------------------------- | ----------------------------------------- |
| *“I feel heartbroken, I need something inspiring.”* | Shows motivational and hopeful stories    |
| *“Give me a suspenseful sci-fi novel.”*             | Pulls thrilling futuristic plots          |
| *“Happy romance in a magical world.”*               | Recommends uplifting fantasy love stories |

---

## 🛠️ Workflow

1. **Data Preparation** → Clean metadata, merge title+description, generate tagged text.
2. **Category Enrichment** → Zero-shot classification fills missing genres.
3. **Emotion Analysis** → Hugging Face DistilRoBERTa assigns tone scores.
4. **Embeddings** → Google AI embeddings create high-dimensional vectors.
5. **Vector Database** → Store embeddings in ChromaDB / FAISS.
6. **Semantic Search** → Convert user query → embedding → cosine similarity search.
7. **Filtering** → Apply category + emotion tone filters.
8. **Final Recommendations** → Ranked list shown via Streamlit frontend.

---

## 🤝 Contribution

Contributions are welcome!
Open an issue or submit a PR to improve emotion detection, expand the dataset, or optimize performance.

---

## 📜 License

Distributed under the **MIT License**.
See [LICENSE](LICENSE) for more information.

---

## 🙏 Acknowledgments

* [ChromaDB](https://www.trychroma.com/) – Vector search database
* [FAISS](https://faiss.ai/) – Scalable similarity search
* [Hugging Face](https://huggingface.co/) – Emotion & classification models
* [Google Generative AI](https://ai.google.dev/) – Embeddings
* [Streamlit](https://streamlit.io/) – UI framework

---

## 🌐 Final Thought

> This is more than a tool. It’s a **companion for readers**.
> A system that listens to your emotions, understands your thoughts, and recommends books that truly **resonate**.

---

## 📬 Contact

Created by Krishna Kumar Sah
📧 Email: [krishnasah2060@gmail.com](mailto:krishnasah2060@gmail.com)
🔗 GitHub: [Krishnasah206](https://github.com/Krishnasah206)

### Collaborator

🔗 GitHub: [raahulmaurya1](https://github.com/raahulmaurya1)
