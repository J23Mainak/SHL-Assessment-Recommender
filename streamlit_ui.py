import streamlit as st
import requests
import pandas as pd
import os

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Assessment test Recommender",
    layout="wide"
)

# Ensure session state defaults
if "API_URL" not in st.session_state:
    st.session_state["API_URL"] = DEFAULT_API_URL
if "query" not in st.session_state:
    st.session_state["query"] = ""

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .footer {
        margin: 0; 
        padding: 0;
    }
    .footer p {
        margin: 0; 
        padding: 0;
    }
    .assessment-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<p style="font-size: 2rem; font-weight: bold; color: #1f77b4; text-align: center; padding: 0.6rem 0;">SHL Assessment Recommender 💖</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="font-size: 1.4rem; font-weight: normal; color: #333;"> Find the most relevant SHL assessments for your hiring needs</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:

    st.markdown("### About")
    st.info(
        """
        This system uses semantic embeddings + lexical matching to recommend relevant SHL assessments from the product catalog.

        **Features:**
        - Semantic search (configurable embedding model)
        - Hybrid BM25 + dense reranking
        - Test-type balancing for result diversity
        """
    )

    st.markdown("---")
    st.markdown("### Example Queries")
    examples = [
        "Java developer with strong collaboration skills",
        "Mid-level Python engineer for data analysis",
        "Sales manager with leadership experience",
        "Entry-level customer service representative",
    ]

    for ex in examples:
        # when clicked, set session_state['query']; Streamlit will rerun automatically
        if st.button(ex, key=ex):
            st.session_state["query"] = ex

    st.markdown("---")
    
    API_URL = st.text_input("API URL", value=st.session_state.get("API_URL", DEFAULT_API_URL))

    k = st.slider("Number of recommendations", min_value=5, max_value=10, value=10)
    

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        data = health.json()
        col1, col2 = st.columns([3, 1])
        with col2:
            st.success(f"Total assessments- {data.get('products_loaded', 0)}")
        with col1:
            st.markdown(f"**Model:** {data.get('embed_model_name', 'unknown')}")
    else:
        st.error(f"API health check failed ({health.status_code}) — check server logs.")
except Exception:
    st.error(
        "API not reachable. Please start the API server (e.g. `uvicorn app:app --reload`) and confirm API URL in the sidebar."
    )
    st.stop()

# Main area
query = st.text_area(
    "Enter Job Description or Query",
    value=st.session_state.get("query", ""),
    height=200,
    placeholder="Example: Looking for a senior Java developer who can collaborate effectively with cross-functional teams and has experience in agile methodologies...",
)

col1, col2, col3 = st.columns([2, 1, 2])

with col2:
    search_button = st.button("Find Assessments")

if search_button and query.strip():
    with st.spinner("Analyzing query and finding best assessments..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={"query": query, "k": k},
                timeout=60,
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception:
                    st.error("Received non-JSON response from API.")
                    st.stop()

                results = data.get("results", [])
                if results:
                    st.success(f"-> Found {len(results)} relevant assessments!")

                    st.markdown("### Recommended Assessments")

                    for i, rec in enumerate(results, 1):
                        with st.container():
                            colA, colB = st.columns([4, 1])

                            with colA:
                                name = rec.get("assessment_name") or "Untitled Assessment"
                                url = rec.get("url") or "#"
                                st.markdown(f"#### {i}. [{name}]({url})")

                            with colB:
                                score = rec.get("score", 0.0)
                                st.markdown(f"**Score:** {score:.3f}")

                            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

                            with meta_col1:
                                test_type = rec.get("test_type", "N/A")
                                st.markdown(f"**Type:** {test_type}")

                            with meta_col2:
                                duration = rec.get("duration", "N/A") or "N/A"
                                st.markdown(f"**Duration:** {duration}")

                            with meta_col3:
                                remote = rec.get("remote_support", "")
                                icon = "✅" if str(remote).strip().lower() in ("yes", "true", "1") else "❌"
                                st.markdown(f"**Remote:** {icon}")

                            with meta_col4:
                                adaptive = rec.get("adaptive_support", "")
                                icon = "✅" if str(adaptive).strip().lower() in ("yes", "true", "1") else "❌"
                                st.markdown(f"**Adaptive:** {icon}")

                            st.markdown("---")

                    # Export results
                    st.markdown("### 💾 Export Results")

                    export_data = []
                    for rec in results:
                        export_data.append(
                            {
                                "Assessment Name": rec.get("assessment_name", ""),
                                "URL": rec.get("url", ""),
                                "Test Type": rec.get("test_type", ""),
                                "Duration": rec.get("duration", ""),
                                "Remote Support": rec.get("remote_support", ""),
                                "Adaptive Support": rec.get("adaptive_support", ""),
                            }
                        )

                    df = pd.DataFrame(export_data)

                    colD1, colD2 = st.columns(2)
                    with colD1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="shl_recommendations.csv",
                            mime="text/csv",
                        )

                    with colD2:
                        submission_data = []
                        for rec in results:
                            submission_data.append({"Query": query, "Assessment_url": rec.get("url", "")})
                        submission_df = pd.DataFrame(submission_data)
                        submission_csv = submission_df.to_csv(index=False)

                        st.download_button(
                            label="📥 Download Submission Format",
                            data=submission_csv,
                            file_name="submission_predictions.csv",
                            mime="text/csv",
                        )

                else:
                    st.warning("No assessments found for this query. Try different keywords.")

            else:
                text = response.text
                st.error(f"Error from API ({response.status_code}): {text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Network/API error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

elif search_button:
    st.warning("Please enter a query")

# Footer
st.markdown("---")
st.markdown(
    """
<div class='footer' style='text-align: center; color: #666;'>
    <p class='footer p'>SHL Assessment Recommendation System &nbsp; | &nbsp; AI-powered semantic + lexical retrieval</p>
    <p class='footer p'>For support, contact system administrator!</p>
</div>
""",
    unsafe_allow_html=True,
)
