import streamlit as st
import pickle
import pandas as pd
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from io import BytesIO
from streamlit_lottie import st_lottie
import json
import requests

# ======================
# KONFIGURASI TAMPILAN
# ======================
st.set_page_config(layout="wide", page_title="Analisis Topik LDA", page_icon="📊")

# Fungsi untuk memuat animasi Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animasi Lottie
lottie_loading = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_raiw2hpe.json")
lottie_success = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_yk7gni7y.json")
lottie_chart = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_0clcyw1e.json")

# CSS Custom dengan animasi
st.markdown("""
    <style>
    .stApp {
        background-color: #e5e5f7;
        opacity: 0.8;
        background-image:  radial-gradient(#444cf7 0.5px, transparent 0.5px), radial-gradient(#444cf7 0.5px, #e5e5f7 0.5px);
        background-size: 20px 20px;
        background-position: 0 0,10px 10px;
    }
    .title {
        font-size: 36px !important;
        text-align: center;
        margin-bottom: 30px;
        color: #2E86C1;
        animation: fadeIn 2s;
    }
    .section {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform .2s;
    }
    .section:hover {
        transform: scale(1.01);
    }
    .topic-card {
        border-left: 5px solid #2E86C1;
        padding: 15px;
        margin-bottom: 10px;
        background-color: white;
        border-radius: 5px;
        animation: slideIn 0.5s ease-out;
    }
    .metric-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #e8f4f8;
        margin-bottom: 10px;
        transition: all 0.3s;
    }
    .metric-box:hover {
        box-shadow: 0 5px 15px rgba(46, 134, 193, 0.3);
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { 
            transform: translateX(-20px);
            opacity: 0;
        }
        to { 
            transform: translateX(0);
            opacity: 1;
        }
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# FUNGSI UTAMA
# ======================
@st.cache_resource
def load_data():
    try:
        with st.spinner('Memuat data...'):
            if lottie_loading:
                st_lottie(lottie_loading, height=200, key="loading")
            
            with open("lda_full_result.pkl", "rb") as f:
                data = pickle.load(f)
            
            required_keys = ['dataset', 'topics', 'alpha_results', 'beta_results', 
                            'global_results', 'lda_best_model', 'corpus', 'gensim_dict']
            missing_keys = [k for k in required_keys if k not in data]
            
            if missing_keys:
                st.error(f"Data tidak lengkap! Komponen yang hilang: {', '.join(missing_keys)}")
                st.stop()
                
            if lottie_success:
                st_lottie(lottie_success, height=150, key="success")
            
            return data
    except FileNotFoundError:
        st.error("File lda_full_result.pkl tidak ditemukan di direktori ini!")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        st.stop()

# ======================
# TAMPILAN UTAMA
# ======================
def main():
    # Header dengan animasi
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<h1 class="title pulse">📊 Analisis Topik dengan LDA</h1>', unsafe_allow_html=True)
        if lottie_chart:
            st_lottie(lottie_chart, height=150, key="header")
    
    # Memuat data
    data = load_data()
    
    # ======================
    # BAGIAN 1: DATA PREPROCESSING
    # ======================
    with st.expander("📁 Data Komentar Asli", expanded=False):
        df = pd.DataFrame(data['dataset'])
        
        col1, col2 = st.columns([2,3])
        rows_per_page = col1.selectbox("Baris per halaman", [10, 20, 50, 100], index=0)
        total_pages = max(1, (len(df) - 1) // rows_per_page + 1)
        page_number = col2.number_input("Halaman", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page_number - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        st.dataframe(
            df.iloc[start_idx:end_idx],
            height=400
        )

    # ======================
    # BAGIAN 2: TOPIK YANG DIHASILKAN
    # ======================
    st.markdown('<div class="section"><h2>🎯 Topik yang Dihasilkan</h2></div>', unsafe_allow_html=True)
    
    cols = st.columns(2)
    for i, (idx, topic) in enumerate(data['topics']):
        with cols[i%2]:
            st.markdown(f"""
            <div class="topic-card">
                <h4>Topik {idx}</h4>
                <p>{topic}</p>
            </div>
            """, unsafe_allow_html=True)

    # ======================
    # BAGIAN 3: EVALUASI MODEL
    # ======================
    st.markdown('<div class="section"><h2>📈 Evaluasi Model</h2></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Parameter Alpha", "Parameter Beta"])
    
    with tab1:
        alpha_df = pd.DataFrame(data['alpha_results'], 
                              columns=["Alpha", "Beta", "Perplexity", "Coherence"])
        st.dataframe(
            alpha_df.style.format({
                "Perplexity": "{:.2f}",
                "Coherence": "{:.4f}"
            }).applymap(lambda x: "background-color: #e6f7ff" if isinstance(x, (int, float)) else "")
        )
        
    with tab2:
        beta_df = pd.DataFrame(data['beta_results'], 
                             columns=["Alpha", "Beta", "Perplexity", "Coherence"])
        st.dataframe(
            beta_df.style.format({
                "Perplexity": "{:.2f}",
                "Coherence": "{:.4f}"
            }).applymap(lambda x: "background-color: #e6f7ff" if isinstance(x, (int, float)) else "")
        )

    # ======================
    # BAGIAN 4: HASIL TERBAIK
    # ======================
        st.markdown('<div class="section"><h2>🏆 Hasil Terbaik</h2></div>', unsafe_allow_html=True)
    
    if isinstance(data['global_results'], dict):
        best_params = data['global_results']
    else:
        best_params = data['global_results'].to_dict()
    
    col1, col2 = st.columns(2)
    col1.metric("Alpha Terbaik", best_params.get('Perplexity', '-'), delta_color="off")
    col2.metric("Nilai Coherence", f"{best_params.get('Coherence', 0):.4f}", delta_color="off")

    # ======================
    # BAGIAN 5: VISUALISASI INTERAKTIF
    # ======================
    st.markdown('<div class="section"><h2>🔍 Visualisasi Interaktif</h2></div>', unsafe_allow_html=True)
    
    with st.spinner("Menyiapkan visualisasi..."):
        try:
            vis = gensimvis.prepare(
                data['lda_best_model'],
                data['corpus'],
                data['gensim_dict']
            )
            
            html = pyLDAvis.prepared_data_to_html(vis)
            st.components.v1.html(html, width=1300, height=800)
            
            html_bytes = BytesIO()
            html_bytes.write(html.encode())
            st.download_button(
                label="💾 Download Visualisasi",
                data=html_bytes.getvalue(),
                file_name="lda_visualization.html",
                mime="text/html",
                key="download-btn"
            )
        except Exception as e:
            st.error(f"Gagal membuat visualisasi: {str(e)}")

    # ======================
    # SIDEBAR INFORMASI
    # ======================
    with st.sidebar:
        st.markdown('<div class="metric-box pulse">', unsafe_allow_html=True)
        st.header("ℹ Informasi Model")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-box">
            <h4>📊 Statistik Model</h4>
            <p>• Jumlah Topik: {data['lda_best_model'].num_topics}</p>
            <p>• Kata Unik: {len(data['gensim_dict'])}</p>
            <p>• Dokumen: {len(data['dataset'])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("🔑 Kata Kunci Topik")
        for topic_id in range(data['lda_best_model'].num_topics):
            topic_words = data['lda_best_model'].show_topic(topic_id, topn=5)
            words = "<br>".join([f"▪ {word} ({prob:.2f})" for word, prob in topic_words])
            
            st.markdown(f"""
            <div class="metric-box">
                <h4>Topik {topic_id + 1}</h4>
                {words}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()