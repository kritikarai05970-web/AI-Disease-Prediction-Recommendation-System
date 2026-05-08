import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pickle
import numpy as np
import ast


st.set_page_config(
    page_title="AI Disease Prediction & Recommendation System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

 
@st.cache_resource
def load_resources():
    model      = pickle.load(open("model.pkl", "rb"))
    desc       = pd.read_csv("description.csv")
    precaution = pd.read_csv("precautions_df.csv")
    meds       = pd.read_csv("medications.csv")
    diet       = pd.read_csv("diets.csv")
    workout    = pd.read_csv("workout_df.csv")
    training   = pd.read_csv("Training.csv")
    features   = training.columns[:-1].tolist()

    for df in [desc, precaution, meds, diet]:
        df['Disease'] = df['Disease'].str.strip().str.lower()
    workout['disease'] = workout['disease'].str.strip().str.lower()

    return model, desc, precaution, meds, diet, workout, features

model, desc, precaution, meds, diet, workout, features = load_resources()

components.html("""
<script>
(function() {
  var css = `
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Instrument+Sans:wght@400;500;600&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    [data-testid="stAppViewContainer"] {
      background: #050D0A !important;
      font-family: 'Instrument Sans', sans-serif;
    }
    [data-testid="stHeader"]           { background: transparent !important; }
    [data-testid="stSidebar"]          { display: none !important; }
    section.main > div                 { padding-top: 0 !important; }
    #MainMenu, footer, header          { visibility: hidden !important; }
    [data-testid="stToolbar"]          { display: none !important; }
    [data-testid="stDecoration"]       { display: none !important; }
    [data-testid="stStatusWidget"]     { display: none !important; }
    [data-testid="manage-app-button"]  { display: none !important; }

    [data-testid="stAppViewContainer"]::before {
      content: '';
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(16,185,95,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(5,150,105,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(16,185,95,0.04) 0%, transparent 70%);
      pointer-events: none;
      z-index: 0;
    }

    [data-testid="stMultiSelect"] > div {
      background: rgba(255,255,255,0.04) !important;
      border: 1px solid rgba(16,185,95,0.25) !important;
      border-radius: 12px !important;
      color: #E8F5E9 !important;
    }
    [data-testid="stMultiSelect"] > div:hover {
      border-color: rgba(16,185,95,0.5) !important;
    }
    [data-testid="stMultiSelect"] label {
      color: rgba(200,230,210,0.7) !important;
      font-family: 'Syne', sans-serif !important;
      font-size: 11px !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase !important;
    }
    [data-baseweb="tag"] {
      background: rgba(16,185,95,0.15) !important;
      border: 1px solid rgba(16,185,95,0.4) !important;
      border-radius: 100px !important;
      color: #6EE7A0 !important;
    }
    [data-baseweb="tag"] span { color: #6EE7A0 !important; }
    [data-testid="stMultiSelect"] input { color: #E8F5E9 !important; }

    [data-testid="stButton"] > button {
      width: 100%;
      background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
      color: #fff !important;
      border: none !important;
      border-radius: 12px !important;
      padding: 14px 28px !important;
      font-family: 'Syne', sans-serif !important;
      font-size: 15px !important;
      font-weight: 600 !important;
      letter-spacing: 0.04em !important;
      cursor: pointer !important;
      transition: opacity 0.2s, transform 0.1s !important;
      margin-top: 8px;
    }
    [data-testid="stButton"] > button:hover  { opacity: 0.88 !important; }
    [data-testid="stButton"] > button:active { transform: scale(0.99) !important; }

    [data-testid="stTabs"] [role="tablist"] {
      background: rgba(255,255,255,0.03) !important;
      border-radius: 10px !important;
      padding: 4px !important;
      border: 1px solid rgba(16,185,95,0.12) !important;
      gap: 2px !important;
    }
    [data-testid="stTabs"] [role="tab"] {
      font-family: 'Syne', sans-serif !important;
      font-size: 13px !important;
      font-weight: 500 !important;
      color: rgba(200,230,210,0.55) !important;
      border-radius: 8px !important;
      padding: 8px 16px !important;
      transition: all 0.15s !important;
      border: none !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
      background: rgba(16,185,95,0.18) !important;
      color: #6EE7A0 !important;
    }
    [data-testid="stTabs"] [role="tab"]:hover { color: #A7F3D0 !important; }
    [data-testid="stTabPanel"] {
      background: rgba(255,255,255,0.02) !important;
      border-radius: 0 0 12px 12px !important;
      padding: 20px 16px !important;
      border: 1px solid rgba(16,185,95,0.08) !important;
      border-top: none !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li { color: #B8D4C0 !important; }

    [data-testid="stAlert"] {
      border-radius: 10px !important;
      border: none !important;
    }

    ::-webkit-scrollbar       { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(16,185,95,0.3); border-radius: 3px; }

    @keyframes pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,95,0.5); }
      50%       { box-shadow: 0 0 0 8px rgba(16,185,95,0); }
    }
  `;

  var style = window.parent.document.createElement('style');
  style.id = 'medisense-styles';
  style.innerHTML = css;

  // Remove old injection if re-running
  var old = window.parent.document.getElementById('medisense-styles');
  if (old) old.remove();

  window.parent.document.head.appendChild(style);

  // Inject Google Fonts
  if (!window.parent.document.getElementById('medisense-fonts')) {
    var link = window.parent.document.createElement('link');
    link.id   = 'medisense-fonts';
    link.rel  = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Instrument+Sans:wght@400;500;600&display=swap';
    window.parent.document.head.appendChild(link);
  }
})();
</script>
""", height=0, scrolling=False)


for key in ["prediction", "confidence", "top3"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown("""
<div style="padding: 52px 0 36px; position: relative; z-index: 1;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#10B981;
                animation: pulse 2s infinite;"></div>
    <span style="font-family:'Syne',sans-serif; font-size:11px; letter-spacing:0.14em;
                 text-transform:uppercase; color:#10B981; font-weight:600;">
      AI Diagnostic System · Live
    </span>
  </div>
  <h1 style="font-family:'Syne',sans-serif; font-size:clamp(36px,6vw,64px);
             font-weight:800; color:#F0FDF4; line-height:1.08;
             letter-spacing:-0.03em; margin:0 0 16px;">
    Know your body.<br>
    <span style="color:#10B981;">Understand</span> your symptoms.
  </h1>
  <p style="font-family:'Instrument Sans',sans-serif; font-size:16px;
            color:rgba(200,230,210,0.6); max-width:500px; line-height:1.7; margin:0;">
    Our machine learning model analyses your symptoms and predicts likely conditions —
    with personalised diet, medication, and wellness recommendations.
  </p>
</div>
<div style="height:1px; background:linear-gradient(90deg,
  transparent 0%, rgba(16,185,95,0.3) 30%, rgba(16,185,95,0.3) 70%, transparent 100%);
  margin-bottom:32px;"></div>
""", unsafe_allow_html=True)


st.markdown("""
<p style="font-family:'Syne',sans-serif; font-size:11px; letter-spacing:0.12em;
          text-transform:uppercase; color:rgba(16,185,95,0.7); margin-bottom:6px;">
  Step 1 — Select your symptoms
</p>
""", unsafe_allow_html=True)

selected_symptoms = st.multiselect(
    "Select Symptoms",
    options=features,
    placeholder="Type to search symptoms…",
    label_visibility="collapsed"
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
predict_clicked = st.button("⚕  Analyse Symptoms", use_container_width=True)

 
if predict_clicked:
    if not selected_symptoms:
        st.markdown("""
        <div style="margin-top:12px; padding:14px 18px;
                    background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
                    border-radius:10px; color:#FCA5A5; font-size:14px;
                    font-family:'Instrument Sans',sans-serif;">
          ⚠ Please select at least one symptom before analysing.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analysing symptoms…"):
            input_vector = np.zeros(len(features))
            for symptom in selected_symptoms:
                input_vector[features.index(symptom)] = 1

            prediction    = model.predict([input_vector])[0]
            probabilities = model.predict_proba([input_vector])[0]
            confidence    = round(np.max(probabilities) * 100, 2)

            top3_indices  = np.argsort(probabilities)[-3:][::-1]
            top3_diseases = model.classes_[top3_indices]
            top3_probs    = probabilities[top3_indices] * 100

            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.top3       = list(zip(top3_diseases, top3_probs))


if st.session_state.prediction:
    disease = st.session_state.prediction.strip().lower()
    conf    = st.session_state.confidence
    top3    = st.session_state.top3

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="height:1px; background:linear-gradient(90deg,
      transparent, rgba(16,185,95,0.2), transparent); margin-bottom:32px;"></div>""",
      unsafe_allow_html=True)

    
    symp_count = len(selected_symptoms) if selected_symptoms else 0
    st.markdown(f"""
    <div style="position:relative; overflow:hidden; border-radius:20px;
                background:linear-gradient(135deg, #022C22 0%, #064E3B 50%, #065F46 100%);
                border:1px solid rgba(16,185,95,0.25); padding:36px 40px; margin-bottom:20px;">
      <div style="position:absolute;right:-40px;top:-40px;width:200px;height:200px;
                  border-radius:50%;background:rgba(16,185,95,0.07);pointer-events:none;"></div>
      <div style="position:absolute;right:40px;bottom:-60px;width:140px;height:140px;
                  border-radius:50%;background:rgba(16,185,95,0.05);pointer-events:none;"></div>
      <div style="font-family:'Syne',sans-serif;font-size:11px;letter-spacing:0.14em;
                  text-transform:uppercase;color:rgba(110,231,160,0.6);margin-bottom:10px;">
        Most Likely Condition
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:clamp(28px,4vw,42px);
                  font-weight:800;color:#F0FDF4;letter-spacing:-0.02em;margin-bottom:12px;">
        {disease.title()}
      </div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:10px;">
          <div style="height:6px;width:120px;background:rgba(255,255,255,0.1);
                      border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:{min(conf,100):.0f}%;background:#10B981;border-radius:3px;"></div>
          </div>
          <span style="font-family:'Syne',sans-serif;font-size:14px;
                       color:#6EE7A0;font-weight:600;">{conf:.1f}% confidence</span>
        </div>
        <span style="font-size:13px;color:rgba(200,230,210,0.45);
                     font-family:'Instrument Sans',sans-serif;">
          Based on {symp_count} symptom{'s' if symp_count != 1 else ''}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    
    rank_labels = ["🥇 Top match", "🥈 2nd likely", "🥉 3rd likely"]
    card_styles = [
        ("rgba(16,185,95,0.12)", "rgba(16,185,95,0.3)",   "#10B981"),
        ("rgba(255,255,255,0.04)", "rgba(255,255,255,0.1)", "#94A3B8"),
        ("rgba(255,255,255,0.03)", "rgba(255,255,255,0.07)","#64748B"),
    ]
    cols = st.columns(3)
    for i, ((d_name, d_prob), col) in enumerate(zip(top3, cols)):
        bg, border, bar = card_styles[i]
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:14px;
                        padding:20px;min-height:130px;">
              <div style="font-family:'Syne',sans-serif;font-size:10px;letter-spacing:0.1em;
                          text-transform:uppercase;color:rgba(200,230,210,0.55);margin-bottom:8px;">
                {rank_labels[i]}
              </div>
              <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                          color:#E8F5E9;margin-bottom:12px;line-height:1.3;">
                {d_name.title()}
              </div>
              <div style="height:4px;background:rgba(255,255,255,0.07);
                          border-radius:2px;overflow:hidden;margin-bottom:6px;">
                <div style="height:100%;width:{min(d_prob,100):.0f}%;
                            background:{bar};border-radius:2px;"></div>
              </div>
              <div style="font-family:'Instrument Sans',sans-serif;font-size:13px;
                          color:rgba(200,230,210,0.6);font-weight:500;">{d_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-family:'Syne',sans-serif;font-size:11px;letter-spacing:0.12em;
              text-transform:uppercase;color:rgba(16,185,95,0.7);margin-bottom:12px;">
      Step 2 — Review recommendations
    </p>
    """, unsafe_allow_html=True)

    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋  Description", "🛡️  Precautions", "💊  Medications", "🥗  Diet", "🏃  Workout"
    ])

    def render_items(items, dot="#10B981", bg="rgba(16,185,95,0.08)", border="rgba(16,185,95,0.2)"):
        html = '<div style="display:flex;flex-direction:column;gap:8px;">'
        for item in items:
            html += f"""
            <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 16px;
                        background:{bg};border:1px solid {border};border-radius:10px;">
              <span style="width:6px;height:6px;border-radius:50%;background:{dot};
                           flex-shrink:0;margin-top:7px;"></span>
              <span style="font-family:'Instrument Sans',sans-serif;font-size:14px;
                           color:#C8E6C9;line-height:1.6;">{item}</span>
            </div>"""
        html += "</div>"
        return html

    with tab1:
        d_row = desc[desc['Disease'] == disease]['Description']
        text  = d_row.values[0] if len(d_row) > 0 else "No description available."
        st.markdown(f"""<p style="font-family:'Instrument Sans',sans-serif;font-size:15px;
                  color:#B8D4C0;line-height:1.8;margin:8px 4px;">{text}</p>""",
                  unsafe_allow_html=True)

    with tab2:
        p_df = precaution[precaution['Disease'] == disease]
        if not p_df.empty:
            items = [str(x) for x in p_df.iloc[0, 1:] if pd.notna(x)]
            st.markdown(render_items(items, "#F59E0B", "rgba(245,158,11,0.08)", "rgba(245,158,11,0.2)"), unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#B8D4C0;'>No precautions available.</p>", unsafe_allow_html=True)

    with tab3:
        m_row = meds[meds['Disease'] == disease]['Medication']
        if len(m_row) > 0:
            try:    med_list = ast.literal_eval(m_row.values[0])
            except: med_list = [m_row.values[0]]
            st.markdown(render_items(med_list, "#60A5FA", "rgba(96,165,250,0.08)", "rgba(96,165,250,0.2)"), unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#B8D4C0;'>No medications available.</p>", unsafe_allow_html=True)

    with tab4:
        d_row = diet[diet['Disease'] == disease]['Diet']
        if len(d_row) > 0:
            try:    diet_list = ast.literal_eval(d_row.values[0])
            except: diet_list = [d_row.values[0]]
            st.markdown(render_items(diet_list, "#34D399", "rgba(52,211,153,0.08)", "rgba(52,211,153,0.2)"), unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#B8D4C0;'>No diet plan available.</p>", unsafe_allow_html=True)

    with tab5:
        w_row = workout[workout['disease'] == disease]['workout']
        if len(w_row) > 0:
            st.markdown(render_items(list(w_row), "#A78BFA", "rgba(167,139,250,0.08)", "rgba(167,139,250,0.2)"), unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#B8D4C0;'>No workout advice available.</p>", unsafe_allow_html=True)

    #  Disclaimer 
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:16px 20px;background:rgba(245,158,11,0.06);
                border:1px solid rgba(245,158,11,0.2);border-radius:12px;
                display:flex;gap:12px;align-items:flex-start;">
      <span style="font-size:16px;flex-shrink:0;">⚠️</span>
      <p style="font-family:'Instrument Sans',sans-serif;font-size:13px;
                color:rgba(251,191,36,0.75);line-height:1.6;margin:0;">
        <strong style="color:rgba(251,191,36,0.9);">Medical disclaimer:</strong>
        This tool is for informational purposes only and does not constitute medical advice,
        diagnosis, or treatment. Always consult a qualified healthcare professional.
      </p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px 0;">
  <p style="font-family:'Syne',sans-serif;font-size:12px;letter-spacing:0.08em;
            color:rgba(200,230,210,0.2);">
    AI Disease Prediction & Recommendation System · FOR EDUCATIONAL PURPOSES ONLY
  </p>
</div>
""", unsafe_allow_html=True)
