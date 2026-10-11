import streamlit as st
import pandas as pd
from major_project_1 import df, similarity_df, clf, user_mapping

st.set_page_config(
    page_title="Profile Recommendation System",
    page_icon="🤝",
    layout="wide",
)

st.markdown("""
<style>
    .rec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        background: #1e1e2e;
        border-radius: 8px;
        overflow: hidden;
    }
    .rec-table thead tr {
        background: #2a2a3d;
    }
    .rec-table thead th {
        padding: 0.75rem 1.1rem;
        text-align: left;
        color: #a0a0c0;
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid #3a3a52;
    }
    .rec-table tbody tr {
        border-bottom: 1px solid #2e2e42;
    }
    .rec-table tbody tr:last-child { border-bottom: none; }
    .rec-table tbody td {
        padding: 0.75rem 1.1rem;
        color: #ffffff;
        vertical-align: middle;
    }
    .rec-table .score-cell { color: #6ee7b7; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title(" Profile Recommendation System")
st.caption("Enter a member's name to find their top 5 most compatible matches.")
st.divider()

all_names = [info['name'] for info in user_mapping.values()]

col_input, col_btn = st.columns([4, 1])
with col_input:
    typed_name = st.text_input(
        "Member Name",
        placeholder="e.g. Rahul, Priya, Vikram …",
        label_visibility="collapsed",
    )
with col_btn:
    search_clicked = st.button("Find Matches", use_container_width=True, type="primary")

if typed_name:
    suggestions = [n for n in all_names if typed_name.lower() in n.lower()]
    if suggestions:
        st.caption(f"Suggestions: {', '.join(suggestions[:8])}")

if search_clicked and typed_name:

    matched = next(
        (n for n in all_names if n.lower() == typed_name.strip().lower()), None
    )

    if not matched:
        st.error(f"No member named **'{typed_name}'** found. Please check the spelling.")
    else:
        with st.spinner("Finding best matches …"):

            target_uid  = next(uid for uid, info in user_mapping.items()
                               if info['name'] == matched)
            target_info = user_mapping[target_uid]

            candidates = []
            for uid, info in user_mapping.items():
                if uid == target_uid:
                    continue
                cname = info['name']
                if cname not in similarity_df.columns:
                    continue
                candidates.append({
                    'user_id':              uid,
                    'name':                 cname,
                    'profession':           info['profession'],
                    'age':                  info['age'],
                    'experience_years':     info['experience_years'],
                    'mbti':                 info['mbti'],
                    'interests':            info['interests'],
                    'location':             info['location'],
                    'professional_summary': info['professional_summary'],
                    'similarity':           similarity_df.loc[matched, cname],
                    'age_diff':             abs(target_info['age'] - info['age']),
                    'exp_diff':             abs(target_info['experience_years'] - info['experience_years']),
                })

            cdf = pd.DataFrame(candidates)
            cdf['match_probability'] = clf.predict_proba(
                cdf[['similarity', 'age_diff', 'exp_diff']]
            )[:, 1]
            recommendations = cdf.sort_values('match_probability', ascending=False).head(5)

        st.subheader(f"👤 {matched}")
        st.write(
            f"**ID:** {target_uid}  |  "
            f"**Profession:** {target_info['profession'].title()}  |  "
            f"**Age:** {target_info['age']}  |  "
            f"**Experience:** {target_info['experience_years']} yrs  |  "
            f"**MBTI:** {target_info['mbti']}  |  "
            f"**Location:** {target_info['location'].title()}"
        )
        st.write(f"**Interests:** {target_info['interests'].title()}")
        st.write(target_info['professional_summary'].capitalize())

        st.divider()

        st.subheader("⭐ Top 5 Recommended Matches")

        rows_html = ""
        for _, rec in recommendations.iterrows():
            score = f"{rec['match_probability'] * 100:.2f}%"
            rows_html += f"""
            <tr>
                <td>{rec['user_id']}</td>
                <td>{rec['name'].title()}</td>
                <td>{rec['profession'].title()}</td>
                <td>{rec['location'].title()}</td>
                <td>{rec['mbti']}</td>
                <td class="score-cell">{score}</td>
            </tr>"""

        st.markdown(f"""
        <table class="rec-table">
            <thead>
                <tr>
                    <th>Rank ID</th>
                    <th>Full Name</th>
                    <th>Role / Profession</th>
                    <th>City Location</th>
                    <th>MBTI Type</th>
                    <th>Compatibility Score</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

elif not typed_name:
    st.info("Enter a member's name above and click **Find Matches** to get started.")