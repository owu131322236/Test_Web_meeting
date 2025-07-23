import streamlit as st
import collections
import time
import pandas as pd
import altair as alt
import datetime

from input_backend_app_preparation import MeetingAssistant, TextEmbedder, TopicDeviationDetector

st.set_page_config(
    page_title="FlowLink",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🗣️FlowLink")
st.write("会議支援ツール ---どんな人もスムーズな会議の進行を")

if 'topic_info' not in st.session_state:
    st.session_state.topic_info = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'all_utterances' not in st.session_state:
    st.session_state.all_utterances = collections.deque(maxlen=20)
if 'current_user_utterance' not in st.session_state:
    st.session_state.current_user_utterance = ""
if 'is_meeting_active' not in st.session_state:
    st.session_state.is_meeting_active = False
if 'meeting_start_time' not in st.session_state:
    st.session_state.meeting_start_time = None
if 'similarity_data' not in st.session_state:
    st.session_state.similarity_data = []


tab_titles = ['会議設定', '会議アシスタント', 'レポート']
tab1, tab2, tab3 = st.tabs(tab_titles)

with tab1:
    with st.expander("会議の最終目標", expanded=True):
        topic_thema = st.text_input("会議のテーマ", key="topic_thema")
        topic_text_input = st.text_input("会議の目的", key="topic_input_area")
        set_topic_button = st.button("決定", key="set_topic_button")
        if set_topic_button or st.session_state.topic_info is None:
            if topic_text_input:
                topic_manager = MeetingAssistant()
                meeting_id = "current_meeting"
                topic_manager.add_topic(meeting_id, topic_text_input)
                st.session_state.topic_info = topic_manager.get_topic(meeting_id)

                @st.cache_resource
                def get_detector_instance(current_topic_info_dict):
                    embedder_instance = TextEmbedder()
                    topic_embedding = embedder_instance.get_embedding(current_topic_info_dict['topic_text'])
                    detector_instance = TopicDeviationDetector(
                        topic_embedding,
                        current_topic_info_dict,
                        similarity_threshold=0.8,
                        consecutive_deviations_needed=2,
                        cooldown_period_seconds=10
                    )
                    return detector_instance

                st.session_state.detector = get_detector_instance(st.session_state.topic_info)

                st.success(f"会議の情報が設定されました: {st.session_state.topic_info['topic_text']}")
            else:
                st.warning("情報を入力してください。")

with tab2: # 会議アシスタント (チャット形式)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("会議スタート", key="start_meeting_button"):
            if st.session_state.topic_info:
                st.session_state.is_meeting_active = True
                st.session_state.meeting_start_time = time.time()
                st.session_state.all_utterances.append({"role": "system", "content": "--- 会議が開始されました ---", "timestamp": time.time()})
                st.session_state.similarity_data = []
                st.rerun()
            else:
                st.warning("先に「会議設定」タブで会議の目的を設定してください。")

    with col2:
        if st.button("会議終了", key="end_meeting_button"):
            st.session_state.is_meeting_active = False
            st.session_state.meeting_start_time = None
            st.session_state.all_utterances.append({"role": "system", "content": "--- 会議が終了されました ---", "timestamp": time.time()})
            st.rerun()

    if st.session_state.is_meeting_active:
        elapsed_time = time.time() - st.session_state.meeting_start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        st.metric(label="現在の会議時間", value=f"{minutes}分 {seconds}秒")
    else:
        st.info("会議は開始されていません。")

    st.subheader("会議チャット欄")
    st.write("⚠️本来は音声機能での入力が可能ですが、デプロイの関係上公開しているものはテキストベースでの入力が必要です。")

    for message in st.session_state.all_utterances:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_utterance = st.text_input("あなたの発言", key="user_utterance_input", value=st.session_state.current_user_utterance)
    send_utterance_button = st.button("発言を送信", key="send_utterance_button")

    if send_utterance_button and user_utterance:
        if st.session_state.detector:
            if st.session_state.is_meeting_active:
                st.session_state.all_utterances.append({"role": "user", "content": user_utterance, "timestamp": time.time()})

                is_deviated, message, similarity_score = st.session_state.detector.process_utterance(user_utterance)

                st.session_state.similarity_data.append({
                    'timestamp': time.time() - st.session_state.meeting_start_time,
                    'similarity_score': similarity_score
                })
                if message: # ここで条件分岐を追加
                    if is_deviated:
                        st.session_state.all_utterances.append({"role": "assistant", "content": f"(逸脱): {message}", "timestamp": time.time()})
                    else:
                        st.session_state.all_utterances.append({"role": "assistant", "content": message, "timestamp": time.time()})
                st.session_state.current_user_utterance = ""
                st.rerun()
            else:
                st.warning("会議を開始してから発言してください。")
        else:
            st.warning("先に「会議設定」タブで会議の目的を設定してください。")
    elif send_utterance_button and not user_utterance:
        st.warning("発言を入力してください。")


with tab3: # レポート (テキストベースの履歴)
    st.subheader("会議レポート✍️")
    st.write("ここに会議の要約、テーマ逸脱の統計、詳細な全発言ログなどが表示されます。")

    if st.session_state.similarity_data:
        st.markdown("テーマ類似度推移")
        df_similarity = pd.DataFrame(st.session_state.similarity_data)
        df_similarity['時間 (秒)'] = df_similarity['timestamp']
        df_similarity['類似度'] = df_similarity['similarity_score']

        chart = alt.Chart(df_similarity).mark_line().encode(
            x=alt.X('時間 (秒):Q', axis=alt.Axis(title='会議開始からの時間 (秒)')),
            y=alt.Y('類似度:Q', axis=alt.Axis(title='テーマとの類似度', format=".2f"), scale=alt.Scale(domain=[0, 1])),
            tooltip=['時間 (秒)', '類似度']
        ).properties(
            title='発言とテーマの類似度推移'
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("会議を開始し、発言するとテーマ類似度推移が表示されます。")

    st.markdown("---")
    if st.session_state.all_utterances:
        st.markdown("全発言ログ")
        # レポートタブではテキストベースのログとして表示
        for i, message_obj in enumerate(list(st.session_state.all_utterances)):
            role_display = ""
            if message_obj["role"] == "user":
                role_display = "あなた"
            elif message_obj["role"] == "assistant":
                role_display = "AIアシスタント"
            elif message_obj["role"] == "system":
                role_display = "システム"

            timestamp_str = datetime.datetime.fromtimestamp(message_obj["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')

            st.text(f"[{timestamp_str}] {role_display}: {message_obj['content']}")
