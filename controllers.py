import streamlit as st


class PageController:
    def __init__(self):
        if "page" not in st.session_state:
            st.session_state.page = "Home"

    def set_page(self, page):
        st.session_state.page = page
        st.rerun()

    def get_current_page(self):
        return st.session_state.page


class SessionController:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        if "page" not in st.session_state:
            st.session_state.page = "Home"

        if "df" not in st.session_state:
            st.session_state.df = None

        if "snow_triggered" not in st.session_state:
            st.session_state.snow_triggered = False

        if "plot_color_choice" not in st.session_state:
            st.session_state.plot_color_choice = "QuimioAnalytics (Custom)"

        if "standardized" not in st.session_state:
            st.session_state.standardized = False

        if "pca_ready" not in st.session_state:
            st.session_state.pca_ready = False

        if "anova_done" not in st.session_state:
            st.session_state.anova_done = False

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "clear_input" not in st.session_state:
            st.session_state.clear_input = False
